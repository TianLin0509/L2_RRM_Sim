"""多小区仿真引擎

在单小区引擎基础上扩展:
- 多小区拓扑 (HexGrid + wrap-around)
- 小区间干扰
- UE 小区关联 (max RSRP)
- 各小区独立调度
"""

import time
import numpy as np
from ..config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig, ChannelConfig
)
from .data_types import SlotContext, UEState, ChannelState, SlotResult, SchedulingDecision
from .resource_grid import ResourceGrid
from .topology import HexGridTopology
from .nr_constants import SLOTS_PER_FRAME, SLOTS_PER_SUBFRAME
from ..channel.pathloss_models import PATHLOSS_MODELS
from ..channel.interference_model import InterCellInterference
from ..channel.statistical_channel import StatisticalChannel
from ..scheduler.pf_scheduler import PFSchedulerSUMIMO
from ..scheduler.rank_adaptation import RankAdapter
from ..link_adaptation.bler_tables import BLERTableManager
from ..link_adaptation.effective_sinr import EESM
from ..link_adaptation.illa import ILLA
from ..link_adaptation.olla import OLLA
from ..link_adaptation.phy_abstraction import PHYAbstraction
from ..traffic.full_buffer import FullBufferTraffic
from ..traffic.buffer_manager import BufferManager
from ..kpi.kpi_collector import KPICollector
from ..kpi.kpi_reporter import KPIReporter
from ..utils.math_utils import db_to_linear, dbm_to_watt, linear_to_db
from ..utils.nr_utils import get_spectral_efficiency
from ..utils.random_utils import SimRNG


class MultiCellSimulationEngine:
    """多小区仿真引擎

    管理多个小区，计算小区间干扰，
    各小区独立调度但干扰耦合。
    """

    def __init__(self, config: dict,
                 num_rings: int = 1,
                 num_ue_per_cell: int = 10,
                 csi_delay_slots: int = 0,
                 ici_load_factor: float = 1.0):
        """
        Args:
            config: 配置字典
            num_rings: 六角网格环数 (0=1小区, 1=7站21小区, 2=19站57小区)
            num_ue_per_cell: 每小区 UE 数
            csi_delay_slots: CSI 反馈延迟 (slots)
            ici_load_factor: 邻小区负载因子
        """
        self.sim_config: SimConfig = config['sim']
        self.carrier_config: CarrierConfig = config['carrier']
        self.cell_config: CellConfig = config['cell']
        self.ue_config: UEConfig = config['ue']
        self.scheduler_config: SchedulerConfig = config['scheduler']
        self.la_config: LinkAdaptationConfig = config['link_adaptation']
        self.traffic_config: TrafficConfig = config['traffic']
        self.channel_config: ChannelConfig = config['channel']

        self.rng = SimRNG(self.sim_config.random_seed)
        self.resource_grid = ResourceGrid(self.carrier_config)
        self.num_ue_per_cell = num_ue_per_cell

        # 拓扑
        self.topology = HexGridTopology(
            num_rings=num_rings,
            isd=self.cell_config.cell_radius_m * 2,
            cell_height=self.cell_config.height_m,
            wraparound=True
        )
        self.num_cells = self.topology.num_cells

        # 在各小区内撒 UE
        ue_positions = self.topology.drop_ues(
            num_ue_per_cell,
            min_distance=self.ue_config.min_distance_m,
            max_distance=self.cell_config.cell_radius_m,
            ue_height=self.ue_config.height_m,
            rng=self.rng.general,
        )

        # 初始化每小区的 UE 状态
        self.cell_ue_states = {}
        ue_counter = 0
        for cell_idx in range(self.num_cells):
            ues = []
            for ue_idx in range(num_ue_per_cell):
                ue = UEState(
                    ue_id=ue_counter,
                    position=ue_positions[cell_idx, ue_idx],
                )
                ues.append(ue)
                ue_counter += 1
            self.cell_ue_states[cell_idx] = ues

        self.total_ue = ue_counter

        # 干扰计算器
        self.ici = InterCellInterference(
            num_cells=self.num_cells,
            total_power_dbm=self.cell_config.total_power_dbm,
            num_prb=self.carrier_config.num_prb,
            carrier_freq_ghz=self.carrier_config.carrier_freq_ghz,
            cell_height=self.cell_config.height_m,
            scenario=self.cell_config.scenario,
            noise_figure_db=self.cell_config.noise_figure_db,
            scs_khz=self.carrier_config.subcarrier_spacing,
            load_factor=ici_load_factor,
        )

        # 共享模块
        self.bler_table = BLERTableManager()
        self.eesm = EESM()
        self.rank_adapter = RankAdapter(
            max_rank=self.cell_config.max_layers,
            fixed_rank=1
        )
        self.traffic = FullBufferTraffic()

        # 每小区独立的调度器、OLLA
        self.cell_schedulers = {}
        self.cell_ollas = {}
        self.cell_kpis = {}

        for cell_idx in range(self.num_cells):
            illa = ILLA(self.bler_table,
                        bler_target=self.la_config.bler_target,
                        mcs_table_index=self.la_config.mcs_table_index,
                        num_re_per_prb=self.resource_grid.num_re_per_prb)

            self.cell_schedulers[cell_idx] = PFSchedulerSUMIMO(
                num_ue=num_ue_per_cell,
                num_prb=self.carrier_config.num_prb,
                num_re_per_prb=self.resource_grid.num_re_per_prb,
                mcs_table_index=self.la_config.mcs_table_index,
                beta=self.scheduler_config.beta,
            )

            self.cell_ollas[cell_idx] = OLLA(
                num_ue=num_ue_per_cell,
                illa=illa,
                bler_target=self.la_config.bler_target,
                delta_up=self.la_config.olla_delta_up,
            )

            self.cell_kpis[cell_idx] = KPICollector(
                self.sim_config.num_slots,
                num_ue_per_cell,
                warmup_slots=self.sim_config.warmup_slots
            )

        # PHY 抽象
        self.phy_abs = PHYAbstraction(
            self.bler_table, self.eesm,
            num_re_per_prb=self.resource_grid.num_re_per_prb,
            mcs_table_index=self.la_config.mcs_table_index,
            rng=self.rng.phy,
        )

        # Per-cell HARQ feedback
        self._cell_last_harq = {
            c: np.ones(num_ue_per_cell, dtype=bool) for c in range(self.num_cells)
        }
        self._cell_last_scheduled = {
            c: np.zeros(num_ue_per_cell, dtype=bool) for c in range(self.num_cells)
        }

        # 预计算干扰
        self._precompute_interference()

    def _precompute_interference(self):
        """预计算每个 UE 的小区间干扰"""
        self._ue_interference = {}
        for cell_idx in range(self.num_cells):
            for ue_idx, ue in enumerate(self.cell_ue_states[cell_idx]):
                intf = self.ici.compute_interference(
                    ue.position, cell_idx,
                    self.topology.cell_positions,
                    self.topology
                )
                self._ue_interference[(cell_idx, ue_idx)] = np.mean(intf)

    def run(self) -> dict:
        """运行多小区仿真"""
        num_slots = self.sim_config.num_slots
        print(f"=== 多小区 L2 RRM 仿真 ===")
        print(f"  Cells: {self.num_cells}, UEs/cell: {self.num_ue_per_cell}, "
              f"Total UEs: {self.total_ue}")
        print(f"  PRBs: {self.carrier_config.num_prb}, "
              f"BW: {self.carrier_config.bandwidth_mhz}MHz")

        t_start = time.time()
        pl_func = PATHLOSS_MODELS.get(self.cell_config.scenario)

        for slot_idx in range(num_slots):
            slot_ctx = SlotContext(
                slot_idx=slot_idx,
                time_s=slot_idx * self.carrier_config.slot_duration_s,
            )

            # 每小区独立处理
            for cell_idx in range(self.num_cells):
                result = self._run_cell_slot(cell_idx, slot_ctx, pl_func)
                self.cell_kpis[cell_idx].collect(slot_idx, result)

            if (slot_idx + 1) % 1000 == 0 or slot_idx == num_slots - 1:
                elapsed = time.time() - t_start
                speed = (slot_idx + 1) / elapsed
                print(f"  Slot {slot_idx + 1}/{num_slots} "
                      f"({speed:.0f} slots/s)")

        elapsed_total = time.time() - t_start
        print(f"=== 仿真完成 ({elapsed_total:.1f}s) ===\n")

        return self._aggregate_report()

    def _run_cell_slot(self, cell_idx: int, slot_ctx: SlotContext,
                       pl_func) -> SlotResult:
        """运行单小区单 slot (简化版，含干扰)"""
        ue_states = self.cell_ue_states[cell_idx]
        num_ue = len(ue_states)
        num_prb = self.carrier_config.num_prb
        scheduler = self.cell_schedulers[cell_idx]
        olla = self.cell_ollas[cell_idx]

        # 流量
        self.traffic.generate(slot_ctx, ue_states)

        # 信道 + 干扰 → SINR
        sinr_per_prb = np.zeros((num_ue, self.cell_config.max_layers, num_prb))
        sinr_eff_db = np.zeros(num_ue)

        tx_power_per_prb = dbm_to_watt(self.cell_config.total_power_dbm) / num_prb
        bw_prb_hz = self.carrier_config.subcarrier_spacing * 1e3 * 12
        noise_power = (1.380649e-23 * 290 * bw_prb_hz
                       * db_to_linear(self.cell_config.noise_figure_db))

        for ue_idx, ue in enumerate(ue_states):
            d_2d = self.topology.compute_distance_2d(ue.position, cell_idx)
            d_2d = max(d_2d, 10.0)

            is_los = d_2d < 100.0
            pl_db = pl_func(d_2d, self.cell_config.height_m,
                            ue.position[2], self.carrier_config.carrier_freq_ghz,
                            is_los)
            pl_linear = db_to_linear(pl_db)

            # 快衰落
            fading = self.rng.channel.exponential(
                1.0, (self.cell_config.max_layers, num_prb)
            )

            # 干扰
            intf = self._ue_interference.get((cell_idx, ue_idx), 0.0)

            # SINR = signal / (noise + interference)
            sinr_per_prb[ue_idx] = (
                tx_power_per_prb * fading / (pl_linear * (noise_power + intf))
            )
            sinr_eff_db[ue_idx] = linear_to_db(np.mean(sinr_per_prb[ue_idx, 0, :]))

        # Rank
        ue_rank = np.ones(num_ue, dtype=np.int32)

        # OLLA
        olla.update_offsets_batch(
            self._cell_last_harq[cell_idx],
            self._cell_last_scheduled[cell_idx]
        )
        est_prbs = np.full(num_ue, max(num_prb // num_ue, 1), dtype=np.int32)
        mcs_indices = olla.select_mcs(sinr_eff_db, est_prbs, ue_rank)

        # PF 调度: 使用 Shannon 容量 (含 rank)
        achievable = np.zeros((num_ue, num_prb))
        re_per_prb = self.resource_grid.num_re_per_prb
        for ue in range(num_ue):
            r = int(ue_rank[ue])
            sinr_prb = sinr_per_prb[ue, :r, :]
            capacity = np.sum(np.log2(1.0 + np.maximum(sinr_prb, 0)), axis=0)
            achievable[ue, :] = capacity * re_per_prb

        ue_buffer = np.array([u.buffer_bytes for u in ue_states])
        sched = scheduler.schedule(
            slot_ctx, ue_states, None,
            achievable, ue_buffer, mcs_indices, ue_rank
        )

        # PHY
        phy_results = self.phy_abs.evaluate_batch(
            sinr_per_prb, mcs_indices,
            sched.ue_num_prbs, ue_rank, sched.prb_assignment
        )

        # 更新状态
        self._cell_last_harq[cell_idx] = phy_results['is_success'].copy()
        self._cell_last_scheduled[cell_idx] = sched.ue_num_prbs > 0
        scheduler.update_throughput_history(phy_results['decoded_bits'].astype(float))

        # BufferManager
        for ue_idx, ue in enumerate(ue_states):
            tx_bytes = int(phy_results['decoded_bits'][ue_idx]) // 8
            ue.buffer_bytes = max(0, ue.buffer_bytes - tx_bytes)

        slot_duration_s = self.carrier_config.slot_duration_s
        return SlotResult(
            slot_idx=slot_ctx.slot_idx,
            ue_decoded_bits=phy_results['decoded_bits'],
            ue_bler=phy_results['tbler'],
            ue_tb_success=phy_results['is_success'],
            ue_mcs=mcs_indices,
            ue_rank=ue_rank,
            ue_sinr_eff_db=sinr_eff_db,
            ue_throughput_inst=phy_results['decoded_bits'].astype(float) / slot_duration_s,
            scheduling_decision=sched,
        )

    def _aggregate_report(self) -> dict:
        """汇总所有小区 KPI"""
        all_cell_tp = []
        all_ue_tp = []

        reporter = KPIReporter(
            self.cell_kpis[0], self.carrier_config
        )
        slot_duration_s = self.carrier_config.slot_duration_s

        for cell_idx in range(self.num_cells):
            kpi = self.cell_kpis[cell_idx]
            s = kpi.get_valid_range()
            if s.stop <= s.start:
                continue

            cell_bits = kpi.cell_throughput_bits[s]
            cell_tp = np.mean(cell_bits / slot_duration_s / 1e6)
            all_cell_tp.append(cell_tp)

            ue_bits = kpi.ue_throughput_bits[s]
            ue_avg = np.mean(ue_bits, axis=0) / slot_duration_s / 1e6
            all_ue_tp.extend(ue_avg.tolist())

        all_ue_tp = np.array(all_ue_tp)
        all_cell_tp = np.array(all_cell_tp)

        report = {
            'num_cells': self.num_cells,
            'total_ue': self.total_ue,
            'avg_cell_throughput_mbps': float(np.mean(all_cell_tp)) if len(all_cell_tp) > 0 else 0,
            'cell_edge_throughput_mbps': float(np.percentile(all_ue_tp, 5)) if len(all_ue_tp) > 0 else 0,
            'avg_ue_throughput_mbps': float(np.mean(all_ue_tp)) if len(all_ue_tp) > 0 else 0,
            'median_ue_throughput_mbps': float(np.median(all_ue_tp)) if len(all_ue_tp) > 0 else 0,
            'spectral_efficiency': float(np.mean(all_cell_tp)) * 1e6 / (self.carrier_config.bandwidth_mhz * 1e6) if len(all_cell_tp) > 0 else 0,
        }

        print("=" * 60)
        print("  多小区 KPI 报告")
        print("=" * 60)
        print(f"  小区数: {report['num_cells']}, 总 UE 数: {report['total_ue']}")
        print(f"  小区平均吞吐量: {report['avg_cell_throughput_mbps']:.2f} Mbps")
        print(f"  小区边缘吞吐量 (5%): {report['cell_edge_throughput_mbps']:.2f} Mbps")
        print(f"  UE 平均吞吐量: {report['avg_ue_throughput_mbps']:.2f} Mbps")
        print(f"  频谱效率: {report['spectral_efficiency']:.2f} bps/Hz")
        print("=" * 60)

        return report

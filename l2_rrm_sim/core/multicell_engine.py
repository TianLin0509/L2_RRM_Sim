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
from ..scheduler.pf_scheduler import PFSchedulerSUMIMO
from ..traffic.full_buffer import FullBufferTraffic
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
        self.traffic = FullBufferTraffic()

        # PHY 层: Sionna 或 legacy
        try:
            from ..link_adaptation.sionna_phy import SionnaPHY
            self._use_sionna_phy = True
        except ImportError:
            self._use_sionna_phy = False

        # 每小区独立的调度器、PHY
        self.cell_schedulers = {}
        self.cell_phy = {}
        self.cell_kpis = {}

        for cell_idx in range(self.num_cells):
            self.cell_schedulers[cell_idx] = PFSchedulerSUMIMO(
                num_ue=num_ue_per_cell,
                num_prb=self.carrier_config.num_prb,
                num_re_per_prb=self.resource_grid.num_re_per_prb,
                mcs_table_index=self.la_config.mcs_table_index,
                beta=self.scheduler_config.beta,
            )

            if self._use_sionna_phy:
                self.cell_phy[cell_idx] = SionnaPHY(
                    num_ue=num_ue_per_cell,
                    bler_target=self.la_config.bler_target,
                    delta_up=self.la_config.olla_delta_up,
                    mcs_table_index=self.la_config.mcs_table_index,
                )
            else:
                from ..link_adaptation.bler_tables import BLERTableManager
                from ..link_adaptation.effective_sinr import EESM
                from ..link_adaptation.illa import ILLA
                from ..link_adaptation.olla import OLLA
                bt = BLERTableManager(); eesm = EESM()
                illa = ILLA(bt, self.la_config.bler_target,
                            self.la_config.mcs_table_index,
                            self.resource_grid.num_re_per_prb)
                self.cell_phy[cell_idx] = {
                    'olla': OLLA(num_ue_per_cell, illa, self.la_config.bler_target,
                                 self.la_config.olla_delta_up),
                    'eesm': eesm,
                }

            self.cell_kpis[cell_idx] = KPICollector(
                self.sim_config.num_slots,
                num_ue_per_cell,
                warmup_slots=self.sim_config.warmup_slots
            )

        # Legacy PHY 抽象 (fallback)
        if not self._use_sionna_phy:
            from ..link_adaptation.bler_tables import BLERTableManager
            from ..link_adaptation.effective_sinr import EESM
            from ..link_adaptation.phy_abstraction import PHYAbstraction
            self.phy_abs = PHYAbstraction(
                BLERTableManager(), EESM(),
                self.resource_grid.num_re_per_prb,
                self.la_config.mcs_table_index,
                self.rng.phy,
            )

        # Per-cell HARQ feedback (Sionna 用 int32: 1=ACK, 0=NACK, -1=unscheduled)
        self._cell_last_harq = {
            c: np.ones(num_ue_per_cell, dtype=np.int32) for c in range(self.num_cells)
        }
        self._cell_last_scheduled = {
            c: np.zeros(num_ue_per_cell, dtype=bool) for c in range(self.num_cells)
        }

        # 预计算干扰
        self._precompute_interference()

    def _precompute_interference(self):
        """预计算路径损耗缓存 + 静态干扰 (第一次)"""
        self.ici.precompute_pathloss(
            self.cell_ue_states, self.topology, self.topology.cell_positions
        )
        # 静态干扰 (回退用)
        self._ue_interference = {}
        for cell_idx in range(self.num_cells):
            for ue_idx in range(len(self.cell_ue_states[cell_idx])):
                ue_key = (cell_idx, ue_idx)
                self._ue_interference[ue_key] = self.ici.compute_static_interference(
                    ue_key, cell_idx
                )
        # 上一 slot 各小区的 PRB 负载 (动态干扰用)
        self._cell_prb_loads = {
            c: np.full(self.carrier_config.num_prb, self.ici.load_factor)
            for c in range(self.num_cells)
        }

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
                result, buf_before, buf_after = self._run_cell_slot(cell_idx, slot_ctx, pl_func)
                self.cell_kpis[cell_idx].collect(slot_idx, result, buf_before, buf_after)

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
        """运行单小区单 slot (含干扰, 支持 Sionna PHY)"""
        ue_states = self.cell_ue_states[cell_idx]
        num_ue = len(ue_states)
        num_prb = self.carrier_config.num_prb
        re_per_prb = self.resource_grid.num_re_per_prb
        scheduler = self.cell_schedulers[cell_idx]

        # 流量
        self.traffic.generate(slot_ctx, ue_states)
        buf_before_tx = np.array([u.buffer_bytes for u in ue_states], dtype=np.int64)

        # 信道 + 干扰 → SINR
        sinr_per_prb = np.zeros((num_ue, self.cell_config.max_layers, num_prb))
        tx_power_per_prb = dbm_to_watt(self.cell_config.total_power_dbm) / num_prb
        bw_prb_hz = self.carrier_config.subcarrier_spacing * 1e3 * 12
        noise_power = (1.380649e-23 * 290 * bw_prb_hz
                       * db_to_linear(self.cell_config.noise_figure_db))

        for ue_idx, ue in enumerate(ue_states):
            d_2d = self.topology.compute_distance_2d(ue.position, cell_idx)
            d_2d = max(d_2d, 10.0)
            is_los = d_2d < 100.0
            pl_db = pl_func(d_2d, self.cell_config.height_m,
                            ue.position[2], self.carrier_config.carrier_freq_ghz, is_los)
            pl_linear = db_to_linear(pl_db)
            fading = self.rng.channel.exponential(1.0, (self.cell_config.max_layers, num_prb))
            # 动态干扰: 基于上一 slot 各邻小区的 PRB 调度负载
            ue_key = (cell_idx, ue_idx)
            intf_per_prb = self.ici.compute_dynamic_interference(
                ue_key, cell_idx, self._cell_prb_loads
            )
            sinr_per_prb[ue_idx] = (
                tx_power_per_prb * fading / (pl_linear * (noise_power + intf_per_prb[np.newaxis, :]))
            )

        ue_rank = np.ones(num_ue, dtype=np.int32)

        # MCS 选择 + PHY 评估
        if self._use_sionna_phy:
            phy = self.cell_phy[cell_idx]
            est_re = np.full(num_ue, re_per_prb * max(num_prb // num_ue, 1), dtype=np.int32)
            mcs_indices = phy.select_mcs(
                num_allocated_re=est_re,
                harq_feedback=self._cell_last_harq[cell_idx],
                scheduled_mask=self._cell_last_scheduled[cell_idx],
            )
        else:
            phy_dict = self.cell_phy[cell_idx]
            phy_dict['olla'].update_offsets_batch(
                self._cell_last_harq[cell_idx].astype(bool),
                self._cell_last_scheduled[cell_idx])
            sinr_eff_db = np.array([linear_to_db(np.mean(sinr_per_prb[u, 0, :]))
                                    for u in range(num_ue)])
            est_prbs = np.full(num_ue, max(num_prb // num_ue, 1), dtype=np.int32)
            mcs_indices = phy_dict['olla'].select_mcs(sinr_eff_db, est_prbs, ue_rank)

        # PF 调度
        achievable = np.zeros((num_ue, num_prb))
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

        # PHY 评估
        if self._use_sionna_phy:
            ue_num_re = sched.ue_num_prbs * re_per_prb
            phy_results = phy.evaluate(
                mcs_indices=mcs_indices,
                sinr_per_re=sinr_per_prb,
                num_allocated_re=ue_num_re.astype(np.int32),
                prb_assignment=sched.prb_assignment,
            )
        else:
            phy_results = self.phy_abs.evaluate_batch(
                sinr_per_prb, mcs_indices,
                sched.ue_num_prbs, ue_rank, sched.prb_assignment
            )

        # 更新状态
        if self._use_sionna_phy:
            self._cell_last_harq[cell_idx] = phy_results['harq_feedback'].copy()
        else:
            self._cell_last_harq[cell_idx] = np.where(
                phy_results['is_success'], 1, 0).astype(np.int32)
        self._cell_last_scheduled[cell_idx] = sched.ue_num_prbs > 0
        scheduler.update_throughput_history(phy_results['decoded_bits'].astype(float))

        # 更新本小区 PRB 负载 (供下一 slot 其他小区的干扰计算)
        prb_load = np.zeros(num_prb)
        prb_load[sched.prb_assignment >= 0] = 1.0
        self._cell_prb_loads[cell_idx] = prb_load

        for ue_idx, ue in enumerate(ue_states):
            tx_bytes = int(phy_results['decoded_bits'][ue_idx]) // 8
            ue.buffer_bytes = max(0, ue.buffer_bytes - tx_bytes)
        buf_after_tx = np.array([u.buffer_bytes for u in ue_states], dtype=np.int64)

        sinr_eff_db = np.zeros(num_ue)
        if 'sinr_eff' in phy_results and phy_results['sinr_eff'] is not None:
            sinr_eff_db = np.where(phy_results['sinr_eff'] > 0,
                                    linear_to_db(phy_results['sinr_eff']), -30.0)

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
        ), buf_before_tx, buf_after_tx

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

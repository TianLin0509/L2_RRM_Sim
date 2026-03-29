"""仿真主循环引擎

TTI (slot) 级仿真循环，协调所有子模块。
"""

import time
import numpy as np
from ..config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig, ChannelConfig
)
from .data_types import SlotContext, UEState, SlotResult
from .resource_grid import ResourceGrid
from .nr_constants import SLOTS_PER_FRAME, SLOTS_PER_SUBFRAME
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
from ..utils.math_utils import linear_to_db
from ..utils.nr_utils import get_spectral_efficiency
from ..utils.random_utils import SimRNG


class SimulationEngine:
    """仿真主引擎

    协调信道、调度、链路自适应、流量、KPI 各模块，
    按 TTI 粒度运行仿真循环。
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典，包含各配置对象:
                sim, carrier, cell, ue, scheduler,
                link_adaptation, traffic, channel
        """
        self.sim_config: SimConfig = config['sim']
        self.carrier_config: CarrierConfig = config['carrier']
        self.cell_config: CellConfig = config['cell']
        self.ue_config: UEConfig = config['ue']
        self.scheduler_config: SchedulerConfig = config['scheduler']
        self.la_config: LinkAdaptationConfig = config['link_adaptation']
        self.traffic_config: TrafficConfig = config['traffic']
        self.channel_config: ChannelConfig = config['channel']

        # 随机数
        self.rng = SimRNG(self.sim_config.random_seed)

        # 资源网格
        self.resource_grid = ResourceGrid(self.carrier_config)

        # 初始化 UE
        self.ue_states = self._init_ue_states()
        self.num_ue = len(self.ue_states)

        # 信道 (支持 sionna/statistical 两种模式)
        channel_type = self.channel_config.type.lower()
        if channel_type == 'sionna':
            from ..channel.sionna_channel import SionnaChannel
            self.channel = SionnaChannel(
                self.cell_config, self.carrier_config, self.channel_config
            )
        else:
            self.channel = StatisticalChannel(
                self.cell_config, self.carrier_config,
                self.channel_config, self.rng.channel
            )
        self.channel.initialize(
            self.cell_config, self.carrier_config, self.ue_states
        )

        # Rank 自适应
        use_fixed_rank = (channel_type != 'sionna')
        self.rank_adapter = RankAdapter(
            max_rank=self.cell_config.max_layers,
            fixed_rank=1 if use_fixed_rank else None,
        )

        # 链路自适应
        self.bler_table = BLERTableManager()
        self.eesm = EESM()
        self.illa = ILLA(
            self.bler_table,
            bler_target=self.la_config.bler_target,
            mcs_table_index=self.la_config.mcs_table_index,
            num_re_per_prb=self.resource_grid.num_re_per_prb
        )
        self.olla = OLLA(
            num_ue=self.num_ue,
            illa=self.illa,
            bler_target=self.la_config.bler_target,
            delta_up=self.la_config.olla_delta_up,
            offset_min=self.la_config.olla_offset_min,
            offset_max=self.la_config.olla_offset_max,
        )
        self.phy_abs = PHYAbstraction(
            self.bler_table, self.eesm,
            num_re_per_prb=self.resource_grid.num_re_per_prb,
            mcs_table_index=self.la_config.mcs_table_index,
            rng=self.rng.phy,
        )

        # 调度器
        self.scheduler = PFSchedulerSUMIMO(
            num_ue=self.num_ue,
            num_prb=self.carrier_config.num_prb,
            num_re_per_prb=self.resource_grid.num_re_per_prb,
            mcs_table_index=self.la_config.mcs_table_index,
            beta=self.scheduler_config.beta,
        )

        # 流量模型
        self.traffic = FullBufferTraffic()
        self.buffer_mgr = BufferManager(self.num_ue)

        # KPI
        self.kpi = KPICollector(
            self.sim_config.num_slots, self.num_ue,
            warmup_slots=self.sim_config.warmup_slots
        )

        # HARQ 反馈缓存 (上一 TTI)
        self._last_harq_ack = np.ones(self.num_ue, dtype=bool)
        self._last_scheduled_mask = np.zeros(self.num_ue, dtype=bool)

    def _init_ue_states(self) -> list:
        """初始化 UE 位置和状态"""
        ue_states = []
        num_ue = self.ue_config.num_ue
        r_min = self.ue_config.min_distance_m
        r_max = self.ue_config.max_distance_m

        for i in range(num_ue):
            # 均匀分布在小区内 (极坐标)
            r = np.sqrt(self.rng.general.uniform(r_min**2, r_max**2))
            theta = self.rng.general.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = self.ue_config.height_m

            # 速度 (随机方向)
            speed = self.ue_config.speed_kmh / 3.6  # m/s
            v_theta = self.rng.general.uniform(0, 2 * np.pi)
            vx = speed * np.cos(v_theta)
            vy = speed * np.sin(v_theta)

            ue = UEState(
                ue_id=i,
                position=np.array([x, y, z]),
                velocity=np.array([vx, vy, 0.0]),
            )
            ue_states.append(ue)

        return ue_states

    def _make_slot_context(self, slot_idx: int) -> SlotContext:
        """构造 slot 上下文"""
        mu = self.carrier_config.mu
        slots_per_frame = SLOTS_PER_FRAME[mu]
        slots_per_subframe = SLOTS_PER_SUBFRAME[mu]

        return SlotContext(
            slot_idx=slot_idx,
            time_s=slot_idx * self.carrier_config.slot_duration_s,
            frame_idx=slot_idx // slots_per_frame,
            subframe_idx=(slot_idx % slots_per_frame) // slots_per_subframe,
            slot_in_subframe=slot_idx % slots_per_subframe,
        )

    def run(self) -> dict:
        """运行完整仿真"""
        num_slots = self.sim_config.num_slots
        print(f"=== L2 RRM 仿真开始 ===")
        print(f"  Slots: {num_slots}, UEs: {self.num_ue}, "
              f"PRBs: {self.carrier_config.num_prb}, "
              f"BW: {self.carrier_config.bandwidth_mhz}MHz@{self.carrier_config.subcarrier_spacing}kHz")
        print(f"  Scenario: {self.cell_config.scenario}, "
              f"Power: {self.cell_config.total_power_dbm}dBm, "
              f"BLER target: {self.la_config.bler_target}")

        t_start = time.time()

        for slot_idx in range(num_slots):
            slot_result = self.run_slot(slot_idx)
            self.kpi.collect(slot_idx, slot_result, self._buf_after_traffic)

            if (slot_idx + 1) % 1000 == 0 or slot_idx == num_slots - 1:
                elapsed = time.time() - t_start
                speed = (slot_idx + 1) / elapsed
                print(f"  Slot {slot_idx + 1}/{num_slots} "
                      f"({speed:.0f} slots/s, {elapsed:.1f}s elapsed)")

        elapsed_total = time.time() - t_start
        print(f"=== 仿真完成 ({elapsed_total:.1f}s) ===\n")

        # 生成报告
        reporter = KPIReporter(self.kpi, self.carrier_config)
        report = reporter.report()
        reporter.print_report(report)
        return report

    def run_slot(self, slot_idx: int) -> SlotResult:
        """运行单个 slot"""
        slot_ctx = self._make_slot_context(slot_idx)
        num_ue = self.num_ue
        num_prb = self.carrier_config.num_prb

        # 1. 流量生成
        self.traffic.generate(slot_ctx, self.ue_states)

        # 记录流量生成后的 buffer 状态 (体验速率计算用)
        self._buf_after_traffic = np.array(
            [ue.buffer_bytes for ue in self.ue_states], dtype=np.int64
        )

        # 2. 信道更新
        channel_state = self.channel.update(slot_ctx, self.ue_states)

        # 3. Rank 选择
        ue_rank = self.rank_adapter.select_rank_batch(
            channel_state.sinr_per_prb, num_ue
        )
        for ue_idx, ue in enumerate(self.ue_states):
            ue.selected_rank = int(ue_rank[ue_idx])

        # 4. OLLA: 只对上一 TTI 被调度的 UE 更新偏移
        self.olla.update_offsets_batch(
            self._last_harq_ack, self._last_scheduled_mask
        )

        # 5. MCS 选择
        sinr_eff_db = np.zeros(num_ue)
        est_prbs = np.full(num_ue, max(num_prb // num_ue, 1), dtype=np.int32)

        # EESM: 使用 MCS 0 的 beta (保守估计, 避免高估 SINR)
        for ue in range(num_ue):
            r = int(ue_rank[ue])
            sinr_eff_db[ue] = self.eesm.compute(
                channel_state.sinr_per_prb[ue, :r, :],
                mcs_index=0,
                table_index=self.la_config.mcs_table_index
            )

        # OLLA 选择 MCS (内部 ILLA 会用 BLER 表逐 MCS 查找)
        mcs_indices = self.olla.select_mcs(sinr_eff_db, est_prbs, ue_rank)

        # 6. 计算 per-PRB 可达速率 (用于 PF metric)
        #    使用 Shannon 容量作为 per-PRB 速率估计, 避免与 MCS 耦合
        achievable_rate_per_prb = np.zeros((num_ue, num_prb))
        re_per_prb = self.resource_grid.num_re_per_prb
        for ue in range(num_ue):
            r = int(ue_rank[ue])
            # per-PRB Shannon 容量 (考虑 rank)
            sinr_prb = channel_state.sinr_per_prb[ue, :r, :]  # (rank, num_prb)
            # 各层的 log2(1+SINR) 求和, 乘以 RE 数
            capacity_per_prb = np.sum(np.log2(1.0 + np.maximum(sinr_prb, 0)), axis=0)
            achievable_rate_per_prb[ue, :] = capacity_per_prb * re_per_prb

        # 7. PF 调度
        ue_buffer = np.array([ue.buffer_bytes for ue in self.ue_states])
        sched = self.scheduler.schedule(
            slot_ctx, self.ue_states, channel_state,
            achievable_rate_per_prb, ue_buffer,
            mcs_indices, ue_rank
        )

        # 8. 被调度 UE: 用实际分配 PRB 重新计算精确 EESM
        for ue in range(num_ue):
            if sched.ue_num_prbs[ue] > 0:
                r = int(ue_rank[ue])
                ue_prbs = (sched.prb_assignment == ue)
                sinr_ue = channel_state.sinr_per_prb[ue, :r, :][:, ue_prbs]
                sinr_eff_db[ue] = self.eesm.compute(
                    sinr_ue, int(mcs_indices[ue]),
                    self.la_config.mcs_table_index
                )

        # 9. PHY 抽象: 评估传输结果
        phy_results = self.phy_abs.evaluate_batch(
            channel_state.sinr_per_prb,
            mcs_indices, sched.ue_num_prbs, ue_rank,
            sched.prb_assignment
        )

        # 10. 更新 HARQ 反馈和 UE 状态
        self._last_harq_ack = phy_results['is_success'].copy()
        self._last_scheduled_mask = sched.ue_num_prbs > 0
        for ue_idx, ue in enumerate(self.ue_states):
            ue.last_harq_ack = bool(self._last_harq_ack[ue_idx])
            ue.sinr_eff_db_last = float(sinr_eff_db[ue_idx])

        # 11. 更新缓冲区
        self.buffer_mgr.dequeue(self.ue_states, phy_results['decoded_bits'])

        # 12. 更新调度器吞吐量历史
        self.scheduler.update_throughput_history(
            phy_results['decoded_bits'].astype(np.float64)
        )

        # 构造 SlotResult
        slot_duration_s = self.carrier_config.slot_duration_s
        throughput_inst = phy_results['decoded_bits'].astype(np.float64) / slot_duration_s

        return SlotResult(
            slot_idx=slot_idx,
            ue_decoded_bits=phy_results['decoded_bits'],
            ue_bler=phy_results['tbler'],
            ue_tb_success=phy_results['is_success'],
            ue_mcs=mcs_indices,
            ue_rank=ue_rank,
            ue_sinr_eff_db=sinr_eff_db,
            ue_throughput_inst=throughput_inst,
            scheduling_decision=sched,
        )

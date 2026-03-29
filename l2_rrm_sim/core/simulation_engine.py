"""仿真主循环引擎

TTI (slot) 级仿真循环，协调所有子模块。
"""

import time
import numpy as np
from ..config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig, ChannelConfig,
    CSIConfig,
)
from .data_types import SlotContext, UEState, SlotResult
from .resource_grid import ResourceGrid
from .nr_constants import SLOTS_PER_FRAME, SLOTS_PER_SUBFRAME
from ..channel.statistical_channel import StatisticalChannel
from ..scheduler.pf_scheduler import PFSchedulerSUMIMO
from ..scheduler.rank_adaptation import RankAdapter
from ..traffic.full_buffer import FullBufferTraffic
from ..traffic.buffer_manager import BufferManager
from ..kpi.kpi_collector import KPICollector
from ..kpi.kpi_reporter import KPIReporter
from ..utils.math_utils import linear_to_db
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
        self.csi_config: CSIConfig = config.get('csi', CSIConfig())

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
                self.cell_config, self.carrier_config, self.channel_config,
                ue_config=self.ue_config,
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

        # PHY 层 (EESM + OLLA + ILLA + BLER 查表)
        # 优先使用 Sionna PHY (3GPP 合规); 不可用时回退到自实现
        try:
            from ..link_adaptation.sionna_phy import SionnaPHY
            self.sionna_phy = SionnaPHY(
                num_ue=self.num_ue,
                bler_target=self.la_config.bler_target,
                delta_up=self.la_config.olla_delta_up,
                offset_min=self.la_config.olla_offset_min,
                offset_max=self.la_config.olla_offset_max,
                mcs_table_index=self.la_config.mcs_table_index,
            )
            self._use_sionna_phy = True
        except (ImportError, RuntimeError):
            # 回退到自实现
            from ..link_adaptation.bler_tables import BLERTableManager
            from ..link_adaptation.effective_sinr import EESM
            from ..link_adaptation.illa import ILLA
            from ..link_adaptation.olla import OLLA
            from ..link_adaptation.phy_abstraction import PHYAbstraction
            self.bler_table = BLERTableManager()
            self.eesm = EESM()
            self.illa = ILLA(self.bler_table, self.la_config.bler_target,
                             self.la_config.mcs_table_index,
                             self.resource_grid.num_re_per_prb)
            self.olla = OLLA(self.num_ue, self.illa, self.la_config.bler_target,
                             self.la_config.olla_delta_up,
                             self.la_config.olla_offset_min,
                             self.la_config.olla_offset_max)
            self.phy_abs = PHYAbstraction(self.bler_table, self.eesm,
                                          self.resource_grid.num_re_per_prb,
                                          self.la_config.mcs_table_index,
                                          self.rng.phy)
            self._use_sionna_phy = False

        # 调度器
        self.scheduler = PFSchedulerSUMIMO(
            num_ue=self.num_ue,
            num_prb=self.carrier_config.num_prb,
            num_re_per_prb=self.resource_grid.num_re_per_prb,
            mcs_table_index=self.la_config.mcs_table_index,
            beta=self.scheduler_config.beta,
        )

        # CSI 反馈 + SINR 预估
        self._csi_enabled = self.csi_config.enabled and (channel_type == 'sionna')
        if self._csi_enabled:
            from ..csi.csi_feedback import CSIFeedbackManager
            from ..csi.sinr_prediction import SINRPredictor

            noise_per_prb = (1.380649e-23 * 290
                             * self.carrier_config.subcarrier_spacing * 1e3 * 12
                             * 10 ** (self.cell_config.noise_figure_db / 10))
            # CSI codebook 匹配物理天线数 (channel_matrix 的 tx 维度)
            num_tx_for_csi = self.cell_config.num_tx_ant
            if hasattr(self.channel, '_num_tx_ant'):
                num_tx_for_csi = self.channel._num_tx_ant
            self.csi_manager = CSIFeedbackManager(
                num_ue=self.num_ue,
                num_tx_ports=num_tx_for_csi,
                max_rank=min(self.cell_config.max_layers, self.ue_config.num_rx_ant),
                csi_period_slots=self.csi_config.csi_period_slots,
                feedback_delay_slots=self.csi_config.feedback_delay_slots,
                noise_power_per_prb=noise_per_prb,
                codebook_oversampling=self.csi_config.codebook_oversampling,
            )
            self.sinr_predictor = SINRPredictor(
                num_ue=self.num_ue,
                num_tx_ports=num_tx_for_csi,
                codebook=self.csi_manager.codebook,
            )

        # HARQ 管理器
        from ..harq.harq_buffer import HARQManager
        self.harq_mgr = HARQManager(
            num_ue=self.num_ue,
            num_processes=16,
            max_retx=4,
        )
        # per-UE 的活跃 HARQ process ID (-1 = 无)
        self._active_harq_pid = np.full(self.num_ue, -1, dtype=np.int32)

        # 流量模型
        self.traffic = FullBufferTraffic()
        self.buffer_mgr = BufferManager(self.num_ue)

        # KPI
        self.kpi = KPICollector(
            self.sim_config.num_slots, self.num_ue,
            warmup_slots=self.sim_config.warmup_slots
        )

        # 反馈缓存 (上一 TTI)
        self._last_harq = np.ones(self.num_ue, dtype=np.int32)  # 1=ACK
        self._last_harq_ack = np.ones(self.num_ue, dtype=bool)  # legacy 兼容
        self._last_sinr_eff = np.ones(self.num_ue, dtype=np.float32)  # linear
        self._last_scheduled_mask = np.zeros(self.num_ue, dtype=bool)
        self._buf_after_traffic = np.zeros(self.num_ue, dtype=np.int64)

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
            buf_after = np.array([ue.buffer_bytes for ue in self.ue_states], dtype=np.int64)
            self.kpi.collect(slot_idx, slot_result, self._buf_after_traffic, buf_after)

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

        # 2. UE 移动性: 更新位置 (每 slot)
        slot_duration = self.carrier_config.slot_duration_s
        for ue in self.ue_states:
            ue.position = ue.position + ue.velocity * slot_duration

        # 周期性重设 Sionna 拓扑 (每 topology_update_period slots)
        # 让 Sionna 内部更新 LSP 和 Doppler
        topo_period = getattr(self, '_topo_update_period', 20)
        if (slot_idx % topo_period == 0
                and hasattr(self.channel, 'initialize')
                and self.channel_config.type.lower() == 'sionna'):
            self.channel.initialize(self.cell_config, self.carrier_config, self.ue_states)

        # 信道更新
        channel_state = self.channel.update(slot_ctx, self.ue_states)

        # 3. CSI 测量与反馈 (周期性)
        if self._csi_enabled:
            # 接收到期的 CSI 反馈
            self.csi_manager.receive_feedback(slot_idx)

            # 周期性 CSI 测量
            if self.csi_manager.should_measure(slot_idx):
                if channel_state.channel_matrix is not None:
                    tx_pwr = self.cell_config.total_power_dbm
                    from ..utils.math_utils import dbm_to_watt
                    tx_per_prb = dbm_to_watt(tx_pwr) / num_prb
                    self.csi_manager.measure_and_report(
                        slot_idx, channel_state.channel_matrix, tx_per_prb
                    )

        # 4. Rank 选择 (优先使用 CSI RI)
        ue_rank = np.ones(num_ue, dtype=np.int32)
        if self._csi_enabled:
            for ue_idx in range(num_ue):
                report = self.csi_manager.get_latest_report(ue_idx)
                if report is not None:
                    ue_rank[ue_idx] = report.ri
                else:
                    ue_rank[ue_idx] = self.rank_adapter.select_rank(
                        channel_state.sinr_per_prb[ue_idx]
                    )
        else:
            ue_rank = self.rank_adapter.select_rank_batch(
                channel_state.sinr_per_prb, num_ue
            )
        for ue_idx, ue in enumerate(self.ue_states):
            ue.selected_rank = int(ue_rank[ue_idx])

        if self._use_sionna_phy:
            return self._run_slot_sionna_phy(slot_ctx, channel_state, ue_rank)
        else:
            return self._run_slot_legacy_phy(slot_ctx, channel_state, ue_rank)

    def _run_slot_sionna_phy(self, slot_ctx, channel_state, ue_rank):
        """使用 Sionna PHY 层的 slot 处理 (含 HARQ)"""
        num_ue = self.num_ue
        num_prb = self.carrier_config.num_prb
        re_per_prb = self.resource_grid.num_re_per_prb
        slot_idx = slot_ctx.slot_idx

        # --- SINR 预估 (CSI-based) ---
        sinr_eff_for_olla = self._last_sinr_eff.copy()
        if self._csi_enabled and channel_state.channel_matrix is not None:
            csi_reports = self.csi_manager.get_all_latest_reports()
            sinr_pred_db = self.sinr_predictor.predict_all_ue(
                csi_reports, channel_state.channel_matrix, mode='su'
            )
            from ..utils.math_utils import db_to_linear
            for ue in range(num_ue):
                if csi_reports[ue] is not None and sinr_pred_db[ue] > -29:
                    sinr_eff_for_olla[ue] = db_to_linear(sinr_pred_db[ue])

        # --- MCS 选择 (Sionna OLLA) ---
        est_re = np.full(num_ue, re_per_prb * max(num_prb // num_ue, 1), dtype=np.int32)
        mcs_indices = self.sionna_phy.select_mcs(
            num_allocated_re=est_re,
            harq_feedback=self._last_harq.copy(),
            sinr_eff=sinr_eff_for_olla,
            scheduled_mask=self._last_scheduled_mask,
        )

        # --- HARQ: 检查重传需求, 重传 UE 使用原始 MCS ---
        has_retx = self.harq_mgr.has_any_retransmission()
        retx_info = {}  # ue_id → retx_info dict
        for ue in range(num_ue):
            if has_retx[ue]:
                info = self.harq_mgr.get_retx_info(ue)
                if info is not None:
                    retx_info[ue] = info
                    mcs_indices[ue] = info['mcs']  # 重传用原始 MCS

        # --- 可达速率 (Shannon 容量) ---
        achievable_rate_per_prb = np.zeros((num_ue, num_prb))
        for ue in range(num_ue):
            r = int(ue_rank[ue])
            sinr_prb = channel_state.sinr_per_prb[ue, :r, :]
            capacity = np.sum(np.log2(1.0 + np.maximum(sinr_prb, 0)), axis=0)
            # 重传 UE 给予更高优先级 (capacity ×2)
            boost = 2.0 if ue in retx_info else 1.0
            achievable_rate_per_prb[ue, :] = capacity * re_per_prb * boost

        # --- PF 调度 ---
        ue_buffer = np.array([ue.buffer_bytes for ue in self.ue_states])
        # 重传 UE 也需要有 buffer (即使已传数据还没 ACK)
        for ue in retx_info:
            ue_buffer[ue] = max(ue_buffer[ue], retx_info[ue]['tbs'] // 8)
        sched = self.scheduler.schedule(
            slot_ctx, self.ue_states, channel_state,
            achievable_rate_per_prb, ue_buffer,
            mcs_indices, ue_rank
        )

        # --- PHY 评估 ---
        ue_num_re = sched.ue_num_prbs * re_per_prb
        phy_results = self.sionna_phy.evaluate(
            mcs_indices=mcs_indices,
            sinr_per_re=channel_state.sinr_per_prb,
            num_allocated_re=ue_num_re.astype(np.int32),
            prb_assignment=sched.prb_assignment,
        )

        # --- HARQ 反馈处理 ---
        final_decoded_bits = phy_results['decoded_bits'].copy()
        for ue in range(num_ue):
            is_scheduled = sched.ue_num_prbs[ue] > 0
            is_success = bool(phy_results['is_success'][ue])
            sinr_linear = float(phy_results['sinr_eff'][ue])

            if ue in retx_info:
                # 重传 UE: Chase Combining (累积 SINR)
                pid = retx_info[ue]['process_id']
                combined_sinr = self.harq_mgr.get_combining_sinr(ue, pid, sinr_linear)
                # 如果 combining 后 SINR 提升, 重新判定成功
                # 简化: combining 增益 ~3dB → 成功率提升
                if not is_success and combined_sinr > sinr_linear * 1.5:
                    is_success = True  # combining 增益使重传成功
                fb = self.harq_mgr.process_feedback(ue, pid, is_success, sinr_linear)
                final_decoded_bits[ue] = fb['decoded_bits']
            elif is_scheduled:
                # 新传输: 注册 HARQ 进程
                tbs = int(sched.ue_tbs_bits[ue])
                if tbs > 0:
                    pid = self.harq_mgr.start_new_tx(
                        ue, int(mcs_indices[ue]), int(sched.ue_num_prbs[ue]),
                        int(ue_rank[ue]), tbs, slot_idx
                    )
                    if pid >= 0:
                        self._active_harq_pid[ue] = pid
                        if not is_success:
                            # NACK: 加入重传队列
                            self.harq_mgr.process_feedback(ue, pid, False, sinr_linear)
                            final_decoded_bits[ue] = 0
                        else:
                            self.harq_mgr.process_feedback(ue, pid, True, sinr_linear)

        phy_results['decoded_bits'] = final_decoded_bits

        # --- 更新状态 ---
        self._last_harq = phy_results['harq_feedback'].copy()
        self._last_sinr_eff = phy_results['sinr_eff'].copy()
        self._last_scheduled_mask = sched.ue_num_prbs > 0

        sinr_eff_db = np.where(
            phy_results['sinr_eff'] > 0,
            linear_to_db(phy_results['sinr_eff']),
            -30.0
        )

        return self._finalize_slot(slot_ctx, phy_results, mcs_indices,
                                    ue_rank, sinr_eff_db, sched)

    def _run_slot_legacy_phy(self, slot_ctx, channel_state, ue_rank):
        """使用自实现 PHY 层的 slot 处理 (回退模式)"""
        from ..utils.nr_utils import get_spectral_efficiency
        num_ue = self.num_ue
        num_prb = self.carrier_config.num_prb
        re_per_prb = self.resource_grid.num_re_per_prb

        # OLLA 更新
        self.olla.update_offsets_batch(
            self._last_harq_ack, self._last_scheduled_mask
        )

        # EESM + MCS
        sinr_eff_db = np.zeros(num_ue)
        est_prbs = np.full(num_ue, max(num_prb // num_ue, 1), dtype=np.int32)
        for ue in range(num_ue):
            r = int(ue_rank[ue])
            sinr_eff_db[ue] = self.eesm.compute(
                channel_state.sinr_per_prb[ue, :r, :], mcs_index=0,
                table_index=self.la_config.mcs_table_index
            )
        mcs_indices = self.olla.select_mcs(sinr_eff_db, est_prbs, ue_rank)

        # Achievable rate
        achievable_rate_per_prb = np.zeros((num_ue, num_prb))
        for ue in range(num_ue):
            r = int(ue_rank[ue])
            sinr_prb = channel_state.sinr_per_prb[ue, :r, :]
            capacity = np.sum(np.log2(1.0 + np.maximum(sinr_prb, 0)), axis=0)
            achievable_rate_per_prb[ue, :] = capacity * re_per_prb

        # PF 调度
        ue_buffer = np.array([ue.buffer_bytes for ue in self.ue_states])
        sched = self.scheduler.schedule(
            slot_ctx, self.ue_states, channel_state,
            achievable_rate_per_prb, ue_buffer, mcs_indices, ue_rank
        )

        # PHY 评估
        phy_results = self.phy_abs.evaluate_batch(
            channel_state.sinr_per_prb, mcs_indices,
            sched.ue_num_prbs, ue_rank, sched.prb_assignment
        )

        # 更新状态
        self._last_harq_ack = phy_results['is_success'].copy()
        self._last_harq = np.where(phy_results['is_success'], 1, 0).astype(np.int32)
        self._last_sinr_eff = np.ones(num_ue, dtype=np.float32)
        self._last_scheduled_mask = sched.ue_num_prbs > 0

        return self._finalize_slot(slot_ctx, phy_results, mcs_indices,
                                    ue_rank, sinr_eff_db, sched)

    def _finalize_slot(self, slot_ctx, phy_results, mcs_indices,
                       ue_rank, sinr_eff_db, sched):
        """slot 后续处理: 缓冲区更新, 调度器更新, 构造 SlotResult"""
        # 更新缓冲区
        self.buffer_mgr.dequeue(self.ue_states, phy_results['decoded_bits'])

        # 更新调度器吞吐量历史
        self.scheduler.update_throughput_history(
            phy_results['decoded_bits'].astype(np.float64)
        )

        # 构造 SlotResult
        slot_duration_s = self.carrier_config.slot_duration_s
        throughput_inst = phy_results['decoded_bits'].astype(np.float64) / slot_duration_s

        return SlotResult(
            slot_idx=slot_ctx.slot_idx,
            ue_decoded_bits=phy_results['decoded_bits'],
            ue_bler=phy_results['tbler'],
            ue_tb_success=phy_results['is_success'],
            ue_mcs=mcs_indices,
            ue_rank=ue_rank,
            ue_sinr_eff_db=sinr_eff_db,
            ue_throughput_inst=throughput_inst,
            scheduling_decision=sched,
        )

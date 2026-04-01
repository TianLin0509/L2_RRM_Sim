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
from .registry import (
    get_scheduler_class, get_channel_class,
    get_channel_estimator_class, get_rank_adapter_class,
)
from .builtin_registry import ensure_loaded as _ensure_builtin
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
        _ensure_builtin()  # 确保内置组件已注册

        self.sim_config: SimConfig = config['sim']
        self.carrier_config: CarrierConfig = config['carrier']
        self.cell_config: CellConfig = config['cell']
        self.ue_config: UEConfig = config['ue']
        self.scheduler_config: SchedulerConfig = config['scheduler']
        self.la_config: LinkAdaptationConfig = config['link_adaptation']
        self.traffic_config: TrafficConfig = config['traffic']
        self.channel_config: ChannelConfig = config['channel']
        self.csi_config: CSIConfig = config.get('csi', CSIConfig())

        # TDD 配置
        from ..config.sim_config import TDDConfig as TDDCfg
        tdd_cfg = config.get('tdd', TDDCfg())
        self._is_tdd = (tdd_cfg.duplex_mode.upper() == 'TDD')
        if self._is_tdd:
            from .tdd_config import TDDConfig
            self.tdd = TDDConfig(
                pattern=tdd_cfg.pattern,
                special_dl_symbols=tdd_cfg.special_dl_symbols,
                special_gp_symbols=tdd_cfg.special_gp_symbols,
                special_ul_symbols=tdd_cfg.special_ul_symbols,
            )
            self._harq_k1 = tdd_cfg.harq_k1
        else:
            self.tdd = None
            self._harq_k1 = 4

        # 随机数
        self.rng = SimRNG(self.sim_config.random_seed)

        # 资源网格
        self.resource_grid = ResourceGrid(self.carrier_config)

        # 初始化 UE
        self.ue_states = self._init_ue_states()
        self.num_ue = len(self.ue_states)

        # 信道 (通过 registry 查找)
        channel_type = self.channel_config.type.lower()
        if channel_type == 'sionna':
            # Sionna 信道需要特殊参数，暂保留直接导入
            from ..channel.sionna_channel import SionnaChannel
            self.channel = SionnaChannel(
                self.cell_config, self.carrier_config, self.channel_config,
                ue_config=self.ue_config,
            )
        else:
            ChannelClass = get_channel_class(channel_type)
            self.channel = ChannelClass(
                self.cell_config, self.carrier_config,
                self.channel_config, self.rng.channel
            )
        self.channel.initialize(
            self.cell_config, self.carrier_config, self.ue_states
        )

        # Rank 自适应 (通过 registry)
        rank_type = getattr(self.scheduler_config, 'rank_adapter', 'shannon')
        RankClass = get_rank_adapter_class(rank_type)
        self.rank_adapter = RankClass(
            max_rank=self.cell_config.max_layers,
            fixed_rank=None,
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

        # 调度器 (通过 registry)
        sched_type = self.scheduler_config.type.lower()
        SchedulerClass = get_scheduler_class(sched_type)
        self.scheduler = SchedulerClass(
            num_ue=self.num_ue,
            num_prb=self.carrier_config.num_prb,
            num_re_per_prb=self.resource_grid.num_re_per_prb,
            mcs_table_index=self.la_config.mcs_table_index,
            beta=self.scheduler_config.beta,
            resource_grid=self.resource_grid,
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
                subband_size_prb=self.csi_config.subband_size_prb,
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

        # 流量模型 (按 TrafficConfig.type 实例化)
        self.traffic = self._create_traffic_model()
        self.buffer_mgr = BufferManager(self.num_ue)

        # KPI
        self.kpi = KPICollector(
            self.sim_config.num_slots, self.num_ue,
            warmup_slots=self.sim_config.warmup_slots
        )

        # 信道估计器
        # 信道估计器 (通过 registry)
        est_type = getattr(self.channel_config, 'estimator', 'ls')
        EstimatorClass = get_channel_estimator_class(est_type)
        self.channel_estimator = EstimatorClass(self.rng, estimation_error_std=0.05)

        # 事件总线 (Observer Pattern)
        from ..kpi.event_bus import EventBus
        self.event_bus = EventBus()

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

        # TDD 方向
        if self._is_tdd:
            direction = self.tdd.get_slot_direction(slot_idx)
            dl_symbols = self.tdd.get_dl_symbols(slot_idx)
        else:
            direction = 'D'
            dl_symbols = 14

        return SlotContext(
            slot_idx=slot_idx,
            time_s=slot_idx * self.carrier_config.slot_duration_s,
            frame_idx=slot_idx // slots_per_frame,
            subframe_idx=(slot_idx % slots_per_frame) // slots_per_subframe,
            slot_in_subframe=slot_idx % slots_per_subframe,
            slot_direction=direction,
            num_dl_symbols=dl_symbols,
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

        # 启动事件总线异步处理
        from ..kpi.event_bus import SimEvent
        self.event_bus.start()
        self.event_bus.emit(SimEvent('sim_started', data=None))

        for slot_idx in range(num_slots):
            slot_result = self.run_slot(slot_idx)
            buf_after = np.array([ue.buffer_bytes for ue in self.ue_states], dtype=np.int64)
            self.kpi.collect(slot_idx, slot_result, self._buf_after_traffic, buf_after)

            # 发射 slot_completed 事件
            self.event_bus.emit(SimEvent('slot_completed', slot_idx=slot_idx, data=slot_result))

            if (slot_idx + 1) % 1000 == 0 or slot_idx == num_slots - 1:
                elapsed = time.time() - t_start
                speed = (slot_idx + 1) / elapsed
                print(f"  Slot {slot_idx + 1}/{num_slots} "
                      f"({speed:.0f} slots/s, {elapsed:.1f}s elapsed)")

        elapsed_total = time.time() - t_start
        print(f"=== 仿真完成 ({elapsed_total:.1f}s) ===\n")

        # 停止事件总线
        self.event_bus.emit(SimEvent('sim_completed', data={'elapsed_s': elapsed_total}))
        self.event_bus.stop()

        # 生成报告
        reporter = KPIReporter(self.kpi, self.carrier_config,
                               trim_percent=self.sim_config.kpi_trim_percent)
        report = reporter.report()
        reporter.print_report(report)
        return report

    def run_slot(self, slot_idx: int) -> SlotResult:
        """运行单个 slot"""
        slot_ctx = self._make_slot_context(slot_idx)
        num_ue = self.num_ue
        num_prb = self.carrier_config.num_prb

        # 先交付到期 HARQ 反馈，保证 TDD 下 ACK/NACK 在 UL slot 生效
        delivered = self.harq_mgr.deliver_feedback(slot_idx)
        if delivered:
            for ue_id, fb_result in delivered:
                if fb_result.get('is_final_ack'):
                    self._last_harq[ue_id] = 1  # ACK
                    self._last_harq_ack[ue_id] = True
                elif fb_result.get('retx_needed'):
                    self._last_harq[ue_id] = 0  # NACK
                    self._last_harq_ack[ue_id] = False
        delivered_bits = self.harq_mgr.get_delivered_decoded_bits()

        # TDD: UL slot 跳过 DL 调度 (流量仍生成, UE 仍移动)
        if slot_ctx.slot_direction == 'U':
            self.traffic.generate(slot_ctx, self.ue_states)
            self._buf_after_traffic = np.array(
                [ue.buffer_bytes for ue in self.ue_states], dtype=np.int64)
            throughput_inst = delivered_bits.astype(np.float64) / self.carrier_config.slot_duration_s
            return SlotResult(
                slot_idx=slot_idx,
                ue_decoded_bits=delivered_bits,
                ue_bler=np.zeros(num_ue), ue_tb_success=np.zeros(num_ue, dtype=bool),
                ue_mcs=np.zeros(num_ue, dtype=np.int32),
                ue_rank=np.ones(num_ue, dtype=np.int32),
                ue_sinr_eff_db=np.full(num_ue, -30.0),
                ue_throughput_inst=throughput_inst,
                slot_direction=slot_ctx.slot_direction,
                num_dl_symbols=slot_ctx.num_dl_symbols,
            )

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
        
        # 生成估计信道 (包含 LS 估计误差)
        channel_state.estimated_channel_matrix = self.channel_estimator.estimate(channel_state)

        # 3. CSI 测量与反馈 (使用估计信道)
        if self._csi_enabled:
            # 接收到期的 CSI 反馈
            self.csi_manager.receive_feedback(slot_idx)

            # 周期性 CSI 测量
            if self.csi_manager.should_measure(slot_idx):
                h_est = channel_state.estimated_channel_matrix
                if h_est is not None:
                    tx_pwr = self.cell_config.total_power_dbm
                    from ..utils.math_utils import dbm_to_watt
                    tx_per_prb = dbm_to_watt(tx_pwr) / num_prb
                    self.csi_manager.measure_and_report(
                        slot_idx, h_est, tx_per_prb
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

        # Apply rank-dependent power scaling to SINR:
        # channel stores rank-independent P*|s|²/N0, convert to P*|s|²/(r*N0)
        for ue_idx in range(num_ue):
            r = int(ue_rank[ue_idx])
            if r > 0:
                channel_state.sinr_per_prb[ue_idx, :r, :] /= r
                channel_state.sinr_per_prb[ue_idx, r:, :] = 0.0

        if self._use_sionna_phy:
            return self._run_slot_sionna_phy(slot_ctx, channel_state, ue_rank)
        else:
            return self._run_slot_legacy_phy(slot_ctx, channel_state, ue_rank)

    def _build_harq_combined_sinr(self, raw_sinr_linear: np.ndarray,
                                  sched, retx_info: dict) -> np.ndarray:
        """Combine current effective SINR with HARQ history for retransmissions."""
        combined = raw_sinr_linear.copy()
        for ue, info in retx_info.items():
            if sched.ue_num_prbs[ue] <= 0:
                continue
            combined[ue] = self.harq_mgr.get_combining_sinr(
                ue, info['process_id'], float(raw_sinr_linear[ue])
            )
        return combined

    def _compute_legacy_raw_sinr_linear(self, channel_state, mcs_indices,
                                        ue_rank, sched) -> np.ndarray:
        """Compute current-slot effective SINR before HARQ combining."""
        from ..utils.math_utils import db_to_linear
        num_ue = self.num_ue
        raw_sinr_linear = np.zeros(num_ue, dtype=np.float32)
        for ue in range(num_ue):
            if sched.ue_num_prbs[ue] <= 0:
                continue
            n_layers = int(ue_rank[ue])
            ue_prb_mask = (sched.prb_assignment == ue)
            sinr_ue = channel_state.sinr_per_prb[ue, :n_layers, :][:, ue_prb_mask]
            sinr_eff_db = self.eesm.compute(
                sinr_ue,
                int(mcs_indices[ue]),
                self.la_config.mcs_table_index,
            )
            raw_sinr_linear[ue] = db_to_linear(sinr_eff_db)
        return raw_sinr_linear

    def _run_slot_sionna_phy(self, slot_ctx, channel_state, ue_rank):
        """使用 Sionna PHY 层的 slot 处理 (含 HARQ)"""
        num_ue = self.num_ue
        num_prb = self.carrier_config.num_prb
        re_per_prb = self.resource_grid.compute_re_per_prb(slot_ctx.num_dl_symbols)
        slot_idx = slot_ctx.slot_idx

        # 初始 MCS (基于宽带 SINR 的保守估计，随后由 OLLA/CQI 细化)
        mcs_indices = np.zeros(num_ue, dtype=np.int32)

        # --- SINR 预估 (CSI-based) ---
        sinr_eff_for_olla = self._last_sinr_eff.copy()
        if self._csi_enabled and channel_state.actual_channel_matrix is not None:
            csi_reports = self.csi_manager.get_all_latest_reports()
            # 使用估计信道进行预估 (模拟 gNB 侧行为)
            sinr_pred_db = self.sinr_predictor.predict_all_ue(
                csi_reports, channel_state.estimated_channel_matrix, mode='su'
            )
            from ..utils.math_utils import db_to_linear
            from ..utils.cqi_utils import sinr_to_cqi, cqi_to_mcs
            
            for ue in range(num_ue):
                if csi_reports[ue] is not None and sinr_pred_db[ue] > -29:
                    # 模拟 UE 端 SINR -> CQI -> gNB 端 MCS
                    cqi = sinr_to_cqi(sinr_pred_db[ue])
                    mcs_indices[ue] = cqi_to_mcs(cqi)
                    sinr_eff_for_olla[ue] = db_to_linear(sinr_pred_db[ue])

        # --- MCS 优化 (基于 OLLA 进一步细化) ---
        est_re = np.full(num_ue, re_per_prb * max(num_prb // num_ue, 1), dtype=np.int32)
        mcs_indices = self.sionna_phy.select_mcs(
            num_allocated_re=est_re,
            harq_feedback=self._last_harq.copy(),
            sinr_eff=sinr_eff_for_olla,
            scheduled_mask=self._last_scheduled_mask,
        )

        # --- HARQ: 两阶段 peek → 调度 → consume，防止未调度 UE 的进程丢失 ---
        has_retx = self.harq_mgr.has_any_retransmission()
        retx_info = {}  # ue_id → retx_info dict
        for ue in range(num_ue):
            if has_retx[ue]:
                info = self.harq_mgr.peek_retx_info(ue)  # 只读，不消费
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
        for ue in retx_info:
            ue_buffer[ue] = max(ue_buffer[ue], retx_info[ue]['tbs'] // 8)
        sched = self.scheduler.schedule(
            slot_ctx, self.ue_states, channel_state,
            achievable_rate_per_prb, ue_buffer,
            mcs_indices, ue_rank,
            re_per_prb=re_per_prb,
        )

        # 调度完成后，消费已被调度的重传进程（未调度的保留在队列中）
        for ue in retx_info:
            if sched.ue_num_prbs[ue] > 0:
                self.harq_mgr.consume_retx(ue)
        
        # --- L1 评估闭环: 计算 MU-MIMO 真实 SINR ---
        if sched.mu_groups is not None:
            from ..scheduler.mu_mimo_scheduler import compute_mu_mimo_sinr
            from ..utils.math_utils import dbm_to_watt
            
            tx_pwr_prb = dbm_to_watt(self.cell_config.total_power_dbm) / num_prb
            noise_pwr_prb = (1.3806e-23 * 290 * self.carrier_config.subcarrier_spacing * 1e3 * 12 
                             * 10**(self.cell_config.noise_figure_db/10))
            
            # 物理层真实的 SINR 网格 (初始化为全 0)
            actual_sinr_per_prb = np.zeros_like(channel_state.sinr_per_prb)

            for prb in range(num_prb):
                paired = sched.mu_groups[prb]
                if not paired:
                    continue
                
                # 计算该组 UE 在当前 PRB 的后处理 SINR (IRC 接收机)
                mu_sinrs = compute_mu_mimo_sinr(
                    channel_state.actual_channel_matrix,
                    channel_state.estimated_channel_matrix,
                    paired, prb, tx_pwr_prb, noise_pwr_prb
                )
                
                for i, ue_idx in enumerate(paired):
                    actual_sinr_per_prb[ue_idx, 0, prb] = mu_sinrs[i]
            
            # 使用受干扰影响的真实 SINR
            phy_sinr_input = actual_sinr_per_prb
        else:
            phy_sinr_input = channel_state.sinr_per_prb

        # --- PHY 评估 ---
        ue_num_re = sched.ue_num_prbs * re_per_prb
        raw_sinr_eff = self.sionna_phy.compute_sinr_eff(
            phy_sinr_input, mcs_indices, sched.prb_assignment
        )
        combined_sinr_eff = self._build_harq_combined_sinr(
            raw_sinr_eff, sched, retx_info
        )
        phy_results = self.sionna_phy.evaluate(
            mcs_indices=mcs_indices,
            sinr_per_re=phy_sinr_input,
            num_allocated_re=ue_num_re.astype(np.int32),
            prb_assignment=sched.prb_assignment,
            sinr_eff_override=combined_sinr_eff,
        )

        # --- HARQ: K1 延迟反馈 ---
        # PHY 判定结果不立即处理, 排入 K1 延迟队列
        # decoded_bits 在 TX slot 记为 0, 在 feedback 到达时通过 deliver 记录
        final_decoded_bits = np.zeros(num_ue, dtype=np.int64)

        # 加上本 slot K1 到期交付的 decoded bits
        final_decoded_bits += self.harq_mgr.get_delivered_decoded_bits()

        for ue in range(num_ue):
            is_scheduled = sched.ue_num_prbs[ue] > 0
            if not is_scheduled:
                continue

            is_success = bool(phy_results['is_success'][ue])
            sinr_linear = float(phy_results['sinr_eff_raw'][ue])
            tbs = int(sched.ue_tbs_bits[ue])

            # 计算 feedback 到达 slot
            if self._is_tdd:
                fb_slot = self.tdd.get_feedback_slot(slot_idx, self._harq_k1)
            else:
                fb_slot = slot_idx + self._harq_k1

            if ue in retx_info:
                # 重传: 排入 K1 队列
                pid = retx_info[ue]['process_id']
                self.harq_mgr.queue_feedback(
                    fb_slot, ue, pid, is_success, sinr_linear, tbs
                )
            else:
                # 新传输: 先注册 HARQ 进程, 再排入 K1 队列
                if tbs > 0:
                    pid = self.harq_mgr.start_new_tx(
                        ue, int(mcs_indices[ue]), int(sched.ue_num_prbs[ue]),
                        int(ue_rank[ue]), tbs, slot_idx
                    )
                    if pid >= 0:
                        self._active_harq_pid[ue] = pid
                        self.harq_mgr.queue_feedback(
                            fb_slot, ue, pid, is_success, sinr_linear, tbs
                        )

        phy_results['decoded_bits'] = final_decoded_bits

        # --- 更新状态 ---
        self._last_harq = phy_results['harq_feedback'].copy()
        self._last_sinr_eff = phy_results['sinr_eff_raw'].copy()
        self._last_scheduled_mask = sched.ue_num_prbs > 0

        sinr_eff_db = np.where(
            phy_results['sinr_eff'] > 0,
            linear_to_db(phy_results['sinr_eff']),
            -30.0
        )

        return self._finalize_slot(slot_ctx, phy_results, mcs_indices,
                                    ue_rank, sinr_eff_db, sched)

    def _run_slot_legacy_phy(self, slot_ctx, channel_state, ue_rank):
        """使用自实现 PHY 层的 slot 处理 (回退模式, 含 HARQ)"""
        from ..utils.nr_utils import get_spectral_efficiency
        num_ue = self.num_ue
        num_prb = self.carrier_config.num_prb
        re_per_prb = self.resource_grid.compute_re_per_prb(slot_ctx.num_dl_symbols)
        slot_idx = slot_ctx.slot_idx

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
        # TDD special slot uses fewer REs, which changes TBS/CBS and therefore
        # the BLER-constrained MCS selected by ILLA.
        self.illa.num_re_per_prb = re_per_prb
        mcs_indices = self.olla.select_mcs(sinr_eff_db, est_prbs, ue_rank)

        # Achievable rate (rank=1 fast path)
        if np.all(ue_rank == 1):
            sinr_r1 = channel_state.sinr_per_prb[:, 0, :]
            achievable_rate_per_prb = np.log2(1.0 + np.maximum(sinr_r1, 0)) * re_per_prb
        else:
            achievable_rate_per_prb = np.zeros((num_ue, num_prb))
            for ue in range(num_ue):
                r = int(ue_rank[ue])
                sinr_prb = channel_state.sinr_per_prb[ue, :r, :]
                capacity = np.sum(np.log2(1.0 + np.maximum(sinr_prb, 0)), axis=0)
                achievable_rate_per_prb[ue, :] = capacity * re_per_prb

        # --- HARQ: peek retx ---
        has_retx = self.harq_mgr.has_any_retransmission()
        retx_info = {}
        for ue in range(num_ue):
            if has_retx[ue]:
                info = self.harq_mgr.peek_retx_info(ue)
                if info is not None:
                    retx_info[ue] = info
                    mcs_indices[ue] = info['mcs']

        # PF 调度
        ue_buffer = np.array([ue.buffer_bytes for ue in self.ue_states])
        for ue in retx_info:
            ue_buffer[ue] = max(ue_buffer[ue], retx_info[ue]['tbs'] // 8)
        sched = self.scheduler.schedule(
            slot_ctx, self.ue_states, channel_state,
            achievable_rate_per_prb, ue_buffer, mcs_indices, ue_rank,
            re_per_prb=re_per_prb,
        )

        # Consume retx for scheduled UEs
        for ue in retx_info:
            if sched.ue_num_prbs[ue] > 0:
                self.harq_mgr.consume_retx(ue)

        # PHY 评估
        raw_sinr_linear = self._compute_legacy_raw_sinr_linear(
            channel_state, mcs_indices, ue_rank, sched
        )
        combined_sinr_linear = self._build_harq_combined_sinr(
            raw_sinr_linear, sched, retx_info
        )
        phy_results = self.phy_abs.evaluate_batch(
            channel_state.sinr_per_prb, mcs_indices,
            sched.ue_num_prbs, ue_rank, sched.prb_assignment,
            re_per_prb=re_per_prb,
            sinr_eff_override_linear=combined_sinr_linear,
        )

        # --- HARQ K1 feedback ---
        final_decoded_bits = np.zeros(num_ue, dtype=np.int64)
        final_decoded_bits += self.harq_mgr.get_delivered_decoded_bits()

        from ..utils.math_utils import db_to_linear
        for ue in range(num_ue):
            is_scheduled = sched.ue_num_prbs[ue] > 0
            if not is_scheduled:
                continue

            is_success = bool(phy_results['is_success'][ue])
            sinr_linear = float(raw_sinr_linear[ue])
            tbs = int(sched.ue_tbs_bits[ue])

            if self._is_tdd:
                fb_slot = self.tdd.get_feedback_slot(slot_idx, self._harq_k1)
            else:
                fb_slot = slot_idx + self._harq_k1

            if ue in retx_info:
                pid = retx_info[ue]['process_id']
                self.harq_mgr.queue_feedback(
                    fb_slot, ue, pid, is_success, sinr_linear, tbs
                )
            else:
                if tbs > 0:
                    pid = self.harq_mgr.start_new_tx(
                        ue, int(mcs_indices[ue]), int(sched.ue_num_prbs[ue]),
                        int(ue_rank[ue]), tbs, slot_idx
                    )
                    if pid >= 0:
                        self._active_harq_pid[ue] = pid
                        self.harq_mgr.queue_feedback(
                            fb_slot, ue, pid, is_success, sinr_linear, tbs
                        )

        phy_results['decoded_bits'] = final_decoded_bits

        # 更新状态
        sinr_eff_db = phy_results['sinr_eff_db'].copy()
        sinr_eff_db_raw = phy_results['sinr_eff_db_raw'].copy()
        self._last_harq_ack = phy_results['is_success'].copy()
        self._last_harq = np.where(phy_results['is_success'], 1, 0).astype(np.int32)
        self._last_sinr_eff = db_to_linear(sinr_eff_db_raw).astype(np.float32)
        self._last_scheduled_mask = sched.ue_num_prbs > 0

        return self._finalize_slot(slot_ctx, phy_results, mcs_indices,
                                    ue_rank, sinr_eff_db, sched)

    def _finalize_slot(self, slot_ctx, phy_results, mcs_indices,
                       ue_rank, sinr_eff_db, sched):
        """slot 后续处理: 缓冲区更新, 调度器更新, 构造 SlotResult"""
        # 截断 decoded_bits: 不能超过实际 buffer (非满缓冲场景)
        for ue_idx, ue in enumerate(self.ue_states):
            max_bits = ue.buffer_bytes * 8
            if phy_results['decoded_bits'][ue_idx] > max_bits > 0:
                phy_results['decoded_bits'][ue_idx] = max_bits

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
            slot_direction=slot_ctx.slot_direction,
            num_dl_symbols=slot_ctx.num_dl_symbols,
            scheduling_decision=sched,
        )

    def _create_traffic_model(self):
        """按 TrafficConfig.type 创建流量模型"""
        t = self.traffic_config.type.lower()
        if t in ('ftp', 'ftp_model3', 'ftp3'):
            from ..traffic.ftp_model import FTPModel3
            return FTPModel3(
                file_size_bytes=self.traffic_config.ftp_file_size_bytes,
                arrival_rate=self.traffic_config.ftp_lambda,
                slot_duration_s=self.carrier_config.slot_duration_s,
                num_ue=self.num_ue,
                rng=self.rng.traffic,
            )
        elif t in ('poisson', 'bursty'):
            from ..traffic.bursty_traffic import PoissonTraffic
            return PoissonTraffic(
                packet_size_bytes=1500,
                arrival_rate_pps=self.traffic_config.ftp_lambda * 1000,
                slot_duration_s=self.carrier_config.slot_duration_s,
                num_ue=self.num_ue,
                rng=self.rng.traffic,
            )
        elif t == 'realistic':
            from ..traffic.realistic_traffic import RealisticTraffic
            return RealisticTraffic(
                mean_arrival_rate_pps=self.traffic_config.realistic_arrival_rate_pps,
                packet_size_mean_bytes=self.traffic_config.realistic_pkt_size_mean,
                packet_size_std_bytes=self.traffic_config.realistic_pkt_size_std,
                slot_duration_s=self.carrier_config.slot_duration_s,
                num_ue=self.num_ue,
                rng=self.rng.traffic,
            )
        else:
            from ..traffic.full_buffer import FullBufferTraffic
            return FullBufferTraffic()

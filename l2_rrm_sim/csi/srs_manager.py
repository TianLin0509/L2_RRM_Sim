"""SRS (Sounding Reference Signal) 管理模块

在 TDD 模式下，基站利用 SRS 获取上行信道，并通过互惠性指导下行预编码。
实现了 SRS 周期性发送、分频段跳频 (Hopping) 以及处理时延 (Aging)。
"""

import numpy as np
from ..config.sim_config import CSIConfig
from ..utils.random_utils import SimRNG


class SRSManager:
    """SRS 管理器 (gNB 侧)

    管理 gNB 获取的信道缓冲区。
    h_srs_buffer 包含全带宽信道，但每周期只更新部分带宽。
    """

    def __init__(self, num_ue: int, num_rx_ant: int, num_tx_ant: int, 
                 num_prb: int, config: CSIConfig, rng: SimRNG):
        self.num_ue = num_ue
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant
        self.num_prb = num_prb
        self.config = config
        self.rng = rng

        # 基站维护的信道缓冲区 (gNB 视角下的信道，包含时延和跳频后的旧数据)
        # shape: (num_ue, rx_ant, tx_ant, num_prb)
        self.h_srs_buffer = np.zeros((num_ue, num_rx_ant, num_tx_ant, num_prb), dtype=complex)
        self._buffer_initialized = False  # 首次有效测量到达前为 False
        
        # SRS 测量历史，用于处理延迟 (delay line)
        # {delivery_slot: h_snapshot}
        self._measurement_queue = {}
        
        # 记录每 UE 当前跳频到的子带索引 (0 ~ hopping_subbands-1)
        self._ue_subband_idx = np.zeros(num_ue, dtype=int)
        
        # 子带 PRB 分组
        self.prbs_per_subband = int(np.ceil(num_prb / config.srs_hopping_subbands))

    def update_measurements(self, slot_idx: int, h_actual: np.ndarray):
        """执行 SRS 测量 (在 UE 的 SRS 发送 slot)
        
        Args:
            h_actual: (num_ue, rx_ant, tx_ant, num_prb) 真实物理信道
        """
        if h_actual is None: return

        # 检查是否到了该周期的测量点 (假设所有 UE 的 SRS 周期一致，但偏移可调)
        if slot_idx % self.config.srs_period_slots != 0:
            return

        # 模拟基站测量到的快照 (带噪声)
        h_noise_std = self.config.estimation_error_std * np.mean(np.abs(h_actual))
        noise = (self.rng.channel.normal(0, h_noise_std, h_actual.shape) + 
                 1j * self.rng.channel.normal(0, h_noise_std, h_actual.shape))
        h_measured = h_actual + noise

        # 模拟 SRS 测量快照并加入延迟队列
        h_snapshot = self.h_srs_buffer.copy()
        for ue_id in range(self.num_ue):
            sb_idx = self._ue_subband_idx[ue_id]
            prb_start = sb_idx * self.prbs_per_subband
            prb_end = min(prb_start + self.prbs_per_subband, self.num_prb)
            h_snapshot[ue_id, :, :, prb_start:prb_end] = h_measured[ue_id, :, :, prb_start:prb_end]
            
            # 更新下一周期的跳频子带
            self._ue_subband_idx[ue_id] = (sb_idx + 1) % self.config.srs_hopping_subbands

        delivery_slot = slot_idx + self.config.srs_processing_delay
        self._measurement_queue[delivery_slot] = h_snapshot

    def get_estimated_channel(self, slot_idx: int) -> np.ndarray:
        """获取基站当前可用的估计信道 (包含 Aging 和 Hopping 拼接)

        Returns:
            估计信道矩阵，若首次测量尚未到达则返回 None (由调用方回退)
        """
        # 检查延迟队列，获取已处理完成的测量值
        ready_slots = [s for s in self._measurement_queue.keys() if s <= slot_idx]
        if ready_slots:
            latest_ready = max(ready_slots)
            self.h_srs_buffer = self._measurement_queue.pop(latest_ready)
            self._buffer_initialized = True
            # 清理其他已过期的 slot（latest_ready 已被 pop，跳过它）
            for s in ready_slots:
                if s != latest_ready:
                    self._measurement_queue.pop(s, None)

        if not self._buffer_initialized:
            return None  # 调用方会回退到 LS 估计

        return self.h_srs_buffer.copy()

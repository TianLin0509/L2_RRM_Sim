"""外环链路自适应 (Outer Loop Link Adaptation)

根据 HARQ 反馈调整 SINR 偏移量，使实际 BLER 收敛到目标值。

收敛条件 (稳态):
    P(NACK) × delta_up = P(ACK) × delta_down
    => delta_down = delta_up × bler_target / (1 - bler_target)

对齐 Sionna 实现:
    ACK → offset -= delta_down (更激进, 尝试更高 MCS)
    NACK → offset += delta_up (更保守, 降低 MCS)
    adjusted_sinr = sinr_eff - offset
"""

import numpy as np
from .illa import ILLA


class OLLA:
    """外环链路自适应

    维护每 UE 的 SINR 偏移量 (offset_db)。
    """

    def __init__(self, num_ue: int, illa: ILLA,
                 bler_target: float = 0.1,
                 delta_up: float = 0.5,
                 offset_min: float = -10.0,
                 offset_max: float = 10.0):
        """
        Args:
            delta_up: NACK 时偏移增加量 (dB), 较保守的默认值
            offset_min/max: 偏移范围 (dB), [-10, 10] 更合理
        """
        self.num_ue = num_ue
        self.illa = illa
        self.bler_target = bler_target
        self.delta_up = delta_up
        # 收敛条件: delta_down / delta_up = bler_target / (1 - bler_target)
        self.delta_down = delta_up * bler_target / (1.0 - bler_target)
        self.offset_min = offset_min
        self.offset_max = offset_max

        # per-UE 状态
        self._offset = np.zeros(num_ue, dtype=np.float64)

    @property
    def offsets(self) -> np.ndarray:
        return self._offset.copy()

    def update_offset(self, ue_id: int, is_ack: bool):
        """根据 HARQ 反馈更新偏移量"""
        if is_ack:
            self._offset[ue_id] -= self.delta_down
        else:
            self._offset[ue_id] += self.delta_up
        self._offset[ue_id] = np.clip(
            self._offset[ue_id], self.offset_min, self.offset_max
        )

    def update_offsets_batch(self, harq_ack: np.ndarray,
                            scheduled_mask: np.ndarray = None):
        """批量更新被调度 UE 的偏移量"""
        if scheduled_mask is None:
            scheduled_mask = np.ones(self.num_ue, dtype=bool)
        scheduled = scheduled_mask.astype(bool)
        ack_mask = harq_ack.astype(bool) & scheduled
        nack_mask = (~harq_ack.astype(bool)) & scheduled
        self._offset[ack_mask] -= self.delta_down
        self._offset[nack_mask] += self.delta_up
        np.clip(self._offset, self.offset_min, self.offset_max, out=self._offset)

    def select_mcs(self, sinr_eff_db: np.ndarray,
                   num_allocated_prbs: np.ndarray = None,
                   num_layers: np.ndarray = None) -> np.ndarray:
        """使用 OLLA 调整后的 SINR 选择 MCS"""
        if num_allocated_prbs is None:
            num_allocated_prbs = np.ones(self.num_ue, dtype=np.int32)
        if num_layers is None:
            num_layers = np.ones(self.num_ue, dtype=np.int32)

        adjusted_sinr = sinr_eff_db - self._offset

        mcs_indices = np.zeros(self.num_ue, dtype=np.int32)
        for ue in range(self.num_ue):
            mcs_indices[ue] = self.illa.select_mcs(
                float(adjusted_sinr[ue]),
                int(num_allocated_prbs[ue]),
                int(num_layers[ue])
            )
        return mcs_indices

    def reset(self):
        """重置所有偏移量"""
        self._offset[:] = 0.0

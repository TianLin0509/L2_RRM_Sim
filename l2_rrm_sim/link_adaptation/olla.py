"""外环链路自适应 (Outer Loop Link Adaptation) — MCS 域实现

设计:
  - ILLA 用原始 SINR 选出初步 MCS (基础 MCS)
  - OLLA 维护 MCS 域的 offset (浮点)
  - 最终 MCS = round(illa_mcs + olla_offset), clip 到 [0, max_mcs]

OLLA offset 更新:
  - ACK → offset += ack_step (默认 0.01)
  - NACK → offset -= nack_step (默认 0.1)
  - 步长比决定 BLER 稳态: P(NACK)/P(ACK) = ack_step/nack_step
    默认 0.01/0.1 = 0.1 → BLER 稳态 ≈ 9.1% (接近 10% 目标)

初始值 -4 对齐华为默认方案, 快速进入 BLER 收敛域。

参考: 华为商用 BTS 默认 OLLA 方案
"""

import numpy as np
from .illa import ILLA
from .mcs_tables import get_max_mcs_index


class OLLA:
    """外环链路自适应 (MCS 域)

    维护每 UE 的 MCS 偏移量 (MCS 索引单位, 浮点)。
    """

    def __init__(self, num_ue: int, illa: ILLA,
                 bler_target: float = 0.1,
                 nack_step: float = 0.1,
                 ack_step: float = 0.01,
                 initial_offset: float = -4.0,
                 mcs_table_index: int = 1,
                 # --- 以下为兼容旧接口 (SINR dB 域) 的参数, 已弃用 ---
                 delta_up: float = None,
                 offset_min: float = None,
                 offset_max: float = None):
        """
        Args:
            num_ue: UE 数量
            illa: ILLA 实例, 用于预估基础 MCS
            bler_target: BLER 目标 (仅用于记录, 实际收敛由步长比决定)
            nack_step: NACK 时 MCS offset 减少量 (默认 0.1)
            ack_step: ACK 时 MCS offset 增加量 (默认 0.01)
            initial_offset: MCS offset 初始值 (默认 -4.0, 华为商用默认)
            mcs_table_index: MCS 表索引 (用于 clip 上界)

        Deprecated (兼容旧调用):
            delta_up, offset_min, offset_max: 旧 SINR dB 域参数, 不再使用
        """
        self.num_ue = num_ue
        self.illa = illa
        self.bler_target = bler_target
        self.nack_step = nack_step
        self.ack_step = ack_step
        self.mcs_table_index = mcs_table_index
        self._max_mcs = get_max_mcs_index(mcs_table_index)

        # 每 UE 的 MCS offset (浮点)
        self._offset = np.full(num_ue, initial_offset, dtype=np.float64)

    @property
    def offsets(self) -> np.ndarray:
        """返回当前 MCS offset (MCS index 单位)"""
        return self._offset.copy()

    def update_offset(self, ue_id: int, is_ack: bool):
        """根据 HARQ 反馈更新偏移量"""
        if is_ack:
            self._offset[ue_id] += self.ack_step
        else:
            self._offset[ue_id] -= self.nack_step

    def update_offsets_batch(self, harq_ack: np.ndarray,
                            scheduled_mask: np.ndarray = None):
        """批量更新被调度 UE 的偏移量"""
        if scheduled_mask is None:
            scheduled_mask = np.ones(self.num_ue, dtype=bool)
        scheduled = scheduled_mask.astype(bool)
        ack_mask = harq_ack.astype(bool) & scheduled
        nack_mask = (~harq_ack.astype(bool)) & scheduled
        self._offset[ack_mask] += self.ack_step
        self._offset[nack_mask] -= self.nack_step

    def select_mcs(self, sinr_eff_db: np.ndarray,
                   num_allocated_prbs: np.ndarray = None,
                   num_layers: np.ndarray = None) -> np.ndarray:
        """MCS 选择: ILLA 基础 MCS + OLLA 偏移

        Args:
            sinr_eff_db: (num_ue,) 有效 SINR (dB)
            num_allocated_prbs: (num_ue,) 分配 PRB 数 (ILLA 计算 TBLER 用)
            num_layers: (num_ue,) 传输层数

        Returns:
            (num_ue,) 最终 MCS 索引 (int32)
        """
        if num_allocated_prbs is None:
            num_allocated_prbs = np.ones(self.num_ue, dtype=np.int32)
        if num_layers is None:
            num_layers = np.ones(self.num_ue, dtype=np.int32)

        mcs_indices = np.zeros(self.num_ue, dtype=np.int32)
        for ue in range(self.num_ue):
            # 1. ILLA 用原始 SINR 选基础 MCS
            illa_mcs = self.illa.select_mcs(
                float(sinr_eff_db[ue]),
                int(num_allocated_prbs[ue]),
                int(num_layers[ue])
            )
            # 2. 叠加 OLLA offset, 四舍五入, clip 到合法范围
            final_mcs_float = illa_mcs + self._offset[ue]
            final_mcs = int(np.round(final_mcs_float))
            final_mcs = max(0, min(final_mcs, self._max_mcs))
            mcs_indices[ue] = final_mcs

        return mcs_indices

    def reset(self, initial_offset: float = -4.0):
        """重置所有偏移量"""
        self._offset[:] = initial_offset

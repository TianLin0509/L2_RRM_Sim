"""PHY 层抽象接口

统一 Sionna PHY 和 Legacy PHY 的接口，让引擎不再区分两条路径。
"""

from abc import ABC, abstractmethod
import numpy as np


class PHYBase(ABC):
    """PHY 层基类

    所有 PHY 实现 (Sionna, Legacy, Custom) 必须实现此接口。
    引擎通过此接口完成: MCS 选择 → PHY 评估 → 返回解码结果。
    """

    @abstractmethod
    def select_mcs(self, num_allocated_re: np.ndarray,
                   harq_feedback: np.ndarray,
                   sinr_eff: np.ndarray,
                   scheduled_mask: np.ndarray) -> np.ndarray:
        """选择 MCS

        Args:
            num_allocated_re: (num_ue,) 每 UE 估计 RE 数
            harq_feedback: (num_ue,) HARQ 反馈 (1=ACK, 0=NACK, -1=未调度)
            sinr_eff: (num_ue,) 有效 SINR [linear]
            scheduled_mask: (num_ue,) 上一 slot 是否被调度

        Returns:
            (num_ue,) MCS indices
        """
        pass

    @abstractmethod
    def evaluate(self, mcs_indices: np.ndarray,
                 sinr_per_prb: np.ndarray,
                 num_allocated_re: np.ndarray,
                 prb_assignment: np.ndarray,
                 ue_rank: np.ndarray = None,
                 re_per_prb: int = None) -> dict:
        """评估 PHY 传输结果

        Args:
            mcs_indices: (num_ue,) MCS
            sinr_per_prb: (num_ue, max_layers, num_prb) SINR [linear]
            num_allocated_re: (num_ue,) 有效 RE 数
            prb_assignment: (num_prb,) PRB 分配
            ue_rank: (num_ue,) 层数
            re_per_prb: 当前 slot RE/PRB

        Returns:
            dict with keys:
                decoded_bits: (num_ue,) int64
                is_success: (num_ue,) bool
                harq_feedback: (num_ue,) int32 (1=ACK, 0=NACK)
                sinr_eff: (num_ue,) float32 — effective SINR [linear]
                tbler: (num_ue,) float32
                bler: (num_ue,) float32
        """
        pass

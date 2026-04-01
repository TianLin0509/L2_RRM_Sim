"""信道估计器抽象接口"""

from abc import ABC, abstractmethod
import numpy as np
from ..core.data_types import ChannelState


class ChannelEstimatorBase(ABC):
    """信道估计器基类

    所有信道估计算法 (LS, MMSE, DL-based, Perfect) 必须实现此接口。
    """

    @abstractmethod
    def estimate(self, channel_state: ChannelState) -> np.ndarray:
        """根据真实信道计算估计信道

        Args:
            channel_state: 包含 actual_channel_matrix 的信道状态

        Returns:
            estimated_channel_matrix: (num_ue, rx_ant, tx_ant, num_prb) 估计信道
            或 None (无信道矩阵时)
        """
        pass

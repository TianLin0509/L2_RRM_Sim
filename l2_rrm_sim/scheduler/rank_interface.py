"""Rank 自适应抽象接口"""

from abc import ABC, abstractmethod
import numpy as np


class RankAdapterBase(ABC):
    """Rank 自适应基类

    所有 rank 选择策略 (Shannon-based, SVD-based, Fixed, DL-based) 必须实现此接口。
    """

    @abstractmethod
    def select_rank(self, sinr_per_prb: np.ndarray,
                    num_prbs: int = 1) -> int:
        """为单个 UE 选择最优 rank

        Args:
            sinr_per_prb: (max_layers, num_prb) rank-independent SINR [linear]
            num_prbs: 分配的 PRB 数

        Returns:
            最优 rank (1 ~ max_rank)
        """
        pass

    @abstractmethod
    def select_rank_batch(self, sinr_per_prb_all: np.ndarray,
                          num_ue: int, **kwargs) -> np.ndarray:
        """批量选择所有 UE 的 rank

        Args:
            sinr_per_prb_all: (num_ue, max_layers, num_prb) SINR
            num_ue: UE 数

        Returns:
            (num_ue,) rank 数组
        """
        pass

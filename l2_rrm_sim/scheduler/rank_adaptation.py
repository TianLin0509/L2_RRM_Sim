"""秩自适应 (Rank Adaptation)

Phase 1: 固定 rank=1
Phase 2: 基于信道 SVD 的自适应 rank 选择 (rank 1-4)
"""

import numpy as np


class RankAdapter:
    """秩自适应

    评估 rank 1 到 max_rank，选择最大化吞吐量的 rank。
    """

    def __init__(self, max_rank: int = 4, fixed_rank: int = None):
        """
        Args:
            max_rank: 最大允许 rank
            fixed_rank: 若不为 None，强制使用此 rank (Phase 1)
        """
        self.max_rank = max_rank
        self.fixed_rank = fixed_rank

    def select_rank(self, sinr_per_prb: np.ndarray,
                    num_prbs: int = 1) -> int:
        """选择最优 rank

        Args:
            sinr_per_prb: (max_layers, num_prb) per-layer per-PRB SINR [linear]
            num_prbs: 分配的 PRB 数

        Returns:
            选择的 rank (1-max_rank)
        """
        if self.fixed_rank is not None:
            return self.fixed_rank

        # Phase 2: 遍历 rank，选择最大化频谱效率的
        best_rank = 1
        best_se = 0.0

        for r in range(1, min(self.max_rank, sinr_per_prb.shape[0]) + 1):
            # 各层的平均 SINR
            mean_sinr_per_layer = np.mean(sinr_per_prb[:r, :], axis=1)
            # 简化: 用 Shannon 公式估计 SE
            se = np.sum(np.log2(1.0 + mean_sinr_per_layer))
            if se > best_se:
                best_se = se
                best_rank = r

        return best_rank

    def select_rank_batch(self, sinr_per_prb_all: np.ndarray,
                          num_ue: int) -> np.ndarray:
        """批量选择 rank

        Args:
            sinr_per_prb_all: (num_ue, max_layers, num_prb)

        Returns:
            (num_ue,) rank 数组
        """
        ranks = np.ones(num_ue, dtype=np.int32)
        for ue in range(num_ue):
            ranks[ue] = self.select_rank(sinr_per_prb_all[ue])
        return ranks

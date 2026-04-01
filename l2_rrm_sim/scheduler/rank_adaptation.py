"""秩自适应 (Rank Adaptation)

支持两种模式:
1. 基于 SINR 的 Shannon 估计 (无需信道矩阵)
2. 基于信道矩阵 SVD 的精确选择 (需要 channel_matrix)

Rank 选择范围: 1 ~ min(num_rx_ant, num_tx_ports, max_layers)
"""

import numpy as np
from ..utils.nr_utils import compute_tbs, get_spectral_efficiency
from .rank_interface import RankAdapterBase
from ..core.registry import register_rank_adapter


@register_rank_adapter("shannon")
class RankAdapter(RankAdapterBase):
    """秩自适应

    评估 rank 1 到 max_rank，选择最大化期望吞吐量的 rank。
    """

    def __init__(self, max_rank: int = 4, fixed_rank: int = None,
                 num_re_per_prb: int = 132, mcs_table_index: int = 1):
        self.max_rank = max_rank
        self.fixed_rank = fixed_rank
        self.num_re_per_prb = num_re_per_prb
        self.mcs_table_index = mcs_table_index

    def select_rank(self, sinr_per_prb: np.ndarray,
                    num_prbs: int = 1) -> int:
        """基于 SINR 选择最优 rank (Shannon 估计)"""
        if self.fixed_rank is not None:
            return self.fixed_rank

        best_rank = 1
        best_se = 0.0
        available_layers = min(self.max_rank, sinr_per_prb.shape[0])

        for r in range(1, available_layers + 1):
            # sinr_per_prb is rank-independent; apply /r for equal power split
            adjusted_sinr = sinr_per_prb[:r, :] / r
            mean_sinr_per_layer = np.mean(adjusted_sinr, axis=1)
            se = np.sum(np.log2(1.0 + np.maximum(mean_sinr_per_layer, 0)))
            if se > best_se:
                best_se = se
                best_rank = r

        return best_rank

    def select_rank_svd(self, channel_matrix: np.ndarray,
                        tx_power_per_prb: float,
                        noise_power: float,
                        num_prbs: int = 1) -> tuple:
        """基于信道矩阵 SVD 选择 rank

        Args:
            channel_matrix: (num_rx_ant, num_tx_ports, num_prb) 信道矩阵
            tx_power_per_prb: 每 PRB 发射功率 (W)
            noise_power: 噪声功率 (W)
            num_prbs: 分配的 PRB 数

        Returns:
            (rank, precoder, sinr_per_layer):
                rank: 选择的 rank
                precoder: (num_tx_ports, rank, num_prb) 预编码矩阵
                sinr_per_layer: (rank, num_prb) 每层 SINR [linear]
        """
        if self.fixed_rank is not None:
            r = self.fixed_rank
            n_tx = channel_matrix.shape[1]
            n_prb = channel_matrix.shape[2]
            precoder = np.zeros((n_tx, r, n_prb), dtype=complex)
            sinr_per_layer = np.zeros((r, n_prb))
            for prb in range(n_prb):
                U, S, Vh = np.linalg.svd(channel_matrix[:, :, prb], full_matrices=False)
                precoder[:, :r, prb] = Vh[:r, :].conj().T
                for l in range(r):
                    sinr_per_layer[l, prb] = (
                        tx_power_per_prb * S[l]**2 / (r * noise_power)
                    ) if l < len(S) else 0.0
            return r, precoder, sinr_per_layer

        n_rx = channel_matrix.shape[0]
        n_tx = channel_matrix.shape[1]
        n_prb = channel_matrix.shape[2]
        available_layers = min(self.max_rank, n_rx, n_tx)

        # 对每个 PRB 做 SVD，累积各 rank 的 SE
        best_rank = 1
        best_se = 0.0
        singular_values = np.zeros((available_layers, n_prb))

        for prb in range(n_prb):
            U, S, Vh = np.linalg.svd(channel_matrix[:, :, prb], full_matrices=False)
            for l in range(min(available_layers, len(S))):
                singular_values[l, prb] = S[l]

        for r in range(1, available_layers + 1):
            # 等功率分配
            total_se = 0.0
            for l in range(r):
                sinr_l = tx_power_per_prb * singular_values[l, :] ** 2 / (r * noise_power)
                total_se += np.mean(np.log2(1.0 + sinr_l))
            if total_se > best_se:
                best_se = total_se
                best_rank = r

        # 构建预编码矩阵
        precoder = np.zeros((n_tx, best_rank, n_prb), dtype=complex)
        sinr_per_layer = np.zeros((best_rank, n_prb))
        for prb in range(n_prb):
            U, S, Vh = np.linalg.svd(channel_matrix[:, :, prb], full_matrices=False)
            for l in range(best_rank):
                if l < len(S):
                    precoder[:, l, prb] = Vh[l, :].conj().T
                    sinr_per_layer[l, prb] = (
                        tx_power_per_prb * S[l]**2 / (best_rank * noise_power)
                    )

        return best_rank, precoder, sinr_per_layer

    def select_rank_batch(self, sinr_per_prb_all: np.ndarray,
                          num_ue: int,
                          channel_matrix: np.ndarray = None,
                          tx_power_per_prb: float = None,
                          noise_power: float = None) -> np.ndarray:
        """批量选择 rank"""
        ranks = np.ones(num_ue, dtype=np.int32)
        for ue in range(num_ue):
            if channel_matrix is not None and tx_power_per_prb is not None:
                r, _, _ = self.select_rank_svd(
                    channel_matrix[ue], tx_power_per_prb, noise_power
                )
                ranks[ue] = r
            else:
                ranks[ue] = self.select_rank(sinr_per_prb_all[ue])
        return ranks

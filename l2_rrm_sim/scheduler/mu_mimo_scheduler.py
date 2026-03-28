"""MU-MIMO Proportional Fair 调度器

两阶段调度:
1. 基于 PF metric 选候选 UE
2. 基于信道正交性进行 UE 配对

支持 ZF (Zero-Forcing) 和 RZF (Regularized ZF) 预编码。
"""

import numpy as np
from .scheduler_interface import SchedulerBase
from ..core.data_types import (
    SlotContext, UEState, ChannelState, SchedulingDecision
)
from ..utils.nr_utils import compute_tbs


class MUMIMOPFScheduler(SchedulerBase):
    """MU-MIMO PF 调度器

    每 PRB 最多共调度 max_co_ue 个 UE，
    使用 ZF 预编码消除 UE 间干扰。
    """

    def __init__(self, num_ue: int, num_prb: int,
                 num_tx_ports: int = 4,
                 max_co_ue: int = 4,
                 num_re_per_prb: int = 132,
                 mcs_table_index: int = 1,
                 beta: float = 0.98,
                 orthogonality_threshold: float = 0.3):
        """
        Args:
            max_co_ue: 每 PRB 最多共调度的 UE 数
            orthogonality_threshold: 信道方向余弦相似度阈值
                                     小于此值才配对
        """
        self.num_ue = num_ue
        self.num_prb = num_prb
        self.num_tx_ports = num_tx_ports
        self.max_co_ue = min(max_co_ue, num_tx_ports)
        self.num_re_per_prb = num_re_per_prb
        self.mcs_table_index = mcs_table_index
        self.beta = beta
        self.ortho_threshold = orthogonality_threshold

        self._t_avg = np.ones(num_ue, dtype=np.float64)

    @property
    def throughput_avg(self) -> np.ndarray:
        return self._t_avg.copy()

    def schedule(self, slot_ctx: SlotContext,
                 ue_states: list,
                 channel_state: ChannelState,
                 achievable_rate_per_prb: np.ndarray,
                 ue_buffer_bytes: np.ndarray,
                 ue_mcs: np.ndarray,
                 ue_rank: np.ndarray) -> SchedulingDecision:
        """MU-MIMO PF 调度

        如果没有信道矩阵，退化为 SU-MIMO。
        """
        has_data = ue_buffer_bytes > 0
        num_ue = self.num_ue
        num_prb = self.num_prb
        has_channel_matrix = (channel_state.channel_matrix is not None)

        # PF metric
        pf_metric = np.zeros((num_ue, num_prb))
        for ue in range(num_ue):
            if has_data[ue]:
                pf_metric[ue, :] = (
                    achievable_rate_per_prb[ue, :] / max(self._t_avg[ue], 1e-10)
                )

        # PRB 分配: 支持每 PRB 多 UE
        # prb_assignment[prb] = primary UE (-1 if unassigned)
        # mu_mimo_groups[prb] = list of co-scheduled UEs
        prb_assignment = np.full(num_prb, -1, dtype=np.int32)
        mu_mimo_groups = [[] for _ in range(num_prb)]

        if has_channel_matrix:
            self._mu_mimo_schedule(
                pf_metric, has_data, channel_state.channel_matrix,
                prb_assignment, mu_mimo_groups
            )
        else:
            # 退化为 SU-MIMO
            valid_metric = pf_metric.copy()
            valid_metric[~has_data, :] = 0
            primary = np.argmax(valid_metric, axis=0)
            max_val = np.max(valid_metric, axis=0)
            primary[max_val <= 0] = -1
            prb_assignment[:] = primary
            for prb in range(num_prb):
                if prb_assignment[prb] >= 0:
                    mu_mimo_groups[prb] = [int(prb_assignment[prb])]

        # 统计每 UE 的分配
        ue_num_prbs = np.zeros(num_ue, dtype=np.int32)
        ue_effective_rank = np.ones(num_ue, dtype=np.int32)

        for prb in range(num_prb):
            for ue in mu_mimo_groups[prb]:
                ue_num_prbs[ue] += 1

        # MU-MIMO 时，每 UE 的有效 rank = 1 (每 UE 单流)
        # SU-MIMO 时，使用原始 rank
        for ue in range(num_ue):
            if ue_num_prbs[ue] > 0:
                # 检查该 UE 是否参与了 MU-MIMO (共调度)
                is_mu = any(len(mu_mimo_groups[prb]) > 1
                            for prb in range(num_prb)
                            if ue in mu_mimo_groups[prb])
                ue_effective_rank[ue] = 1 if is_mu else int(ue_rank[ue])

        # 计算 TBS
        ue_tbs = np.zeros(num_ue, dtype=np.int64)
        ue_num_re = np.zeros(num_ue, dtype=np.int64)
        for ue in range(num_ue):
            if ue_num_prbs[ue] > 0:
                ue_tbs[ue] = compute_tbs(
                    self.num_re_per_prb, int(ue_num_prbs[ue]),
                    int(ue_mcs[ue]), int(ue_effective_rank[ue]),
                    self.mcs_table_index
                )
                ue_num_re[ue] = self.num_re_per_prb * ue_num_prbs[ue]

        # 用第一个 UE 的 ID 作为 prb_assignment (兼容 SU 接口)
        for prb in range(num_prb):
            if mu_mimo_groups[prb]:
                prb_assignment[prb] = mu_mimo_groups[prb][0]

        return SchedulingDecision(
            prb_assignment=prb_assignment,
            ue_mcs=ue_mcs.copy(),
            ue_rank=ue_effective_rank,
            ue_num_prbs=ue_num_prbs,
            ue_tbs_bits=ue_tbs,
            ue_num_re=ue_num_re,
        )

    def _mu_mimo_schedule(self, pf_metric, has_data, channel_matrix,
                          prb_assignment, mu_mimo_groups):
        """MU-MIMO 配对调度"""
        num_ue = self.num_ue
        num_prb = self.num_prb

        for prb in range(num_prb):
            # 候选 UE: 有数据且 PF metric > 0
            candidates = [ue for ue in range(num_ue)
                          if has_data[ue] and pf_metric[ue, prb] > 0]
            if not candidates:
                continue

            # 按 PF metric 排序
            candidates.sort(key=lambda ue: pf_metric[ue, prb], reverse=True)

            # 贪心配对
            paired = [candidates[0]]
            prb_assignment[prb] = candidates[0]

            if len(candidates) > 1 and self.max_co_ue > 1:
                # 第一个 UE 的信道方向
                h_ref = channel_matrix[candidates[0], 0, :, prb]  # (tx_ports,)
                if np.linalg.norm(h_ref) < 1e-10:
                    mu_mimo_groups[prb] = paired
                    continue
                h_ref_norm = h_ref / np.linalg.norm(h_ref)
                paired_directions = [h_ref_norm]

                for ue in candidates[1:]:
                    if len(paired) >= self.max_co_ue:
                        break
                    h_ue = channel_matrix[ue, 0, :, prb]
                    if np.linalg.norm(h_ue) < 1e-10:
                        continue
                    h_ue_norm = h_ue / np.linalg.norm(h_ue)

                    # 检查与所有已配对 UE 的正交性
                    is_orthogonal = True
                    for h_paired in paired_directions:
                        similarity = np.abs(np.dot(h_ue_norm.conj(), h_paired))
                        if similarity > self.ortho_threshold:
                            is_orthogonal = False
                            break

                    if is_orthogonal:
                        paired.append(ue)
                        paired_directions.append(h_ue_norm)

            mu_mimo_groups[prb] = paired

    def update_throughput_history(self, ue_throughput_bits: np.ndarray):
        """更新 PF 时间平均吞吐量"""
        self._t_avg = (
            self.beta * self._t_avg
            + (1.0 - self.beta) * ue_throughput_bits
        )
        self._t_avg = np.maximum(self._t_avg, 1e-10)


def compute_zf_precoder(channel_matrix: np.ndarray,
                        paired_ues: list,
                        prb_idx: int) -> np.ndarray:
    """计算 ZF 预编码矩阵

    Args:
        channel_matrix: (num_ue, rx_ant, tx_ports, num_prb)
        paired_ues: 配对 UE 列表
        prb_idx: PRB 索引

    Returns:
        W: (tx_ports, num_paired_ue) ZF 预编码矩阵
    """
    num_paired = len(paired_ues)
    tx_ports = channel_matrix.shape[2]

    # 构建等效信道矩阵 H_eq: (num_paired, tx_ports)
    # 每 UE 取第一根天线
    H_eq = np.zeros((num_paired, tx_ports), dtype=complex)
    for i, ue in enumerate(paired_ues):
        H_eq[i, :] = channel_matrix[ue, 0, :, prb_idx]

    # ZF: W = H^H (H H^H)^{-1}
    try:
        HHH = H_eq @ H_eq.conj().T  # (K, K)
        HHH_inv = np.linalg.inv(HHH + 1e-10 * np.eye(num_paired))
        W = H_eq.conj().T @ HHH_inv  # (tx_ports, K)

        # 功率归一化 (每列)
        for k in range(num_paired):
            norm = np.linalg.norm(W[:, k])
            if norm > 0:
                W[:, k] /= norm
    except np.linalg.LinAlgError:
        W = np.eye(tx_ports, num_paired, dtype=complex)

    return W


def compute_mu_mimo_sinr(channel_matrix: np.ndarray,
                         paired_ues: list,
                         prb_idx: int,
                         tx_power_per_prb: float,
                         noise_power: float) -> np.ndarray:
    """计算 MU-MIMO ZF 后各 UE 的 SINR

    Returns:
        sinr: (num_paired,) SINR [linear]
    """
    W = compute_zf_precoder(channel_matrix, paired_ues, prb_idx)
    num_paired = len(paired_ues)
    sinr = np.zeros(num_paired)

    # 每 UE 分到的功率 = 总功率 / 配对数
    power_per_ue = tx_power_per_prb / num_paired

    for i, ue in enumerate(paired_ues):
        h = channel_matrix[ue, 0, :, prb_idx]  # (tx_ports,)
        # 有效信号
        signal = np.abs(np.dot(h, W[:, i])) ** 2 * power_per_ue
        # 干扰 (来自其他流)
        interference = 0.0
        for j in range(num_paired):
            if j != i:
                interference += np.abs(np.dot(h, W[:, j])) ** 2 * power_per_ue
        sinr[i] = signal / (interference + noise_power)

    return sinr

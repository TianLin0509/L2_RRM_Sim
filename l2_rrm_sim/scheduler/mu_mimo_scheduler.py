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
from ..core.registry import register_scheduler


@register_scheduler("mu_mimo")
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

        使用估计信道 (estimated_channel_matrix) 进行调度决策。
        """
        has_data = ue_buffer_bytes > 0
        num_ue = self.num_ue
        num_prb = self.num_prb
        
        # 优先使用估计信道进行调度决策
        h_est = channel_state.estimated_channel_matrix
        if h_est is None:
            h_est = channel_state.actual_channel_matrix
            
        has_channel_matrix = (h_est is not None)

        # PF metric
        pf_metric = np.zeros((num_ue, num_prb))
        for ue in range(num_ue):
            if has_data[ue]:
                pf_metric[ue, :] = (
                    achievable_rate_per_prb[ue, :] / max(self._t_avg[ue], 1e-10)
                )

        # PRB 分配
        prb_assignment = np.full(num_prb, -1, dtype=np.int32)
        mu_mimo_groups = [[] for _ in range(num_prb)]

        if has_channel_matrix:
            self._mu_mimo_schedule(
                pf_metric, has_data, h_est,
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

        # 统计每 UE 的分配（向量化）
        ue_num_prbs = np.zeros(num_ue, dtype=np.int32)
        ue_effective_rank = ue_rank.copy().astype(np.int32)
        group_sizes = np.array([len(g) for g in mu_mimo_groups], dtype=np.int32)

        for prb in range(num_prb):
            for ue in mu_mimo_groups[prb]:
                ue_num_prbs[ue] += 1

        # MU-MIMO 时，有效 rank = 1（向量化：找出出现在 size>1 组中的 UE）
        mu_prbs = np.where(group_sizes > 1)[0]
        if mu_prbs.size > 0:
            mu_ue_set = set()
            for prb in mu_prbs:
                mu_ue_set.update(mu_mimo_groups[prb])
            for ue in mu_ue_set:
                ue_effective_rank[ue] = 1

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

        # 用第一个 UE 的 ID 作为 prb_assignment
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
            mu_groups=mu_mimo_groups,  # 关键：传出配对组信息
        )

    def _mu_mimo_schedule(self, pf_metric, has_data, h_est,
                          prb_assignment, mu_mimo_groups):
        """MU-MIMO 配对调度 (基于估计信道)"""
        num_ue = self.num_ue
        num_prb = self.num_prb

        for prb in range(num_prb):
            candidates = [ue for ue in range(num_ue)
                          if has_data[ue] and pf_metric[ue, prb] > 0]
            if not candidates:
                continue

            candidates.sort(key=lambda ue: pf_metric[ue, prb], reverse=True)

            paired = [candidates[0]]
            prb_assignment[prb] = candidates[0]

            if len(candidates) > 1 and self.max_co_ue > 1:
                # 提取第一个 UE 的等效信道 (dominant RX antenna)
                h_ue_full = h_est[candidates[0], :, :, prb]
                best_rx = np.argmax(np.sum(np.abs(h_ue_full)**2, axis=1))
                h_ref = h_ue_full[best_rx, :]
                
                if np.linalg.norm(h_ref) < 1e-10:
                    mu_mimo_groups[prb] = paired
                    continue
                h_ref_norm = h_ref / np.linalg.norm(h_ref)
                paired_directions = [h_ref_norm]

                for ue in candidates[1:]:
                    if len(paired) >= self.max_co_ue:
                        break
                    
                    h_ue_all = h_est[ue, :, :, prb]
                    b_rx = np.argmax(np.sum(np.abs(h_ue_all)**2, axis=1))
                    h_ue = h_ue_all[b_rx, :]
                    
                    if np.linalg.norm(h_ue) < 1e-10:
                        continue
                    h_ue_norm = h_ue / np.linalg.norm(h_ue)

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


def compute_zf_precoder(h_est: np.ndarray,
                        paired_ues: list,
                        prb_idx: int) -> np.ndarray:
    """计算 ZF 预编码矩阵 (基于估计信道)"""
    num_paired = len(paired_ues)
    tx_ports = h_est.shape[2]

    # 构建等效信道矩阵 H_eq: (num_paired, tx_ports)
    H_eq = np.zeros((num_paired, tx_ports), dtype=complex)
    for i, ue in enumerate(paired_ues):
        h_ue = h_est[ue, :, :, prb_idx]
        best_rx = np.argmax(np.sum(np.abs(h_ue)**2, axis=1))
        H_eq[i, :] = h_ue[best_rx, :]

    # ZF: W = H^H (H H^H)^{-1}
    try:
        HHH = H_eq @ H_eq.conj().T
        HHH_inv = np.linalg.inv(HHH + 1e-10 * np.eye(num_paired))
        W = H_eq.conj().T @ HHH_inv

        # 功率归一化
        for k in range(num_paired):
            norm = np.linalg.norm(W[:, k])
            if norm > 0:
                W[:, k] /= norm
    except np.linalg.LinAlgError:
        W = np.eye(tx_ports, num_paired, dtype=complex)

    return W


def compute_mu_mimo_sinr(h_actual: np.ndarray,
                         h_est: np.ndarray,
                         paired_ues: list,
                         prb_idx: int,
                         tx_power_per_prb: float,
                         noise_power: float) -> np.ndarray:
    """计算 MU-MIMO ZF 后各 UE 的真实 SINR (使用 MMSE-IRC 接收机)

    MMSE-IRC 能够利用多天线空间特性抑制配对 UE 间的残余干扰。
    
    Args:
        h_actual: 真实信道 (Ground Truth), (num_ue, rx_ant, tx_ports, num_prb)
        h_est: 估计信道 (gNB 侧用于预编码设计)
        paired_ues: 当前 PRB 配对的 UE ID 列表
        prb_idx: PRB 索引
        tx_power_per_prb: 该 PRB 总发射功率
        noise_power: 接收端高斯噪声功率
    """
    # 预编码设计基于 gNB 侧的估计信道
    W = compute_zf_precoder(h_est, paired_ues, prb_idx)
    num_paired = len(paired_ues)
    num_rx_ant = h_actual.shape[1]
    sinr = np.zeros(num_paired)

    # 假设各流功率均分
    power_per_ue = tx_power_per_prb / num_paired

    for i, ue_id in enumerate(paired_ues):
        # UE 看到的实际信道 H: (rx_ant, tx_ports)
        H = h_actual[ue_id, :, :, prb_idx]
        
        # 有效信道向量 h_eff_i = H * W_i: (rx_ant,)
        h_eff_target = H @ W[:, i]
        
        # 构建干扰+噪声协方差矩阵 R_in (K_rx x K_rx)
        # R_in = sum_{j != i} (H * W_j) * (H * W_j)^H * P + sigma^2 * I
        R_in = noise_power * np.eye(num_rx_ant, dtype=complex)
        
        for j in range(num_paired):
            if j != i:
                h_eff_interf = H @ W[:, j] # (rx_ant,)
                # 外积得到干扰协方差
                R_in += power_per_ue * np.outer(h_eff_interf, h_eff_interf.conj())
        
        # MMSE-IRC SINR 公式: SINR = P * h_eff^H * R_in^-1 * h_eff
        try:
            # 这里的 R_in 必须是正定的 (由于包含 noise_power * I，通常是可逆的)
            R_in_inv = np.linalg.inv(R_in)
            # 计算后处理 SINR
            # sinr = P * (h^H @ R_inv @ h)
            val = power_per_ue * (h_eff_target.conj().T @ R_in_inv @ h_eff_target)
            sinr[i] = np.abs(val)
        except np.linalg.LinAlgError:
            # 回退到 MRC 如果矩阵奇异
            sinr[i] = power_per_ue * np.abs(np.dot(h_eff_target.conj(), h_eff_target)) / (noise_power + 1e-10)

    return sinr

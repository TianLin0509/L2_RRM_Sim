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
                 orthogonality_threshold: float = 0.5,
                 resource_grid=None,
                 merge_small_packet: bool = True,
                 merge_rb_ratio: float = 0.1,
                 separate_big_packet: bool = True,
                 big_packet_ratio: float = 5.0,
                 max_groups: int = 4):
        """
        Args:
            max_co_ue: 每 PRB 最多共调度的 UE 数
            orthogonality_threshold: 预编码权值相关性阈值 (< 此值才配对)
            merge_small_packet: 启用小包合并 (对标 AirView MubfMergeGroup)
            merge_rb_ratio: 小包 RB 占比门限
            separate_big_packet: 启用大包分离 (对标 AirView MubfSeparateBigGroup)
            big_packet_ratio: 大包判定倍数 (buffer > 平均 * ratio)
            max_groups: 最大分组数 (对标 AirView, 通常 1-4)
        """
        self.num_ue = num_ue
        self.num_prb = num_prb
        self.num_tx_ports = num_tx_ports
        self.max_co_ue = min(max_co_ue, num_tx_ports)
        self.num_re_per_prb = num_re_per_prb
        self.mcs_table_index = mcs_table_index
        self.beta = beta
        self.ortho_threshold = orthogonality_threshold
        self._resource_grid = resource_grid
        self.merge_small_packet = merge_small_packet
        self.merge_rb_ratio = merge_rb_ratio
        self.separate_big_packet = separate_big_packet
        self.big_packet_ratio = big_packet_ratio
        self.max_groups = max_groups

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
                 ue_rank: np.ndarray,
                 re_per_prb: int = None) -> SchedulingDecision:
        """MU-MIMO PF 调度

        使用估计信道 (estimated_channel_matrix) 进行调度决策。
        """
        has_data = ue_buffer_bytes > 0
        ue_buffer_bits = ue_buffer_bytes.astype(np.int64) * 8
        slot_re_per_prb = re_per_prb if re_per_prb is not None else self.num_re_per_prb
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

        # 计算 TBS (使用当前 slot 实际 RE/PRB + buffer 限幅)
        ue_tbs = np.zeros(num_ue, dtype=np.int64)
        ue_num_re = np.zeros(num_ue, dtype=np.int64)
        for ue in range(num_ue):
            if ue_num_prbs[ue] > 0:
                raw_tbs = compute_tbs(
                    slot_re_per_prb, int(ue_num_prbs[ue]),
                    int(ue_mcs[ue]), int(ue_effective_rank[ue]),
                    self.mcs_table_index
                )
                ue_tbs[ue] = min(raw_tbs, max(ue_buffer_bits[ue], 0))
                ue_num_re[ue] = slot_re_per_prb * ue_num_prbs[ue]

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

    def _get_precoding_weight(self, h_est, ue, prb):
        """获取 UE 的预编码权值向量 (用于相关性配对)

        优先使用 SVD 主特征向量 (对标 AirView 的预编码权值相关性)，
        回退到 dominant RX antenna 信道向量。
        """
        h_ue = h_est[ue, :, :, prb]  # (rx_ant, tx_ports)
        # SVD → 主右奇异向量作为预编码权值
        try:
            _, _, Vh = np.linalg.svd(h_ue, full_matrices=False)
            w = Vh[0, :].conj()  # (tx_ports,) 主预编码方向
        except np.linalg.LinAlgError:
            best_rx = np.argmax(np.sum(np.abs(h_ue)**2, axis=1))
            w = h_ue[best_rx, :]
        norm = np.linalg.norm(w)
        return w / norm if norm > 1e-10 else w

    def _mu_mimo_schedule(self, pf_metric, has_data, h_est,
                          prb_assignment, mu_mimo_groups):
        """MU-MIMO 配对调度 (基于预编码权值相关性)

        对标 AirView CorrGroup: |w1^H·w2|/(‖w1‖·‖w2‖) < threshold
        """
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
                # 获取第一个 UE 的预编码权值
                w_ref = self._get_precoding_weight(h_est, candidates[0], prb)
                if np.linalg.norm(w_ref) < 1e-10:
                    mu_mimo_groups[prb] = paired
                    continue
                paired_weights = [w_ref]

                for ue in candidates[1:]:
                    if len(paired) >= self.max_co_ue:
                        break

                    w_ue = self._get_precoding_weight(h_est, ue, prb)
                    if np.linalg.norm(w_ue) < 1e-10:
                        continue

                    # 预编码权值相关性检查 (对标 AirView CorrGroup)
                    is_orthogonal = True
                    for w_paired in paired_weights:
                        corr = np.abs(np.dot(w_ue.conj(), w_paired))
                        if corr > self.ortho_threshold:
                            is_orthogonal = False
                            break

                    if is_orthogonal:
                        paired.append(ue)
                        paired_weights.append(w_ue)

            mu_mimo_groups[prb] = paired

    def update_throughput_history(self, ue_throughput_bits: np.ndarray):
        """更新 PF 时间平均吞吐量"""
        self._t_avg = (
            self.beta * self._t_avg
            + (1.0 - self.beta) * ue_throughput_bits
        )
        self._t_avg = np.maximum(self._t_avg, 1e-10)


def normalize_precoder(W: np.ndarray, mode: str = 'nebf') -> np.ndarray:
    """天线级功率归一化 (对标 AirView WeightNPEBFUnit)

    Args:
        W: (num_tx, num_streams) 预编码矩阵
        mode: 'nebf' (SU: 每天线等功率, 总功率=额定) 或
              'pebf' (MU: 最大天线功率归一化, 总功率≤额定)

    Returns:
        归一化后的预编码矩阵
    """
    W = W.copy()
    num_tx = W.shape[0]
    if num_tx == 0:
        return W

    # 每根天线的总功率: P_ant[n] = sum_k |W[n,k]|^2
    ant_power = np.sum(np.abs(W)**2, axis=1)  # (num_tx,)

    if mode == 'nebf':
        # NEBF: 每天线独立归一化到 1/num_tx → 总功率=1
        for n in range(num_tx):
            if ant_power[n] > 1e-20:
                W[n, :] *= np.sqrt(1.0 / (ant_power[n] * num_tx))
    elif mode == 'pebf':
        # PEBF: 最大天线功率归一化 → 总功率≤1
        max_power = np.max(ant_power)
        if max_power > 1e-20:
            W *= np.sqrt(1.0 / (max_power * num_tx))
    else:
        # 回退: 列归一化 (原始行为)
        for k in range(W.shape[1]):
            norm = np.linalg.norm(W[:, k])
            if norm > 0:
                W[:, k] /= norm
    return W


def compute_zf_precoder(h_est: np.ndarray,
                        paired_ues: list,
                        prb_idx: int,
                        power_norm: str = 'pebf') -> np.ndarray:
    """计算 ZF 预编码矩阵 (基于估计信道)

    Args:
        power_norm: 'nebf', 'pebf', 或 'column' (原始列归一化)
    """
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

        # 天线级功率归一化 (NEBF/PEBF)
        W = normalize_precoder(W, mode=power_norm)
    except np.linalg.LinAlgError:
        W = np.eye(tx_ports, num_paired, dtype=complex)

    return W


def compute_mu_mimo_sinr(h_actual: np.ndarray,
                         h_est: np.ndarray,
                         paired_ues: list,
                         prb_idx: int,
                         tx_power_per_prb: float,
                         noise_power: float,
                         ue_rank: np.ndarray = None) -> np.ndarray:
    """计算 MU-MIMO ZF 后各 UE 的真实 SINR (使用 MMSE-IRC 接收机)

    MMSE-IRC 能够利用多天线空间特性抑制配对 UE 间的残余干扰。
    功率按 rank/maxLayer 比例分配 (对标 AirView MuCaclRbforSuMuADPWithPowerCoff)。

    Args:
        h_actual: 真实信道 (Ground Truth), (num_ue, rx_ant, tx_ports, num_prb)
        h_est: 估计信道 (gNB 侧用于预编码设计)
        paired_ues: 当前 PRB 配对的 UE ID 列表
        prb_idx: PRB 索引
        tx_power_per_prb: 该 PRB 总发射功率
        noise_power: 接收端高斯噪声功率
        ue_rank: (num_ue,) 每 UE 的 rank (用于功率系数计算)
    """
    # 预编码设计基于 gNB 侧的估计信道 (PEBF 归一化)
    W = compute_zf_precoder(h_est, paired_ues, prb_idx, power_norm='pebf')
    num_paired = len(paired_ues)
    num_rx_ant = h_actual.shape[1]
    sinr = np.zeros(num_paired)

    # 功率按 rank/maxLayer 比例分配 (对标 AirView)
    if ue_rank is not None and num_paired > 1:
        ranks = np.array([max(int(ue_rank[ue]), 1) for ue in paired_ues], dtype=np.float64)
        max_layer = np.max(ranks)
        power_ratios = ranks / max_layer  # rank/maxLayer
        power_sum = np.sum(power_ratios)
        power_per_ue = tx_power_per_prb * power_ratios / power_sum
    else:
        power_per_ue = np.full(num_paired, tx_power_per_prb / max(num_paired, 1))

    for i, ue_id in enumerate(paired_ues):
        # UE 看到的实际信道 H: (rx_ant, tx_ports)
        H = h_actual[ue_id, :, :, prb_idx]

        # 有效信道向量 h_eff_i = H * W_i: (rx_ant,)
        h_eff_target = H @ W[:, i]

        # 构建干扰+噪声协方差矩阵 R_in (K_rx x K_rx)
        R_in = noise_power * np.eye(num_rx_ant, dtype=complex)

        for j in range(num_paired):
            if j != i:
                h_eff_interf = H @ W[:, j]
                R_in += power_per_ue[j] * np.outer(h_eff_interf, h_eff_interf.conj())

        # MMSE-IRC SINR: P_i * h^H * R_in^{-1} * h
        try:
            R_in_inv = np.linalg.inv(R_in)
            val = power_per_ue[i] * (h_eff_target.conj().T @ R_in_inv @ h_eff_target)
            sinr[i] = np.abs(val)
        except np.linalg.LinAlgError:
            sinr[i] = power_per_ue[i] * np.abs(np.dot(h_eff_target.conj(), h_eff_target)) / (noise_power + 1e-10)

    return sinr

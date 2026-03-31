"""Proportional Fair (PF) 调度器 — SU-MIMO

调度粒度: RBG (Resource Block Group), 符合 TS 38.214
PF metric: R_achievable[ue, rbg] / T_avg[ue]
每 RBG 分配给 metric 最大的 UE。
只调度有数据的 UE (buffer-aware)。
"""

import numpy as np
from .scheduler_interface import SchedulerBase
from ..core.data_types import (
    SlotContext, UEState, ChannelState, SchedulingDecision
)
from ..utils.nr_utils import compute_tbs


class PFSchedulerSUMIMO(SchedulerBase):
    """SU-MIMO Proportional Fair 调度器 (RBG 粒度)

    T_avg[ue] = beta × T_avg[ue] + (1-beta) × R_last[ue]
    调度以 RBG 为单位分配, 展开为 PRB 级结果。
    """

    def __init__(self, num_ue: int, num_prb: int,
                 num_re_per_prb: int = 132,
                 mcs_table_index: int = 1,
                 beta: float = 0.98,
                 resource_grid=None):
        self.num_ue = num_ue
        self.num_prb = num_prb
        self.num_re_per_prb = num_re_per_prb
        self.mcs_table_index = mcs_table_index
        self.beta = beta
        self._resource_grid = resource_grid

        # RBG 参数
        if resource_grid is not None:
            self.rbg_size = resource_grid.rbg_size
            self.num_rbg = resource_grid.num_rbg
        else:
            # 回退: 16 PRB per RBG (100MHz@30kHz 默认)
            from ..core.resource_grid import get_rbg_size
            self.rbg_size = get_rbg_size(num_prb)
            self.num_rbg = (num_prb + self.rbg_size - 1) // self.rbg_size

        # 时间平均吞吐量
        self._t_avg = np.ones(num_ue, dtype=np.float64)

    @property
    def throughput_avg(self) -> np.ndarray:
        return self._t_avg.copy()

    _LARGE_BUFFER_THRESHOLD = 10**7 * 8  # bits, fast path if all UEs above this

    def _rbg_prb_count(self, rbg: int) -> int:
        """Return number of PRBs in a given RBG."""
        s = rbg * self.rbg_size
        e = min(s + self.rbg_size, self.num_prb)
        return e - s

    def schedule(self, slot_ctx: SlotContext,
                 ue_states: list,
                 channel_state: ChannelState,
                 achievable_rate_per_prb: np.ndarray,
                 ue_buffer_bytes: np.ndarray,
                 ue_mcs: np.ndarray,
                 ue_rank: np.ndarray,
                 re_per_prb: int = None) -> SchedulingDecision:
        """PF 调度 (RBG 粒度), buffer-aware

        Args:
            re_per_prb: 当前 slot 实际 RE/PRB (TDD special slot 时 < 132)
        """
        num_ue = self.num_ue
        num_prb = self.num_prb
        slot_re_per_prb = re_per_prb if re_per_prb is not None else self.num_re_per_prb

        has_data = ue_buffer_bytes > 0
        ue_buffer_bits = ue_buffer_bytes.astype(np.int64) * 8

        # 将 per-PRB achievable rate 聚合到 per-RBG
        if self._resource_grid is not None:
            rate_per_rbg = self._resource_grid.aggregate_prb_to_rbg(achievable_rate_per_prb)
        else:
            rate_per_rbg = self._aggregate_to_rbg(achievable_rate_per_prb)

        # PF metric per RBG: (num_ue, num_rbg)
        pf_metric = np.zeros((num_ue, self.num_rbg))
        for ue in range(num_ue):
            if has_data[ue]:
                pf_metric[ue, :] = rate_per_rbg[ue, :] / max(self._t_avg[ue], 1e-10)

        # Fast path: if all active UEs have large buffers, simple argmax
        active_mask = has_data
        all_large = np.all(ue_buffer_bits[active_mask] >= self._LARGE_BUFFER_THRESHOLD) if np.any(active_mask) else True

        if all_large:
            # 每 RBG 分配给 metric 最���的 UE
            rbg_assignment = np.argmax(pf_metric, axis=0)  # (num_rbg,)
            max_metric = np.max(pf_metric, axis=0)
            rbg_assignment = rbg_assignment.astype(np.int32)
            rbg_assignment[max_metric <= 0] = -1
        else:
            # Greedy RBG-by-RBG allocation with buffer satisfaction check
            rbg_assignment = np.full(self.num_rbg, -1, dtype=np.int32)
            remaining_bits = ue_buffer_bits.copy()
            # Sort RBGs by max metric descending for better allocation
            rbg_order = np.argsort(-np.max(pf_metric, axis=0))
            for rbg in rbg_order:
                best_ue = -1
                best_val = 0.0
                for ue in range(num_ue):
                    if remaining_bits[ue] <= 0:
                        continue
                    if pf_metric[ue, rbg] > best_val:
                        best_val = pf_metric[ue, rbg]
                        best_ue = ue
                if best_ue >= 0:
                    rbg_assignment[rbg] = best_ue
                    # Estimate bits this RBG contributes
                    prb_cnt = self._rbg_prb_count(rbg)
                    est_bits = rate_per_rbg[best_ue, rbg] * prb_cnt
                    remaining_bits[best_ue] -= int(est_bits)

        # 展开 RBG → PRB
        if self._resource_grid is not None:
            prb_assignment = self._resource_grid.expand_rbg_to_prb(rbg_assignment)
        else:
            prb_assignment = self._expand_to_prb(rbg_assignment)

        # 统计每 UE 分配的 PRB ���
        ue_num_prbs = np.bincount(
            prb_assignment[prb_assignment >= 0],
            minlength=num_ue
        ).astype(np.int32)[:num_ue]

        # 计算 TBS 和 RE 数 (使用当前 slot 实际 RE/PRB)
        ue_tbs = np.zeros(num_ue, dtype=np.int64)
        ue_num_re = np.zeros(num_ue, dtype=np.int64)
        for ue in range(num_ue):
            if ue_num_prbs[ue] > 0:
                raw_tbs = compute_tbs(
                    slot_re_per_prb, int(ue_num_prbs[ue]),
                    int(ue_mcs[ue]), int(ue_rank[ue]),
                    self.mcs_table_index
                )
                ue_tbs[ue] = min(raw_tbs, ue_buffer_bits[ue]) if ue_buffer_bits[ue] > 0 else raw_tbs
                ue_num_re[ue] = slot_re_per_prb * ue_num_prbs[ue]

        return SchedulingDecision(
            prb_assignment=prb_assignment,
            ue_mcs=ue_mcs.copy(),
            ue_rank=ue_rank.copy(),
            ue_num_prbs=ue_num_prbs,
            ue_tbs_bits=ue_tbs,
            ue_num_re=ue_num_re,
        )

    def _aggregate_to_rbg(self, per_prb: np.ndarray) -> np.ndarray:
        """回退: 无 resource_grid 时手动聚合"""
        result = np.zeros((per_prb.shape[0], self.num_rbg))
        for rbg in range(self.num_rbg):
            s = rbg * self.rbg_size
            e = min(s + self.rbg_size, self.num_prb)
            result[:, rbg] = np.mean(per_prb[:, s:e], axis=1)
        return result

    def _expand_to_prb(self, rbg_assignment: np.ndarray) -> np.ndarray:
        """回退: 无 resource_grid 时手动展开"""
        prb_assignment = np.full(self.num_prb, -1, dtype=np.int32)
        for rbg in range(self.num_rbg):
            s = rbg * self.rbg_size
            e = min(s + self.rbg_size, self.num_prb)
            prb_assignment[s:e] = rbg_assignment[rbg]
        return prb_assignment

    def update_throughput_history(self, ue_throughput_bits: np.ndarray):
        """更新 PF 时间平均吞吐量"""
        self._t_avg = (
            self.beta * self._t_avg
            + (1.0 - self.beta) * ue_throughput_bits
        )
        self._t_avg = np.maximum(self._t_avg, 1e-10)

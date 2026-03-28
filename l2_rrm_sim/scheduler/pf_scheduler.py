"""Proportional Fair (PF) 调度器 — SU-MIMO

PF metric: R_achievable[ue, prb] / T_avg[ue]
每 PRB 分配给 metric 最大的 UE。
只调度有数据的 UE (buffer-aware)。
"""

import numpy as np
from .scheduler_interface import SchedulerBase
from ..core.data_types import (
    SlotContext, UEState, ChannelState, SchedulingDecision
)
from ..utils.nr_utils import compute_tbs


class PFSchedulerSUMIMO(SchedulerBase):
    """SU-MIMO Proportional Fair 调度器

    T_avg[ue] = β × T_avg[ue] + (1-β) × R_last[ue]
    """

    def __init__(self, num_ue: int, num_prb: int,
                 num_re_per_prb: int = 132,
                 mcs_table_index: int = 1,
                 beta: float = 0.98):
        self.num_ue = num_ue
        self.num_prb = num_prb
        self.num_re_per_prb = num_re_per_prb
        self.mcs_table_index = mcs_table_index
        self.beta = beta

        # 时间平均吞吐量 (初始化为 1 避免除零)
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
        """PF 调度"""
        num_ue = self.num_ue
        num_prb = self.num_prb

        # 只调度有数据的 UE
        has_data = ue_buffer_bytes > 0  # (num_ue,)

        # PF metric: (num_ue, num_prb)
        pf_metric = np.zeros((num_ue, num_prb))
        for ue in range(num_ue):
            if has_data[ue]:
                pf_metric[ue, :] = (
                    achievable_rate_per_prb[ue, :] / max(self._t_avg[ue], 1e-10)
                )

        # 每 PRB 分配给 metric 最大的 UE
        prb_assignment = np.argmax(pf_metric, axis=0)  # (num_prb,)

        # 如果某 PRB 上所有 metric 为 0，标记为 -1
        max_metric = np.max(pf_metric, axis=0)
        prb_assignment = prb_assignment.astype(np.int32)
        prb_assignment[max_metric <= 0] = -1

        # 统计每 UE 分配的 PRB 数
        ue_num_prbs = np.zeros(num_ue, dtype=np.int32)
        for ue in range(num_ue):
            ue_num_prbs[ue] = np.sum(prb_assignment == ue)

        # 计算 TBS 和 RE 数
        ue_tbs = np.zeros(num_ue, dtype=np.int64)
        ue_num_re = np.zeros(num_ue, dtype=np.int64)
        for ue in range(num_ue):
            if ue_num_prbs[ue] > 0:
                ue_tbs[ue] = compute_tbs(
                    self.num_re_per_prb, int(ue_num_prbs[ue]),
                    int(ue_mcs[ue]), int(ue_rank[ue]),
                    self.mcs_table_index
                )
                ue_num_re[ue] = self.num_re_per_prb * ue_num_prbs[ue]

        return SchedulingDecision(
            prb_assignment=prb_assignment,
            ue_mcs=ue_mcs.copy(),
            ue_rank=ue_rank.copy(),
            ue_num_prbs=ue_num_prbs,
            ue_tbs_bits=ue_tbs,
            ue_num_re=ue_num_re,
        )

    def update_throughput_history(self, ue_throughput_bits: np.ndarray):
        """更新 PF 时间平均吞吐量"""
        self._t_avg = (
            self.beta * self._t_avg
            + (1.0 - self.beta) * ue_throughput_bits
        )
        # 防止退化到 0
        self._t_avg = np.maximum(self._t_avg, 1e-10)

"""下行功率控制

支持:
1. 等功率分配: 每 PRB 平均分配总功率
2. Water-filling: 基于信道质量的注水功率分配
3. 公平功率控制: 保证最低功率 + 剩余按信道质量分配
"""

import numpy as np
from ..utils.math_utils import dbm_to_watt, watt_to_dbm, db_to_linear


class DLPowerControl:
    """下行功率控制"""

    def __init__(self, total_power_dbm: float = 46.0,
                 num_prb: int = 273,
                 method: str = 'equal',
                 min_power_ratio: float = 0.5):
        """
        Args:
            total_power_dbm: 总发射功率 (dBm)
            num_prb: PRB 总数
            method: 功率分配方法 ('equal', 'waterfilling', 'fair')
            min_power_ratio: 公平功控中每 UE 保证的最低功率占比
        """
        self.total_power_w = dbm_to_watt(total_power_dbm)
        self.num_prb = num_prb
        self.method = method
        self.min_power_ratio = min_power_ratio

        # 等功率下每 PRB 的功率
        self._equal_power_per_prb = self.total_power_w / num_prb

    def allocate_power(self, prb_assignment: np.ndarray,
                       channel_quality: np.ndarray = None,
                       num_ue: int = 0) -> np.ndarray:
        """分配功率

        Args:
            prb_assignment: (num_prb,) PRB→UE 映射 (-1=未分配)
            channel_quality: (num_ue, num_prb) 信道质量 (SINR linear)
            num_ue: UE 数

        Returns:
            power_per_prb: (num_prb,) 每 PRB 的发射功率 (W)
        """
        if self.method == 'equal':
            return self._equal_power(prb_assignment)
        elif self.method == 'waterfilling':
            return self._waterfilling(prb_assignment, channel_quality, num_ue)
        elif self.method == 'fair':
            return self._fair_power(prb_assignment, channel_quality, num_ue)
        else:
            return self._equal_power(prb_assignment)

    def _equal_power(self, prb_assignment: np.ndarray) -> np.ndarray:
        """等功率分配"""
        power = np.zeros(self.num_prb)
        allocated = prb_assignment >= 0
        if np.any(allocated):
            num_allocated = np.sum(allocated)
            power[allocated] = self.total_power_w / num_allocated
        return power

    def _waterfilling(self, prb_assignment: np.ndarray,
                      channel_quality: np.ndarray,
                      num_ue: int) -> np.ndarray:
        """注水功率分配

        在分配的 PRB 上，按信道质量反比分配功率:
        差信道分配更多功率 (均衡 SINR)。
        """
        power = np.zeros(self.num_prb)
        allocated_mask = prb_assignment >= 0
        if not np.any(allocated_mask) or channel_quality is None:
            return self._equal_power(prb_assignment)

        allocated_prbs = np.where(allocated_mask)[0]
        num_allocated = len(allocated_prbs)

        # 每 PRB 的信道质量 (对应 UE)
        cq = np.zeros(num_allocated)
        for i, prb in enumerate(allocated_prbs):
            ue = prb_assignment[prb]
            cq[i] = max(channel_quality[ue, prb], 1e-10)

        # 注水: 功率正比于 1/cq (差信道给更多)
        inv_cq = 1.0 / cq
        total_inv = np.sum(inv_cq)
        power_allocated = self.total_power_w * inv_cq / total_inv

        for i, prb in enumerate(allocated_prbs):
            power[prb] = power_allocated[i]

        return power

    def _fair_power(self, prb_assignment: np.ndarray,
                    channel_quality: np.ndarray,
                    num_ue: int) -> np.ndarray:
        """公平功率控制

        1. 保证每 UE 最低功率 (min_power_ratio × P_total / num_ue)
        2. 剩余功率按信道质量比例分配
        """
        power = np.zeros(self.num_prb)
        allocated_mask = prb_assignment >= 0
        if not np.any(allocated_mask):
            return power

        # 每 UE 分配的 PRB
        ue_prbs = {}
        for prb in range(self.num_prb):
            ue = prb_assignment[prb]
            if ue >= 0:
                ue_prbs.setdefault(ue, []).append(prb)

        num_active_ue = len(ue_prbs)
        if num_active_ue == 0:
            return power

        # 保证最低功率
        min_power_per_ue = self.min_power_ratio * self.total_power_w / num_active_ue
        remaining_power = self.total_power_w * (1.0 - self.min_power_ratio)

        # 分配最低功率
        for ue, prbs in ue_prbs.items():
            power_per_prb = min_power_per_ue / len(prbs)
            for prb in prbs:
                power[prb] = power_per_prb

        # 剩余功率按信道质量分配
        if channel_quality is not None and remaining_power > 0:
            total_cq = 0
            ue_cq = {}
            for ue, prbs in ue_prbs.items():
                avg_cq = np.mean([max(channel_quality[ue, p], 1e-10) for p in prbs])
                ue_cq[ue] = avg_cq
                total_cq += avg_cq

            if total_cq > 0:
                for ue, prbs in ue_prbs.items():
                    extra_power = remaining_power * ue_cq[ue] / total_cq
                    extra_per_prb = extra_power / len(prbs)
                    for prb in prbs:
                        power[prb] += extra_per_prb

        return power

    def get_power_per_prb_equal(self) -> float:
        """获取等功率分配下每 PRB 的功率"""
        return self._equal_power_per_prb

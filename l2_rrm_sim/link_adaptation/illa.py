"""内环链路自适应 (Inner Loop Link Adaptation)

给定有效 SINR，选择满足 BLER ≤ target 的最高 MCS。
"""

import numpy as np
from .bler_tables import BLERTableManager
from .mcs_tables import get_max_mcs_index
from ..utils.nr_utils import compute_tbs, compute_num_code_blocks


class ILLA:
    """内环链路自适应

    遍历 MCS 从高到低，找到满足 BLER 目标的最高 MCS。
    """

    def __init__(self, bler_table: BLERTableManager,
                 bler_target: float = 0.1,
                 mcs_table_index: int = 1,
                 num_re_per_prb: int = 132):
        self.bler_table = bler_table
        self.bler_target = bler_target
        self.mcs_table_index = mcs_table_index
        self.num_re_per_prb = num_re_per_prb

    def select_mcs(self, sinr_eff_db: float,
                   num_allocated_prbs: int = 1,
                   num_layers: int = 1) -> int:
        """选择最优 MCS

        Args:
            sinr_eff_db: 有效 SINR (dB)
            num_allocated_prbs: 分配的 PRB 数 (用于计算 CBS)
            num_layers: 传输层数

        Returns:
            选择的 MCS index
        """
        max_mcs = get_max_mcs_index(self.mcs_table_index)

        # 从高到低遍历 MCS
        for mcs in range(max_mcs, -1, -1):
            # 计算 TBS 和 CBS
            tbs = compute_tbs(
                self.num_re_per_prb, num_allocated_prbs,
                mcs, num_layers, self.mcs_table_index
            )
            if tbs <= 0:
                continue

            from .mcs_tables import get_mcs_params
            _, rate_x1024 = get_mcs_params(mcs, self.mcs_table_index)
            r = rate_x1024 / 1024.0
            num_cb, cbs = compute_num_code_blocks(tbs, r)
            if cbs <= 0:
                continue

            # 查询 BLER
            bler = self.bler_table.lookup_bler(
                sinr_eff_db, mcs, cbs, self.mcs_table_index
            )

            # 计算 TBLER
            if num_cb > 1:
                tbler = 1.0 - (1.0 - bler) ** num_cb
            else:
                tbler = bler

            if tbler <= self.bler_target:
                return mcs

        # 没有 MCS 满足目标，返回最低 MCS
        return 0

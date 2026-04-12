"""Legacy PHY 适配器

将自实现的 EESM + ILLA + OLLA + PHYAbstraction 包装为 PHYBase 统一接口，
使引擎无需区分 Sionna 和 Legacy 两条路径。
"""

import numpy as np
from .phy_interface import PHYBase
from .bler_tables import BLERTableManager
from .effective_sinr import EESM
from .illa import ILLA
from .olla import OLLA
from .phy_abstraction import PHYAbstraction


class LegacyPHYAdapter(PHYBase):
    """Legacy PHY 适配器 — 实现 PHYBase 统一接口"""

    def __init__(self, num_ue: int,
                 bler_target: float = 0.1,
                 delta_up: float = 0.5,
                 offset_min: float = -10.0,
                 offset_max: float = 10.0,
                 mcs_table_index: int = 1,
                 num_re_per_prb: int = 132,
                 rng=None):
        self.num_ue = num_ue
        self.mcs_table_index = mcs_table_index

        self.bler_table = BLERTableManager()
        self.eesm = EESM()
        self.illa = ILLA(self.bler_table, bler_target,
                         mcs_table_index, num_re_per_prb)
        self.olla = OLLA(num_ue, self.illa,
                         bler_target=bler_target,
                         mcs_table_index=mcs_table_index)
        self.phy_abs = PHYAbstraction(self.bler_table, self.eesm,
                                       num_re_per_prb, mcs_table_index,
                                       rng)

        self._last_sinr_eff = np.ones(num_ue, dtype=np.float32)
        self._last_scheduled = np.zeros(num_ue, dtype=bool)

    @property
    def num_re_per_prb(self):
        return self.illa.num_re_per_prb

    @num_re_per_prb.setter
    def num_re_per_prb(self, value):
        self.illa.num_re_per_prb = value

    def select_mcs(self, num_allocated_re: np.ndarray,
                   harq_feedback: np.ndarray,
                   sinr_eff: np.ndarray,
                   scheduled_mask: np.ndarray,
                   ue_rank: np.ndarray = None,
                   sinr_eff_db: np.ndarray = None,
                   **kwargs) -> np.ndarray:
        """通过 OLLA 选择 MCS (统一接口)

        支持两种输入模式:
        - sinr_eff (linear): 由外部计算的有效 SINR
        - sinr_eff_db: 直接传入 dB 值 (Legacy EESM 输出)
        """
        is_ack = (harq_feedback == 1)
        sched_bool = scheduled_mask.astype(bool) if scheduled_mask is not None else np.ones(self.num_ue, dtype=bool)
        self.olla.update_offsets_batch(is_ack, sched_bool)

        if sinr_eff_db is None:
            from ..utils.math_utils import linear_to_db
            sinr_eff_db = np.where(sinr_eff > 0, linear_to_db(sinr_eff), -30.0)

        est_prbs = np.maximum(num_allocated_re // max(self.illa.num_re_per_prb, 1), 1).astype(np.int32)
        if ue_rank is None:
            ue_rank = np.ones(self.num_ue, dtype=np.int32)
        return self.olla.select_mcs(sinr_eff_db, est_prbs, ue_rank)

    def compute_sinr_eff(self, sinr_per_prb: np.ndarray,
                         mcs_indices: np.ndarray,
                         prb_assignment: np.ndarray = None) -> np.ndarray:
        """计算 per-UE 有效 SINR (EESM)"""
        from ..utils.math_utils import db_to_linear
        num_ue = self.num_ue
        sinr_eff_linear = np.zeros(num_ue, dtype=np.float32)

        for ue in range(num_ue):
            if prb_assignment is not None:
                ue_prbs = (prb_assignment == ue)
                if not np.any(ue_prbs):
                    continue
                # 使用所有活跃层
                n_layers = sinr_per_prb.shape[1]
                sinr_ue = sinr_per_prb[ue, :, :][:, ue_prbs]
                # 去掉全零层
                active = np.any(sinr_ue > 0, axis=1)
                if not np.any(active):
                    continue
                sinr_ue = sinr_ue[active, :]
            else:
                sinr_ue = sinr_per_prb[ue]
                active = np.any(sinr_ue > 0, axis=1)
                if not np.any(active):
                    continue
                sinr_ue = sinr_ue[active, :]

            sinr_eff_db = self.eesm.compute(
                sinr_ue, int(mcs_indices[ue]), self.mcs_table_index
            )
            sinr_eff_linear[ue] = db_to_linear(sinr_eff_db)

        return sinr_eff_linear

    def evaluate(self, mcs_indices: np.ndarray,
                 sinr_per_prb: np.ndarray = None,
                 num_allocated_re: np.ndarray = None,
                 prb_assignment: np.ndarray = None,
                 ue_rank: np.ndarray = None,
                 re_per_prb: int = None,
                 sinr_eff_override: np.ndarray = None,
                 sinr_per_re: np.ndarray = None) -> dict:
        # 兼容 SionnaPHY 的 sinr_per_re 参数名
        if sinr_per_prb is None and sinr_per_re is not None:
            sinr_per_prb = sinr_per_re
        """评估 PHY 传输结果 (统一接口)"""
        num_ue = len(mcs_indices)
        if ue_rank is None:
            ue_rank = np.ones(num_ue, dtype=np.int32)

        num_prbs = np.zeros(num_ue, dtype=np.int32)
        if prb_assignment is not None:
            for ue in range(num_ue):
                num_prbs[ue] = int(np.sum(prb_assignment == ue))

        results = self.phy_abs.evaluate_batch(
            sinr_per_prb, mcs_indices, num_prbs, ue_rank,
            prb_assignment, re_per_prb=re_per_prb,
            sinr_eff_override_linear=sinr_eff_override,
        )

        # 转换为统一格式 (与 SionnaPHY 兼容)
        from ..utils.math_utils import db_to_linear
        results['sinr_eff'] = db_to_linear(results['sinr_eff_db']).astype(np.float32)
        results['sinr_eff_raw'] = db_to_linear(results['sinr_eff_db_raw']).astype(np.float32)
        results['harq_feedback'] = np.where(results['is_success'], 1, 0).astype(np.int32)

        self._last_sinr_eff = results['sinr_eff_raw'].copy()
        self._last_scheduled = (num_prbs > 0)

        return results

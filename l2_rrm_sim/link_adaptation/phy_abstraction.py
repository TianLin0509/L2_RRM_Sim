"""PHY 抽象层

完整管道: per-PRB SINR → EESM有效SINR → MCS → TBS → CBS → BLER查表 → TBLER → ACK/NACK
"""

import numpy as np
from .effective_sinr import EESM
from .bler_tables import BLERTableManager
from .mcs_tables import get_mcs_params
from ..utils.nr_utils import compute_tbs, compute_num_code_blocks


class PHYAbstraction:
    """PHY 层抽象

    给定 SINR、MCS、分配的 RE 数，计算 BLER 和解码结果。
    """

    def __init__(self, bler_table: BLERTableManager, eesm: EESM,
                 num_re_per_prb: int = 132,
                 mcs_table_index: int = 1,
                 rng: np.random.Generator = None):
        self.bler_table = bler_table
        self.eesm = eesm
        self.num_re_per_prb = num_re_per_prb
        self.mcs_table_index = mcs_table_index
        self._rng = rng if rng is not None else np.random.default_rng()

    def evaluate(self, sinr_per_prb: np.ndarray,
                 mcs_index: int, num_prbs: int,
                 num_layers: int) -> dict:
        """评估单个 UE 的 PHY 传输结果

        Args:
            sinr_per_prb: (num_layers, num_prbs_allocated) SINR [linear]
            mcs_index: MCS 索引
            num_prbs: 分配的 PRB 数
            num_layers: 传输层数

        Returns:
            dict with keys:
                sinr_eff_db, tbs, bler, tbler, is_success, decoded_bits
        """
        if num_prbs <= 0:
            return {
                'sinr_eff_db': -30.0, 'tbs': 0, 'bler': 1.0,
                'tbler': 1.0, 'is_success': False, 'decoded_bits': 0
            }

        # 1. 计算有效 SINR
        sinr_eff_db = self.eesm.compute(
            sinr_per_prb, mcs_index, self.mcs_table_index
        )

        # 2. 计算 TBS
        tbs = compute_tbs(
            self.num_re_per_prb, num_prbs,
            mcs_index, num_layers, self.mcs_table_index
        )
        if tbs <= 0:
            return {
                'sinr_eff_db': sinr_eff_db, 'tbs': 0, 'bler': 1.0,
                'tbler': 1.0, 'is_success': False, 'decoded_bits': 0
            }

        # 3. 计算码块数和码块大小
        _, rate_x1024 = get_mcs_params(mcs_index, self.mcs_table_index)
        r = rate_x1024 / 1024.0
        num_cb, cbs = compute_num_code_blocks(tbs, r)

        # 4. 查询 BLER
        bler = self.bler_table.lookup_bler(
            sinr_eff_db, mcs_index, cbs, self.mcs_table_index
        )

        # 5. 计算 TBLER (Transport Block Error Rate)
        if num_cb > 1:
            tbler = 1.0 - (1.0 - bler) ** num_cb
        else:
            tbler = bler

        # 6. 随机判定 TB 是否成功
        is_success = self._rng.random() >= tbler

        decoded_bits = tbs if is_success else 0

        return {
            'sinr_eff_db': sinr_eff_db,
            'tbs': tbs,
            'bler': bler,
            'tbler': tbler,
            'is_success': is_success,
            'decoded_bits': decoded_bits
        }

    def evaluate_batch(self, sinr_per_prb_all: np.ndarray,
                       mcs_indices: np.ndarray,
                       num_prbs_all: np.ndarray,
                       num_layers_all: np.ndarray,
                       prb_assignment: np.ndarray) -> dict:
        """批量评估所有调度 UE

        Args:
            sinr_per_prb_all: (num_ue, max_layers, total_num_prb) SINR [linear]
            mcs_indices: (num_ue,) MCS indices
            num_prbs_all: (num_ue,) 每 UE 分配的 PRB 数
            num_layers_all: (num_ue,) 每 UE 层数
            prb_assignment: (total_num_prb,) PRB 分配

        Returns:
            dict with arrays: sinr_eff_db, tbs, bler, tbler, is_success, decoded_bits
        """
        num_ue = len(mcs_indices)
        results = {
            'sinr_eff_db': np.full(num_ue, -30.0),
            'tbs': np.zeros(num_ue, dtype=np.int64),
            'bler': np.ones(num_ue),
            'tbler': np.ones(num_ue),
            'is_success': np.zeros(num_ue, dtype=bool),
            'decoded_bits': np.zeros(num_ue, dtype=np.int64),
        }

        for ue in range(num_ue):
            n_prbs = int(num_prbs_all[ue])
            if n_prbs <= 0:
                continue

            n_layers = int(num_layers_all[ue])
            mcs = int(mcs_indices[ue])

            # 提取该 UE 被分配的 PRB 对应的 SINR
            ue_prb_mask = (prb_assignment == ue)
            sinr_ue = sinr_per_prb_all[ue, :n_layers, :][:, ue_prb_mask]

            result = self.evaluate(sinr_ue, mcs, n_prbs, n_layers)
            for key in results:
                results[key][ue] = result[key]

        return results

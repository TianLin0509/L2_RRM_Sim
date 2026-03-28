"""EESM 有效 SINR 计算

Exponential Effective SINR Mapping (EESM):
    SINR_eff = -β × ln( (1/N) × Σ exp(-SINR_i / β) )

β 参数按 MCS 索引从 eesm_beta_table.json 加载。
"""

import json
import numpy as np
from pathlib import Path


class EESM:
    """EESM 有效 SINR 映射器

    将 per-PRB per-layer 的 SINR 向量聚合为单个有效 SINR 值。
    """

    def __init__(self, beta_table_path: str = None):
        if beta_table_path is None:
            beta_table_path = str(
                Path(__file__).parent / "esm_params" / "eesm_beta_table.json"
            )
        self._beta_table = {}
        self._load_beta_table(beta_table_path)

    def _load_beta_table(self, path: str):
        """加载 EESM β 参数表"""
        with open(path, 'r') as f:
            data = json.load(f)

        index_data = data.get("index", data)
        for tbl_key, betas in index_data.items():
            self._beta_table[int(tbl_key)] = np.array(betas, dtype=np.float64)

    def get_beta(self, mcs_index: int, table_index: int = 1) -> float:
        """获取 β 参数"""
        if table_index not in self._beta_table:
            return 1.0
        betas = self._beta_table[table_index]
        if mcs_index < 0 or mcs_index >= len(betas):
            return betas[-1]
        return float(betas[mcs_index])

    def compute(self, sinr_linear: np.ndarray, mcs_index: int,
                table_index: int = 1,
                sinr_min_db: float = -30.0,
                sinr_max_db: float = 40.0) -> float:
        """计算有效 SINR

        Args:
            sinr_linear: SINR 向量 (线性值)，shape 任意，会被 flatten
                         只有 > 0 的值参与计算
            mcs_index: MCS 索引 (用于选择 β)
            table_index: MCS 表索引

        Returns:
            有效 SINR (dB)
        """
        sinr = np.asarray(sinr_linear, dtype=np.float64).ravel()
        # 只用有效的 SINR 值
        valid = sinr > 0
        if not np.any(valid):
            return sinr_min_db

        sinr_valid = sinr[valid]
        beta = self.get_beta(mcs_index, table_index)

        # EESM: SINR_eff = -β × ln( mean( exp(-SINR_i / β) ) )
        exponents = -sinr_valid / beta
        # 数值稳定: 使用 logsumexp 技巧
        max_exp = np.max(exponents)
        log_mean = max_exp + np.log(np.mean(np.exp(exponents - max_exp)))
        sinr_eff_linear = -beta * log_mean

        # 转 dB 并裁剪
        sinr_eff_db = 10.0 * np.log10(max(sinr_eff_linear, 1e-30))
        return float(np.clip(sinr_eff_db, sinr_min_db, sinr_max_db))

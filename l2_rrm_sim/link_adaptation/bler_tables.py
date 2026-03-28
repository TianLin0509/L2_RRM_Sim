"""BLER 查找表加载与插值

从 Sionna 导出的 BLER 表 (JSON) 中加载数据，
构建 scipy 插值器用于 SINR → BLER 查询。
"""

import json
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path


class BLERTableManager:
    """BLER 查找表管理器

    加载 PDSCH BLER 表并提供插值查询。
    表结构: category -> index -> MCS -> CBS -> {BLER, SNR_db}
    """

    def __init__(self, table_dir: str = None):
        if table_dir is None:
            table_dir = str(Path(__file__).parent / "bler_tables")
        self._table_dir = table_dir
        # {(table_index, mcs_index): {cbs: interp1d_func}}
        self._interpolators = {}
        # {(table_index, mcs_index): [available_cbs_values]}
        self._cbs_values = {}
        self._load_tables()

    def _load_tables(self):
        """加载所有 PDSCH BLER 表"""
        table_dir = Path(self._table_dir)
        for table_idx in range(1, 5):
            fpath = table_dir / f"PDSCH_table{table_idx}.json"
            if not fpath.exists():
                continue
            with open(fpath, 'r') as f:
                data = json.load(f)

            # 导航 JSON 结构
            category_data = data.get("category", data)
            # PDSCH category = "1"
            cat_key = "1" if "1" in category_data else list(category_data.keys())[0]
            index_data = category_data[cat_key].get("index", category_data[cat_key])

            tbl_key = str(table_idx)
            if tbl_key not in index_data:
                continue
            mcs_data = index_data[tbl_key].get("MCS", index_data[tbl_key])

            for mcs_str, mcs_entry in mcs_data.items():
                mcs_idx = int(mcs_str)
                cbs_data = mcs_entry.get("CBS", mcs_entry)
                snr_db = None
                cbs_interps = {}
                cbs_list = []

                for cbs_str, cbs_entry in cbs_data.items():
                    if cbs_str in ("SNR_db",):
                        continue
                    cbs_val = int(cbs_str)
                    bler_vals = np.array(cbs_entry["BLER"], dtype=np.float64)

                    # SNR 可能在 CBS 条目内或在 MCS 条目内
                    if "SNR_db" in cbs_entry:
                        snr_vals = np.array(cbs_entry["SNR_db"], dtype=np.float64)
                    elif snr_db is None and "SNR_db" in mcs_entry:
                        snr_db = np.array(mcs_entry["SNR_db"], dtype=np.float64)
                        snr_vals = snr_db
                    else:
                        snr_vals = snr_db

                    if snr_vals is None or len(snr_vals) != len(bler_vals):
                        continue

                    # 构建插值器 (线性插值, 边界外截断到 [0, 1])
                    cbs_interps[cbs_val] = interp1d(
                        snr_vals, bler_vals,
                        kind='linear', bounds_error=False,
                        fill_value=(bler_vals[0], bler_vals[-1])
                    )
                    cbs_list.append(cbs_val)

                if cbs_list:
                    self._interpolators[(table_idx, mcs_idx)] = cbs_interps
                    self._cbs_values[(table_idx, mcs_idx)] = sorted(cbs_list)

    def lookup_bler(self, sinr_eff_db: float, mcs_index: int,
                    cbs: int, table_index: int = 1) -> float:
        """查询 BLER

        Args:
            sinr_eff_db: 有效 SINR (dB)
            mcs_index: MCS 索引
            cbs: 码块大小 (bits)
            table_index: MCS 表索引

        Returns:
            BLER 值 (0~1)
        """
        key = (table_index, mcs_index)
        if key not in self._interpolators:
            # 没有此 MCS 的 BLER 数据
            # 如果请求的 MCS 低于表中最低 MCS，则用最低 MCS 的 BLER
            # (更低的 MCS 更鲁棒，BLER 应更低)
            available_mcs = [k[1] for k in self._interpolators if k[0] == table_index]
            if available_mcs:
                min_available = min(available_mcs)
                if mcs_index < min_available:
                    return self.lookup_bler(sinr_eff_db, min_available, cbs, table_index)
            return 1.0

        cbs_interps = self._interpolators[key]
        cbs_list = self._cbs_values[key]

        # 找最接近的 CBS
        if cbs in cbs_interps:
            return float(np.clip(cbs_interps[cbs](sinr_eff_db), 0.0, 1.0))

        # 插值到最近的 CBS
        idx = np.searchsorted(cbs_list, cbs)
        if idx == 0:
            nearest_cbs = cbs_list[0]
        elif idx >= len(cbs_list):
            nearest_cbs = cbs_list[-1]
        else:
            # 选更近的
            if abs(cbs_list[idx] - cbs) < abs(cbs_list[idx - 1] - cbs):
                nearest_cbs = cbs_list[idx]
            else:
                nearest_cbs = cbs_list[idx - 1]

        return float(np.clip(cbs_interps[nearest_cbs](sinr_eff_db), 0.0, 1.0))

    def has_entry(self, table_index: int, mcs_index: int) -> bool:
        return (table_index, mcs_index) in self._interpolators

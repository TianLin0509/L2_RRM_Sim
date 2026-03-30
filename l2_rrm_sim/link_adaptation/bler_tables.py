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

    _GRID_SNR_MIN = -15.0
    _GRID_SNR_MAX = 35.0
    _GRID_SNR_STEP = 0.1

    def __init__(self, table_dir: str = None):
        if table_dir is None:
            table_dir = str(Path(__file__).parent / "bler_tables")
        self._table_dir = table_dir
        # {(table_index, mcs_index): {cbs: interp1d_func}}
        self._interpolators = {}
        # {(table_index, mcs_index): [available_cbs_values]}
        self._cbs_values = {}
        # Precomputed BLER grid for fast np.interp lookup
        # {(table_index, mcs_index): {cbs: bler_array}}
        self._bler_grid = {}
        self._grid_snr = None  # shared SNR axis
        # CBS nearest-neighbor cache
        self._cbs_cache = {}
        self._load_tables()
        self._build_lookup_grid()

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

    def _build_lookup_grid(self):
        """Precompute BLER on a dense SNR grid for fast np.interp lookup."""
        self._grid_snr = np.arange(
            self._GRID_SNR_MIN,
            self._GRID_SNR_MAX + self._GRID_SNR_STEP * 0.5,
            self._GRID_SNR_STEP,
        )
        for key, cbs_interps in self._interpolators.items():
            grid_dict = {}
            for cbs_val, interp_func in cbs_interps.items():
                bler_arr = interp_func(self._grid_snr).astype(np.float64)
                np.clip(bler_arr, 0.0, 1.0, out=bler_arr)
                grid_dict[cbs_val] = bler_arr
            self._bler_grid[key] = grid_dict

    def _find_nearest_cbs(self, key, cbs: int) -> int:
        """Find the nearest CBS value, with caching."""
        cache_key = (key, cbs)
        if cache_key in self._cbs_cache:
            return self._cbs_cache[cache_key]
        cbs_list = self._cbs_values[key]
        idx = np.searchsorted(cbs_list, cbs)
        if idx == 0:
            nearest = cbs_list[0]
        elif idx >= len(cbs_list):
            nearest = cbs_list[-1]
        else:
            if abs(cbs_list[idx] - cbs) < abs(cbs_list[idx - 1] - cbs):
                nearest = cbs_list[idx]
            else:
                nearest = cbs_list[idx - 1]
        self._cbs_cache[cache_key] = nearest
        return nearest

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
        grid_dict = self._bler_grid.get(key)
        if grid_dict is None:
            # 没有此 MCS 的 BLER 数据
            available_mcs = [k[1] for k in self._bler_grid if k[0] == table_index]
            if available_mcs:
                min_available = min(available_mcs)
                if mcs_index < min_available:
                    return self.lookup_bler(sinr_eff_db, min_available, cbs, table_index)
            return 1.0

        # Find nearest CBS
        if cbs in grid_dict:
            bler_arr = grid_dict[cbs]
        else:
            nearest_cbs = self._find_nearest_cbs(key, cbs)
            bler_arr = grid_dict[nearest_cbs]

        # Fast np.interp on precomputed grid
        val = float(np.interp(sinr_eff_db, self._grid_snr, bler_arr))
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    def has_entry(self, table_index: int, mcs_index: int) -> bool:
        return (table_index, mcs_index) in self._interpolators

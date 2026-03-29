"""5G NR 资源网格管理

基于 TS 38.211/38.214 的资源网格和开销计算。
支持 PRB 级和 RBG (Resource Block Group) 级资源管理。

RBG 大小: TS 38.214 Table 5.1.2.2.1-1
权值计算粒度: 4 PRB (子带 CQI/SINR 分组)
"""

import numpy as np
from ..config.sim_config import CarrierConfig
from ..utils.nr_utils import compute_num_re_per_prb, compute_tbs

# TS 38.214 Table 5.1.2.2.1-1: BWP size → RBG size (P)
# {(min_prb, max_prb): (config1_P, config2_P)}
_RBG_SIZE_TABLE = [
    (1,   36,  2, 4),
    (37,  72,  4, 8),
    (73,  144, 8, 16),
    (145, 275, 16, 16),
]

# 权值计算粒度 (PRB)
WEIGHT_GRANULARITY_PRB = 4


def get_rbg_size(num_prb: int, config: int = 1) -> int:
    """查表获取 RBG 大小 (TS 38.214 Table 5.1.2.2.1-1)

    Args:
        num_prb: BWP 内 PRB 数
        config: RBG 配置 (1 或 2)

    Returns:
        RBG 大小 P (PRB 数)
    """
    for min_prb, max_prb, p1, p2 in _RBG_SIZE_TABLE:
        if min_prb <= num_prb <= max_prb:
            return p1 if config == 1 else p2
    return 16  # 默认


class ResourceGrid:
    """5G NR 资源网格

    管理 PRB/RBG/RE 级资源。
    调度以 RBG 为粒度，权值以 4RB 为粒度。
    """

    def __init__(self, carrier_config: CarrierConfig, rbg_config: int = 1):
        self.scs_khz = carrier_config.subcarrier_spacing
        self.num_prb = carrier_config.num_prb
        self.num_pdcch_symbols = carrier_config.num_pdcch_symbols
        self.dmrs_type = carrier_config.dmrs_type
        self.dmrs_cdm_groups = carrier_config.dmrs_cdm_groups
        self.num_dmrs_symbols = carrier_config.num_dmrs_symbols

        # PRB 级 RE 数
        self._num_re_per_prb = compute_num_re_per_prb(
            num_pdcch_symbols=self.num_pdcch_symbols,
            dmrs_type=self.dmrs_type,
            num_dmrs_cdm_groups=self.dmrs_cdm_groups,
            num_dmrs_symbols=self.num_dmrs_symbols
        )

        # RBG 配置 (TS 38.214)
        self.rbg_size = get_rbg_size(self.num_prb, rbg_config)
        self.num_rbg = (self.num_prb + self.rbg_size - 1) // self.rbg_size

        # RBG → PRB 映射
        self._rbg_prb_ranges = []   # [(start_prb, end_prb), ...]
        self._rbg_prb_counts = np.zeros(self.num_rbg, dtype=np.int32)
        for rbg in range(self.num_rbg):
            start = rbg * self.rbg_size
            end = min(start + self.rbg_size, self.num_prb)
            self._rbg_prb_ranges.append((start, end))
            self._rbg_prb_counts[rbg] = end - start

        # 4RB 权值分组
        self.weight_granularity = WEIGHT_GRANULARITY_PRB
        self.num_weight_groups = (self.num_prb + self.weight_granularity - 1) // self.weight_granularity

    @property
    def num_re_per_prb(self) -> int:
        return self._num_re_per_prb

    def get_tbs(self, mcs_index: int, num_prbs: int, num_layers: int,
                mcs_table_index: int = 1) -> int:
        """计算 TBS"""
        return compute_tbs(
            self._num_re_per_prb, num_prbs,
            mcs_index, num_layers, mcs_table_index
        )

    def rbg_prb_range(self, rbg_idx: int) -> tuple:
        """获取 RBG 对应的 PRB 范围 [start, end)"""
        return self._rbg_prb_ranges[rbg_idx]

    def aggregate_prb_to_rbg(self, per_prb: np.ndarray) -> np.ndarray:
        """将 per-PRB 指标聚合到 per-RBG (均值)

        Args:
            per_prb: (..., num_prb) 最后一维是 PRB

        Returns:
            (..., num_rbg)
        """
        shape = per_prb.shape[:-1]
        result = np.zeros(shape + (self.num_rbg,), dtype=per_prb.dtype)
        for rbg in range(self.num_rbg):
            s, e = self._rbg_prb_ranges[rbg]
            result[..., rbg] = np.mean(per_prb[..., s:e], axis=-1)
        return result

    def expand_rbg_to_prb(self, rbg_assignment: np.ndarray) -> np.ndarray:
        """将 RBG 级分配展开为 PRB 级

        Args:
            rbg_assignment: (num_rbg,) RBG→UE 分配 (-1=未分配)

        Returns:
            prb_assignment: (num_prb,) PRB→UE 分配
        """
        prb_assignment = np.full(self.num_prb, -1, dtype=np.int32)
        for rbg in range(self.num_rbg):
            s, e = self._rbg_prb_ranges[rbg]
            prb_assignment[s:e] = rbg_assignment[rbg]
        return prb_assignment

    def aggregate_prb_to_weight_groups(self, per_prb: np.ndarray) -> np.ndarray:
        """将 per-PRB 指标聚合到 4RB 权值分组 (均值)

        用于子带 CQI/SINR/PMI 计算。

        Args:
            per_prb: (..., num_prb)

        Returns:
            (..., num_weight_groups)
        """
        g = self.weight_granularity
        shape = per_prb.shape[:-1]
        result = np.zeros(shape + (self.num_weight_groups,), dtype=per_prb.dtype)
        for i in range(self.num_weight_groups):
            s = i * g
            e = min(s + g, self.num_prb)
            result[..., i] = np.mean(per_prb[..., s:e], axis=-1)
        return result

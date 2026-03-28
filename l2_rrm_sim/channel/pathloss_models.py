"""3GPP TR 38.901 路径损耗模型

实现 UMa, UMi, RMa 场景的路径损耗计算。
公式来源: 3GPP TR 38.901 Table 7.4.1-1
"""

import numpy as np
from ..core.nr_constants import SPEED_OF_LIGHT


def _breakpoint_distance(h_bs: float, h_ut: float, fc_hz: float,
                         h_e: float = 1.0) -> float:
    """计算断点距离 (TS 38.901 Table 7.4.1-1 Note 1)

    d_BP' = 4 × h_BS' × h_UT' × fc / c
    """
    h_bs_prime = h_bs - h_e
    h_ut_prime = h_ut - h_e
    return 4.0 * h_bs_prime * h_ut_prime * fc_hz / SPEED_OF_LIGHT


def compute_pathloss_uma(d_2d: float, h_bs: float, h_ut: float,
                         fc_ghz: float, is_los: bool) -> float:
    """UMa 路径损耗 (TS 38.901 Table 7.4.1-1)

    Args:
        d_2d: 2D 距离 (m), 最小 10m
        h_bs: 基站高度 (m), 通常 25m
        h_ut: UE 高度 (m), 通常 1.5-22.5m
        fc_ghz: 载频 (GHz), 0.5-100
        is_los: 是否 LOS

    Returns:
        路径损耗 (dB), 正值
    """
    d_2d = max(d_2d, 10.0)
    d_3d = np.sqrt(d_2d**2 + (h_bs - h_ut)**2)
    fc_hz = fc_ghz * 1e9

    if is_los:
        # LOS
        d_bp = _breakpoint_distance(h_bs, h_ut, fc_hz, h_e=1.0)

        # PL1
        pl1 = 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz)

        if d_2d <= d_bp:
            return pl1
        else:
            # PL2
            pl2 = (28.0 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz)
                   - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
            return pl2
    else:
        # NLOS
        pl_los = compute_pathloss_uma(d_2d, h_bs, h_ut, fc_ghz, is_los=True)
        pl3 = (13.54 + 39.08 * np.log10(d_3d)
               + 20.0 * np.log10(fc_ghz)
               - 0.6 * (h_ut - 1.5))
        return max(pl_los, pl3)


def compute_pathloss_umi(d_2d: float, h_bs: float, h_ut: float,
                         fc_ghz: float, is_los: bool) -> float:
    """UMi Street Canyon 路径损耗 (TS 38.901 Table 7.4.1-1)"""
    d_2d = max(d_2d, 10.0)
    d_3d = np.sqrt(d_2d**2 + (h_bs - h_ut)**2)
    fc_hz = fc_ghz * 1e9

    if is_los:
        d_bp = _breakpoint_distance(h_bs, h_ut, fc_hz, h_e=1.0)
        pl1 = 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz)
        if d_2d <= d_bp:
            return pl1
        else:
            pl2 = (32.4 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz)
                   - 9.5 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
            return pl2
    else:
        pl_los = compute_pathloss_umi(d_2d, h_bs, h_ut, fc_ghz, is_los=True)
        pl_nlos = (22.4 + 35.3 * np.log10(d_3d)
                   + 21.3 * np.log10(fc_ghz)
                   - 0.3 * (h_ut - 1.5))
        return max(pl_los, pl_nlos)


def compute_pathloss_rma(d_2d: float, h_bs: float, h_ut: float,
                         fc_ghz: float, is_los: bool) -> float:
    """RMa 路径损耗 (TS 38.901 Table 7.4.1-1)"""
    d_2d = max(d_2d, 10.0)
    d_3d = np.sqrt(d_2d**2 + (h_bs - h_ut)**2)
    h = 5.0  # 平均建筑高度

    if is_los:
        fc_hz = fc_ghz * 1e9
        d_bp = 2.0 * np.pi * h_bs * h_ut * fc_hz / SPEED_OF_LIGHT
        pl1 = (20.0 * np.log10(40.0 * np.pi * d_3d * fc_ghz / 3.0)
               + min(0.03 * h**1.72, 10.0) * np.log10(d_3d)
               - min(0.044 * h**1.72, 14.77)
               + 0.002 * np.log10(h) * d_3d)
        if d_2d <= d_bp:
            return pl1
        else:
            pl2 = (pl1 + 40.0 * np.log10(d_3d / d_bp))
            return pl2
    else:
        pl_los = compute_pathloss_rma(d_2d, h_bs, h_ut, fc_ghz, is_los=True)
        pl_nlos = (161.04 - 7.1 * np.log10(h)
                   + 7.5 * np.log10(h)
                   - (24.37 - 3.7 * (h / h_bs)**2) * np.log10(h_bs)
                   + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d_3d) - 3.0)
                   + 20.0 * np.log10(fc_ghz)
                   - (3.2 * (np.log10(11.75 * h_ut))**2 - 4.97))
        return max(pl_los, pl_nlos)


def compute_los_probability_uma(d_2d: float, h_ut: float = 1.5) -> float:
    """UMa LOS 概率 (TS 38.901 Table 7.4.2-1)"""
    if d_2d <= 18.0:
        return 1.0
    c_prime = 0.0 if h_ut <= 13.0 else ((h_ut - 13.0) / 10.0) ** 1.5
    p_los = ((18.0 / d_2d + np.exp(-d_2d / 63.0) * (1.0 - 18.0 / d_2d))
             * (1.0 + c_prime * 5.0 / 4.0 * (d_2d / 100.0)**3
                * np.exp(-d_2d / 150.0)))
    return min(p_los, 1.0)


def compute_los_probability_umi(d_2d: float) -> float:
    """UMi LOS 概率 (TS 38.901 Table 7.4.2-1)"""
    if d_2d <= 18.0:
        return 1.0
    return 18.0 / d_2d + np.exp(-d_2d / 36.0) * (1.0 - 18.0 / d_2d)


# 路径损耗函数映射
PATHLOSS_MODELS = {
    'uma': compute_pathloss_uma,
    'umi': compute_pathloss_umi,
    'rma': compute_pathloss_rma,
}

LOS_PROBABILITY_MODELS = {
    'uma': compute_los_probability_uma,
    'umi': compute_los_probability_umi,
}

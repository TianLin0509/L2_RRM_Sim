"""3GPP 天线模型 (TR 38.901 Table 7.3-1)

Phase 1: 简化为全向天线
Phase 2: 3GPP 2D 天线阵列
"""

import numpy as np


def antenna_gain_isotropic(theta_deg: float, phi_deg: float) -> float:
    """全向天线增益 (0 dBi)"""
    return 0.0


def antenna_gain_3gpp_element(theta_deg: float, phi_deg: float,
                               theta_3db: float = 65.0,
                               phi_3db: float = 65.0,
                               am: float = 30.0,
                               sla_v: float = 30.0,
                               ge_max: float = 8.0) -> float:
    """3GPP TR 38.901 Table 7.3-1 天线单元增益

    Args:
        theta_deg: 垂直角 (度, 0=天顶)
        phi_deg: 水平角 (度, 0=正前方)
        theta_3db: 垂直 3dB 波束宽度
        phi_3db: 水平 3dB 波束宽度
        am: 前后比 (dB)
        sla_v: 侧瓣电平限制 (dB)
        ge_max: 最大增益 (dBi)

    Returns:
        天线增益 (dBi)
    """
    # 垂直方向图
    a_ev = -min(12.0 * ((theta_deg - 90.0) / theta_3db) ** 2, sla_v)
    # 水平方向图
    a_eh = -min(12.0 * (phi_deg / phi_3db) ** 2, am)
    # 合成
    a_e = -min(-(a_ev + a_eh), am)
    return ge_max + a_e

"""CQI 映射工具 (TS 38.214 Table 5.2.2.1-1)"""

import numpy as np

# TS 38.214 Table 5.2.2.1-1: 4-bit CQI Table 1
# CQI index -> (Modulation, Code Rate x 1024, Efficiency)
CQI_TABLE1 = {
    0:  (None, 0, 0),
    1:  (2, 78, 0.1523),
    2:  (2, 120, 0.2344),
    3:  (2, 193, 0.3770),
    4:  (2, 308, 0.6016),
    5:  (2, 449, 0.8770),
    6:  (2, 602, 1.1758),
    7:  (4, 378, 1.4766),
    8:  (4, 490, 1.9141),
    9:  (4, 616, 2.4063),
    10: (6, 466, 2.7305),
    11: (6, 567, 3.3223),
    12: (6, 666, 3.9023),
    13: (6, 772, 4.5234),
    14: (6, 873, 5.1152),
    15: (6, 948, 5.5547),
}

# 对应的 SINR 门限 (dB), 典型经验值 (针对 10% BLER)
# 这些值通常随信道模型和接收机实现不同而不同
SINR_TO_CQI_THRESH_DB = np.array([
    -10.0, -6.9, -5.1, -3.1, -1.4, 0.3, 2.1, 4.0, 6.1, 8.4, 10.3, 12.3, 14.2, 15.9, 17.8, 19.8
])

def sinr_to_cqi(sinr_db: float) -> int:
    """将 SINR [dB] 映射为 CQI 索引 (0-15)"""
    idx = np.searchsorted(SINR_TO_CQI_THRESH_DB, sinr_db)
    return int(np.clip(idx, 0, 15))

def cqi_to_mcs(cqi: int, olla_offset_db: float = 0.0) -> int:
    """CQI 到 MCS 的粗略映射 (基于 SE 匹配)

    在 SLS 中，通常直接用 SINR + OLLA 确定 MCS。
    如果要严格模拟反馈，则是 SINR -> CQI -> gNB 侧 MCS。
    """
    # 简化实现: CQI 直接映射为 MCS 表 1 中的索引
    # 实际上 CQI 1-15 并不完全对应 MCS 0-28
    # 这里的简单策略是让 MCS = CQI * 1.8 左右，或查表
    if cqi == 0: return 0
    
    # 经验映射 (Table 1: 0-28)
    # CQI 1 -> MCS 0, CQI 15 -> MCS 28
    mcs = int(np.round(np.interp(cqi, [1, 15], [0, 28])))
    return mcs

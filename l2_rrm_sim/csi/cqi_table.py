"""CQI 表与 SINR 映射 (3GPP TS 38.214 Table 5.2.2.1-2/3/4)

CQI 定义: UE 反馈的信道质量指标 (0-15)
- CQI 0: 信道太差, out of range
- CQI 1-15: 对应不同的 (调制阶数, 码率) 组合
- 含义: 在当前信道条件下, 使用该 CQI 对应的 MCS 传输, BLER ≤ 10%

SINR → CQI 映射: 通过链路级仿真校准的阈值
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CQIEntry:
    """CQI 表项"""
    cqi_index: int
    modulation: str           # QPSK, 16QAM, 64QAM, 256QAM
    modulation_order: int     # Qm: 2, 4, 6, 8
    code_rate_x1024: int      # R × 1024
    spectral_efficiency: float  # bits/RE


# ============================================================
# CQI Table 1 (TS 38.214 Table 5.2.2.1-2) — 64QAM
# ============================================================
CQI_TABLE_1 = [
    CQIEntry(0,  'out_of_range', 0, 0,    0.0),
    CQIEntry(1,  'QPSK',    2,  78,   0.1523),
    CQIEntry(2,  'QPSK',    2,  120,  0.2344),
    CQIEntry(3,  'QPSK',    2,  193,  0.3770),
    CQIEntry(4,  'QPSK',    2,  308,  0.6016),
    CQIEntry(5,  'QPSK',    2,  449,  0.8770),
    CQIEntry(6,  'QPSK',    2,  602,  1.1758),
    CQIEntry(7,  '16QAM',   4,  378,  1.4766),
    CQIEntry(8,  '16QAM',   4,  490,  1.9141),
    CQIEntry(9,  '16QAM',   4,  616,  2.4063),
    CQIEntry(10, '64QAM',   6,  466,  2.7305),
    CQIEntry(11, '64QAM',   6,  567,  3.3223),
    CQIEntry(12, '64QAM',   6,  666,  3.9023),
    CQIEntry(13, '64QAM',   6,  772,  4.5234),
    CQIEntry(14, '64QAM',   6,  873,  5.1152),
    CQIEntry(15, '64QAM',   6,  948,  5.5547),
]

# ============================================================
# CQI Table 2 (TS 38.214 Table 5.2.2.1-3) — 256QAM
# ============================================================
CQI_TABLE_2 = [
    CQIEntry(0,  'out_of_range', 0, 0,    0.0),
    CQIEntry(1,  'QPSK',    2,  78,   0.1523),
    CQIEntry(2,  'QPSK',    2,  193,  0.3770),
    CQIEntry(3,  'QPSK',    2,  449,  0.8770),
    CQIEntry(4,  '16QAM',   4,  378,  1.4766),
    CQIEntry(5,  '16QAM',   4,  490,  1.9141),
    CQIEntry(6,  '16QAM',   4,  616,  2.4063),
    CQIEntry(7,  '64QAM',   6,  466,  2.7305),
    CQIEntry(8,  '64QAM',   6,  567,  3.3223),
    CQIEntry(9,  '64QAM',   6,  666,  3.9023),
    CQIEntry(10, '64QAM',   6,  772,  4.5234),
    CQIEntry(11, '64QAM',   6,  873,  5.1152),
    CQIEntry(12, '256QAM',  8,  711,  5.5547),
    CQIEntry(13, '256QAM',  8,  797,  6.2266),
    CQIEntry(14, '256QAM',  8,  885,  6.9141),
    CQIEntry(15, '256QAM',  8,  948,  7.4063),
]

CQI_TABLES = {1: CQI_TABLE_1, 2: CQI_TABLE_2}

# ============================================================
# SINR 阈值 (dB) — 链路级仿真校准结果
# CQI i 的 SINR 阈值: 使用 CQI i 对应 MCS 传输, BLER=10% 时的 SINR
# 参考: 3GPP R1-073505, 业界广泛使用的校准值
# ============================================================
CQI_SINR_THRESHOLDS_DB = {
    1: np.array([
        -999.0,  # CQI 0 (out of range)
        -6.7,    # CQI 1
        -4.7,    # CQI 2
        -2.3,    # CQI 3
         0.2,    # CQI 4
         2.4,    # CQI 5
         4.3,    # CQI 6
         5.9,    # CQI 7
         8.1,    # CQI 8
        10.3,    # CQI 9
        11.7,    # CQI 10
        14.1,    # CQI 11
        16.3,    # CQI 12
        18.7,    # CQI 13
        21.0,    # CQI 14
        22.7,    # CQI 15
    ]),
}


class CQITable:
    """CQI 表管理器"""

    def __init__(self, table_index: int = 1):
        self.table_index = table_index
        self.entries = CQI_TABLES[table_index]
        self._thresholds = CQI_SINR_THRESHOLDS_DB.get(table_index,
                                                        CQI_SINR_THRESHOLDS_DB[1])

    def get_entry(self, cqi_index: int) -> CQIEntry:
        return self.entries[cqi_index]


def sinr_to_cqi(sinr_db: float, table_index: int = 1) -> int:
    """SINR (dB) → CQI 映射

    选择满足 SINR ≥ threshold 的最高 CQI。
    """
    thresholds = CQI_SINR_THRESHOLDS_DB.get(table_index, CQI_SINR_THRESHOLDS_DB[1])
    cqi = 0
    for i in range(15, 0, -1):
        if sinr_db >= thresholds[i]:
            cqi = i
            break
    return cqi


def cqi_to_sinr(cqi_index: int, table_index: int = 1) -> float:
    """CQI → SINR 阈值 (dB)"""
    thresholds = CQI_SINR_THRESHOLDS_DB.get(table_index, CQI_SINR_THRESHOLDS_DB[1])
    if cqi_index < 0 or cqi_index >= len(thresholds):
        return -30.0
    return float(thresholds[cqi_index])


def sinr_to_cqi_batch(sinr_db: np.ndarray, table_index: int = 1) -> np.ndarray:
    """批量 SINR → CQI"""
    thresholds = CQI_SINR_THRESHOLDS_DB.get(table_index, CQI_SINR_THRESHOLDS_DB[1])
    cqi = np.zeros_like(sinr_db, dtype=np.int32)
    for i in range(15, 0, -1):
        cqi = np.where((sinr_db >= thresholds[i]) & (cqi == 0), i, cqi)
    return cqi

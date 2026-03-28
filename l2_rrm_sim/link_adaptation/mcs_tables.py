"""MCS 表管理 (TS 38.214 Table 5.1.3.1-1/2/3)

提供 MCS index 到 (调制阶数, 码率) 的映射。
"""

from ..core.nr_constants import MCS_TABLES


def get_mcs_params(mcs_index: int, table_index: int = 1):
    """获取 MCS 参数

    Args:
        mcs_index: MCS 索引
        table_index: MCS 表索引 (1/2/3)

    Returns:
        (Qm, R_x1024): 调制阶数和码率×1024
    """
    qm_arr, rate_arr = MCS_TABLES[table_index]
    if mcs_index < 0 or mcs_index >= len(qm_arr):
        raise ValueError(f"MCS index {mcs_index} out of range for table {table_index}")
    return int(qm_arr[mcs_index]), float(rate_arr[mcs_index])


def get_spectral_efficiency(mcs_index: int, table_index: int = 1) -> float:
    """获取频谱效率 (bits/RE)"""
    qm, rate_x1024 = get_mcs_params(mcs_index, table_index)
    return qm * rate_x1024 / 1024.0


def get_max_mcs_index(table_index: int = 1) -> int:
    """获取最大有效 MCS index"""
    qm_arr, _ = MCS_TABLES[table_index]
    return len(qm_arr) - 1

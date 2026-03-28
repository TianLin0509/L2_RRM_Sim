"""5G NR 相关计算工具

TBS 计算: TS 38.214 Section 5.1.3.2
RE 计算: TS 38.214 Section 5.1.3.2
"""

import numpy as np
from ..core.nr_constants import (
    MCS_TABLES, TBS_TABLE_SMALL, NUM_SC_PER_PRB, NUM_SYMBOLS_PER_SLOT
)


def get_mcs_params(mcs_index: int, table_index: int = 1):
    """获取 MCS 参数

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


def compute_num_re_per_prb(num_pdcch_symbols: int = 2,
                           dmrs_type: int = 1,
                           num_dmrs_cdm_groups: int = 2,
                           num_dmrs_symbols: int = 1,
                           overhead: int = 0) -> int:
    """计算每 PRB 可用 RE 数 (TS 38.214 Section 5.1.3.2)

    N_RE' = N_sc_RB × N_symb_sh - N_DMRS_PRB - N_oh_PRB
    N_RE = min(156, N_RE')
    """
    n_symb_sh = NUM_SYMBOLS_PER_SLOT - num_pdcch_symbols

    # DMRS overhead per PRB
    if dmrs_type == 1:
        n_dmrs_per_symbol = num_dmrs_cdm_groups * 6
    else:  # Type 2
        n_dmrs_per_symbol = num_dmrs_cdm_groups * 4
    n_dmrs_prb = n_dmrs_per_symbol * num_dmrs_symbols

    n_re_prime = NUM_SC_PER_PRB * n_symb_sh - n_dmrs_prb - overhead
    return min(156, n_re_prime)


def compute_tbs(num_re_per_prb: int, num_prbs: int,
                mcs_index: int, num_layers: int,
                mcs_table_index: int = 1) -> int:
    """计算传输块大小 TBS (TS 38.214 Section 5.1.3.2)

    Args:
        num_re_per_prb: 每 PRB 可用 RE 数
        num_prbs: 分配的 PRB 数
        mcs_index: MCS 索引
        num_layers: 传输层数
        mcs_table_index: MCS 表索引

    Returns:
        TBS (bits)
    """
    if num_prbs <= 0 or num_layers <= 0:
        return 0

    qm, rate_x1024 = get_mcs_params(mcs_index, mcs_table_index)
    r = rate_x1024 / 1024.0

    # Step 1: 总 RE 数
    n_re = num_re_per_prb * num_prbs

    # Step 2: N_info
    n_info = n_re * r * qm * num_layers

    if n_info <= 3824:
        # Step 3: 量化
        n = max(3, int(np.floor(np.log2(n_info))) - 6)
        n_info_prime = max(24, int(2**n * np.floor(n_info / 2**n)))
        # 从小 TBS 表查找最近的 >= n_info_prime 的 TBS
        idx = np.searchsorted(TBS_TABLE_SMALL, n_info_prime)
        if idx >= len(TBS_TABLE_SMALL):
            idx = len(TBS_TABLE_SMALL) - 1
        tbs = int(TBS_TABLE_SMALL[idx])
    else:
        # Step 4: N_info > 3824
        n = int(np.floor(np.log2(n_info - 24))) - 5
        n_info_prime = max(3840, int(2**n * np.round((n_info - 24) / 2**n)))

        if r <= 0.25:
            c = int(np.ceil((n_info_prime + 24) / 3816))
            tbs = 8 * c * int(np.ceil((n_info_prime + 24) / (8 * c))) - 24
        else:
            if n_info_prime + 24 > 8424:
                c = int(np.ceil((n_info_prime + 24) / 8424))
                tbs = 8 * c * int(np.ceil((n_info_prime + 24) / (8 * c))) - 24
            else:
                tbs = 8 * int(np.ceil((n_info_prime + 24) / 8)) - 24

    return tbs


def compute_num_code_blocks(tbs: int, r: float) -> tuple:
    """计算码块数和码块大小 (简化版)

    Returns:
        (num_cb, cbs): 码块数和码块大小(bits)
    """
    if tbs <= 0:
        return 0, 0

    # LDPC base graph selection (TS 38.212 Section 6.2.2)
    if tbs <= 292 or (tbs <= 3824 and r <= 0.67):
        k_cb_max = 3840  # Base graph 2
    else:
        k_cb_max = 8448  # Base graph 1

    # CRC bits
    if tbs > 3824:
        l_crc = 24  # CRC24A for TB, CRC24B per CB
    else:
        l_crc = 16  # CRC16

    b = tbs + l_crc  # total bits including TB CRC

    if b <= k_cb_max:
        num_cb = 1
        cbs = b
    else:
        # 每个CB加24bit CRC24B
        num_cb = int(np.ceil(b / (k_cb_max - 24)))
        cbs = int(np.ceil(b / num_cb)) + 24  # 近似

    return num_cb, min(cbs, k_cb_max)

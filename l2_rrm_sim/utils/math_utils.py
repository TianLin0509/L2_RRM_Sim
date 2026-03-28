"""数学工具函数"""

import numpy as np


def db_to_linear(x_db):
    """dB 转线性"""
    return 10.0 ** (np.asarray(x_db, dtype=np.float64) / 10.0)


def linear_to_db(x_lin):
    """线性转 dB"""
    return 10.0 * np.log10(np.maximum(np.asarray(x_lin, dtype=np.float64), 1e-30))


def dbm_to_watt(x_dbm):
    """dBm 转 Watt"""
    return 10.0 ** ((np.asarray(x_dbm, dtype=np.float64) - 30.0) / 10.0)


def watt_to_dbm(x_watt):
    """Watt 转 dBm"""
    return 10.0 * np.log10(np.maximum(np.asarray(x_watt, dtype=np.float64), 1e-30)) + 30.0

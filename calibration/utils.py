"""校准验证共享工具 — 绘图样式、报告生成、参考数据"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "calibration_report"
FIGURES_DIR = REPORT_DIR / "figures"


def setup_plot_style():
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'savefig.bbox': 'tight',
        'savefig.dpi': 150,
    })


def save_figure(fig, name: str):
    """保存图表到 calibration_report/figures/"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(str(path))
    plt.close(fig)
    print(f"  Figure saved: {path}")
    return path


def write_report(filename: str, content: str):
    """写入报告到 calibration_report/"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    path.write_text(content, encoding='utf-8')
    print(f"  Report saved: {path}")
    return path


# ---- 3GPP 参考数据 ----

# AWGN BLER=10% 工作点 (dB), 来源: R1-1711982, MATLAB 5G Toolbox
REFERENCE_BLER_10PCT = {
    (1, 0):  -6.7,   # QPSK, R~0.12
    (1, 4):  -1.0,   # QPSK, R~0.44
    (1, 9):   4.0,   # QPSK, R~0.60
    (1, 15): 12.0,   # 64QAM, R~0.65
    (1, 20): 16.0,   # 64QAM, R~0.77
    (1, 27): 22.0,   # 64QAM, R~0.93
}

# TR 38.901 UMa 3.5GHz ISD=500m 校准参考
REFERENCE_CHANNEL_UMA = {
    'coupling_loss_median_db': 110.0,
    'coupling_loss_range_db': (85, 140),
    'geometry_sinr_median_db': 6.0,
    'geometry_sinr_range_db': (-5, 25),
}

# 频谱效率参考 (bps/Hz)
REFERENCE_SE = {
    'vienna_fdd_4x2_pf': {'min': 1.8, 'max': 2.2, 'label': 'Vienna SLS (FDD 4x2 PF)'},
    'r1_1801360_4t4r_pf': {'min': 2.0, 'max': 2.8, 'label': 'R1-1801360 (4T4R PF)'},
}

# ITU-R M.2412 Dense Urban eMBB
REFERENCE_ITU = {
    'dl_avg_se_bps_hz': 7.8,
    'dl_5pct_se_bps_hz': 0.225,
    'config': '200MHz TDD, 30kHz, 4x4 MIMO, PF, 10 UE/cell, 80% DL',
}

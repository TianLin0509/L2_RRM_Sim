"""Layer 1 校准: PHY 抽象层 BLER vs SNR 验证

扫描 6 个代表性 MCS (0, 4, 9, 15, 20, 27) 的 BLER-SNR 曲线,
找到 10% BLER 工作点并与 3GPP 参考值比较。
"""

import sys
import numpy as np
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration.utils import (
    setup_plot_style, save_figure, write_report, REFERENCE_BLER_10PCT,
)
from l2_rrm_sim.link_adaptation.bler_tables import BLERTableManager
from l2_rrm_sim.utils.nr_utils import get_mcs_params, compute_tbs, compute_num_code_blocks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---- 配置 ----
TABLE_INDEX = 1
MCS_LIST = [0, 4, 9, 15, 20, 27]
SNR_RANGE = np.arange(-15.0, 35.01, 0.1)
RE_PER_PRB = 132      # 14-symbol DL slot
NUM_PRBS = 273         # FR1 100MHz @ 30kHz SCS
NUM_LAYERS = 1         # rank 1
BLER_TARGET = 0.10     # 10% BLER operating point


def compute_cbs_for_mcs(mcs_index: int) -> int:
    """计算给定 MCS 的码块大小 (CBS)"""
    qm, r_x1024 = get_mcs_params(mcs_index, table_index=TABLE_INDEX)
    code_rate = r_x1024 / 1024.0
    tbs = compute_tbs(RE_PER_PRB, NUM_PRBS, mcs_index, NUM_LAYERS,
                      mcs_table_index=TABLE_INDEX)
    num_cb, cbs = compute_num_code_blocks(tbs, code_rate)
    return cbs


def find_bler_10pct(snr_arr, bler_arr):
    """从 BLER 曲线中找到 10% BLER 对应的 SNR (线性插值)"""
    # BLER 应该是递减的; 找到跨越 0.10 的区间
    for i in range(len(bler_arr) - 1):
        if bler_arr[i] >= BLER_TARGET >= bler_arr[i + 1]:
            # 线性插值
            if bler_arr[i] == bler_arr[i + 1]:
                return snr_arr[i]
            t = (BLER_TARGET - bler_arr[i]) / (bler_arr[i + 1] - bler_arr[i])
            return snr_arr[i] + t * (snr_arr[i + 1] - snr_arr[i])
    # 未找到交叉点
    return None


def main():
    print("=" * 60)
    print("Layer 1: PHY Abstraction BLER vs SNR Calibration")
    print("=" * 60)

    setup_plot_style()
    bler_mgr = BLERTableManager()

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    results = []

    for idx, mcs in enumerate(MCS_LIST):
        # MCS 0-2 不在 Sionna 导出表中; lookup_bler 内部会回退到最近 MCS
        actual_mcs = mcs
        if not bler_mgr.has_entry(TABLE_INDEX, mcs):
            # 找最近的可用 MCS
            available = sorted(k[1] for k in bler_mgr._cbs_values if k[0] == TABLE_INDEX)
            nearest = min(available, key=lambda x: abs(x - mcs))
            print(f"  [INFO] MCS {mcs} not in BLER table, using MCS {nearest} as proxy")
            actual_mcs = nearest

        qm, r_x1024 = get_mcs_params(mcs, table_index=TABLE_INDEX)
        cbs = compute_cbs_for_mcs(mcs)
        mod_name = {2: 'QPSK', 4: '16QAM', 6: '64QAM', 8: '256QAM'}.get(qm, f'{qm}QAM')

        # 查看实际使用的 CBS (表中最大 2000)
        avail_cbs = bler_mgr._cbs_values.get((TABLE_INDEX, actual_mcs), [])
        used_cbs = min(avail_cbs, key=lambda x: abs(x - cbs)) if avail_cbs else cbs

        # 扫描 BLER (lookup_bler 内部处理 CBS 匹配和 MCS 回退)
        bler_curve = np.array([
            bler_mgr.lookup_bler(snr, mcs, cbs, TABLE_INDEX)
            for snr in SNR_RANGE
        ])

        # 找 10% BLER 工作点
        snr_10pct = find_bler_10pct(SNR_RANGE, bler_curve)
        ref_snr = REFERENCE_BLER_10PCT.get((TABLE_INDEX, mcs))
        deviation = (snr_10pct - ref_snr) if (snr_10pct is not None and ref_snr is not None) else None

        results.append({
            'mcs': mcs,
            'actual_mcs': actual_mcs,
            'mod': mod_name,
            'rate': r_x1024 / 1024.0,
            'cbs': cbs,
            'used_cbs': used_cbs,
            'snr_10pct': snr_10pct,
            'ref_snr': ref_snr,
            'deviation': deviation,
        })

        # 绘制曲线
        label = f'MCS {mcs} ({mod_name}, R={r_x1024/1024:.3f})'
        ax.semilogy(SNR_RANGE, np.clip(bler_curve, 1e-4, 1.0),
                    color=colors[idx], label=label, linewidth=2)

        # 标记 10% BLER 点
        if snr_10pct is not None:
            ax.plot(snr_10pct, BLER_TARGET, 'o', color=colors[idx],
                    markersize=8, zorder=5)

        # 标记参考点
        if ref_snr is not None:
            ax.plot(ref_snr, BLER_TARGET, 'x', color=colors[idx],
                    markersize=10, markeredgewidth=2.5, zorder=5)

    # 图表装饰
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BLER')
    ax.set_title('Layer 1: BLER vs SNR — MCS Table 1 (LDPC, Sionna)')
    ax.set_xlim(-15, 35)
    ax.set_ylim(1e-4, 1.0)
    ax.axhline(y=BLER_TARGET, color='gray', linestyle='--', alpha=0.5, label='10% BLER target')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # 添加图例说明
    ax.annotate('o = simulated 10% point\nx = 3GPP reference',
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    save_figure(fig, 'layer1_bler_vs_snr')

    # ---- 打印偏差表 ----
    print("\n  MCS  | Mod   | Rate  | CBS   | SNR@10% | Ref SNR | Delta (dB) | Status")
    print("  " + "-" * 75)
    all_pass = True
    for r in results:
        snr_str = f"{r['snr_10pct']:+.1f}" if r['snr_10pct'] is not None else "  N/A"
        ref_str = f"{r['ref_snr']:+.1f}" if r['ref_snr'] is not None else "  N/A"
        is_proxy = r['mcs'] != r['actual_mcs']
        if is_proxy:
            dev_str = "  N/A"
            status = f"SKIP(->MCS{r['actual_mcs']})"
        elif r['deviation'] is not None:
            dev_str = f"{r['deviation']:+.2f}"
            status = "PASS" if abs(r['deviation']) <= 1.0 else "FAIL"
            if status == "FAIL":
                all_pass = False
        else:
            dev_str = "  N/A"
            status = "N/A"
        print(f"  {r['mcs']:>3}  | {r['mod']:<5} | {r['rate']:.3f} | {r['cbs']:>5} | {snr_str:>7} | {ref_str:>7} | {dev_str:>10} | {status}")

    print()
    if all_pass:
        print("  RESULT: ALL PASS (deviation <= 1.0 dB)")
    else:
        print("  RESULT: SOME DEVIATIONS EXCEED 1.0 dB")

    # ---- 生成报告 ----
    report = _generate_report(results, all_pass)
    write_report('layer1_phy_abstraction.md', report)

    print("\nDone.")


def _generate_report(results, all_pass):
    lines = [
        "# Layer 1: PHY Abstraction BLER vs SNR Calibration",
        "",
        "## Overview",
        "",
        "Verify that EESM + Sionna BLER lookup tables produce BLER vs SNR curves",
        "consistent with 3GPP LDPC AWGN reference data.",
        "",
        "- **MCS Table**: 1 (TS 38.214 Table 5.1.3.1-1)",
        "- **MCS indices**: 0, 4, 9, 15, 20, 27",
        f"- **Config**: {RE_PER_PRB} RE/PRB, {NUM_PRBS} PRBs, rank {NUM_LAYERS}",
        f"- **Pass criterion**: 10% BLER operating point deviation <= 1.0 dB",
        "",
        "## BLER vs SNR Curves",
        "",
        "![BLER vs SNR](figures/layer1_bler_vs_snr.png)",
        "",
        "## 10% BLER Operating Point Deviation",
        "",
        "| MCS | Modulation | Code Rate | CBS | SNR@10% (dB) | 3GPP Ref (dB) | Delta (dB) | Status |",
        "|-----|-----------|-----------|-----|-------------|---------------|------------|--------|",
    ]
    for r in results:
        snr_str = f"{r['snr_10pct']:+.1f}" if r['snr_10pct'] is not None else "N/A"
        ref_str = f"{r['ref_snr']:+.1f}" if r['ref_snr'] is not None else "N/A"
        is_proxy = r['mcs'] != r['actual_mcs']
        if is_proxy:
            dev_str = "N/A"
            status = f"SKIP (proxy MCS {r['actual_mcs']})"
        elif r['deviation'] is not None:
            dev_str = f"{r['deviation']:+.2f}"
            status = "PASS" if abs(r['deviation']) <= 1.0 else "FAIL"
        else:
            dev_str = "N/A"
            status = "N/A"
        lines.append(
            f"| {r['mcs']} | {r['mod']} | {r['rate']:.3f} | {r['cbs']} | {snr_str} | {ref_str} | {dev_str} | {status} |"
        )

    lines.extend([
        "",
        "## Result",
        "",
        f"**{'ALL PASS' if all_pass else 'SOME DEVIATIONS EXCEED THRESHOLD'}**",
        "",
        "## CBS Mismatch Note",
        "",
        "Computed CBS (273 PRBs, rank 1) ranges from ~7000-8400 bits, but the Sionna",
        "BLER tables only contain CBS values up to 2000 bits. The lookup uses the",
        "nearest available CBS (2000). This CBS mismatch may contribute to deviations",
        "for higher MCS indices where LDPC performance is more CBS-sensitive.",
        "",
        "| MCS | Computed CBS | Used CBS |",
        "|-----|-------------|----------|",
    ])
    for r in results:
        lines.append(f"| {r['mcs']} | {r['cbs']} | {r['used_cbs']} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- BLER tables exported from Sionna LDPC simulation (AWGN channel)",
        "- Reference points from R1-1711982 and MATLAB 5G Toolbox",
        "- MCS 0 not in BLER tables (range starts at MCS 3); MCS 3 used as proxy",
        "- Circle markers (o) = simulated 10% point; Cross markers (x) = 3GPP reference",
    ])

    return "\n".join(lines)


if __name__ == '__main__':
    main()

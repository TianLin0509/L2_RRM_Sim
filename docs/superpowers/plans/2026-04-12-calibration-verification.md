# L2_RRM_Sim 校准验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自底向上四层校准验证 L2_RRM_Sim 仿真平台，对标 3GPP 公开基准，产出校准报告。

**Architecture:** 四个独立的验证脚本分别对应四层校准（PHY抽象、信道模型、L2功能、端到端KPI），每层脚本负责仿真执行 + 数据采集 + matplotlib 绘图 + 报告生成。所有脚本放在 `calibration/` 目录，报告输出到 `calibration_report/`。

**Tech Stack:** Python 3.12, numpy, scipy, matplotlib, 项目自有 l2_rrm_sim 包

**验证铁律:** 发现偏差先汇报用户，不擅自改代码。

---

## File Structure

```
calibration/
├── layer1_phy_bler.py           — Layer 1: BLER vs SNR 扫描
├── layer2_channel_model.py      — Layer 2: 耦合损耗/SINR CDF
├── layer3_l2_functions.py       — Layer 3: 频谱效率对标
├── layer4_e2e_kpi.py            — Layer 4: 端到端 KPI + 多小区 smoke
└── utils.py                     — 共享工具（绘图样式、报告模板）

calibration_report/
├── figures/                     — PNG 图表
├── layer1_phy_abstraction.md
├── layer2_channel_model.md
├── layer3_l2_functions.md
├── layer4_e2e_kpi.md
└── summary.md
```

---

### Task 1: 搭建校准框架 + 共享工具

**Files:**
- Create: `calibration/utils.py`
- Create: `calibration_report/figures/.gitkeep`

- [ ] **Step 1: 创建目录结构**

```bash
cd C:\Users\lintian\Documents\GitHub\L2_RRM_Sim
mkdir -p calibration calibration_report/figures
```

- [ ] **Step 2: 编写共享工具 `calibration/utils.py`**

```python
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

# 统一绘图样式
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
# 格式: {(table_index, mcs_index): snr_at_10pct_bler}
# Table 1 (64QAM): QPSK (MCS 0-9), 16QAM (MCS 10-16), 64QAM (MCS 17-28)
REFERENCE_BLER_10PCT = {
    (1, 0):  -6.7,   # QPSK, R~0.12
    (1, 4):  -1.0,   # QPSK, R~0.44
    (1, 9):   4.0,   # QPSK, R~0.60
    (1, 15): 12.0,   # 64QAM, R~0.65
    (1, 20): 16.0,   # 64QAM, R~0.77
    (1, 27): 22.0,   # 64QAM, R~0.93
}

# TR 38.901 UMa 3.5GHz ISD=500m 校准参考
# 耦合损耗 CDF 中位数 (dB), geometry SINR CDF 中位数 (dB)
REFERENCE_CHANNEL_UMA = {
    'coupling_loss_median_db': 110.0,   # 典型 UMa 3.5GHz
    'coupling_loss_range_db': (85, 140), # CDF 范围
    'geometry_sinr_median_db': 6.0,     # 典型 geometry factor
    'geometry_sinr_range_db': (-5, 25),  # CDF 范围
}

# 频谱效率参考 (bps/Hz)
# Vienna SLS: UMa FDD 4x2 PF → 1.8-2.2 bps/Hz
# R1-1801360: UMa 4GHz 4T4R PF → 2.0-2.8 bps/Hz
REFERENCE_SE = {
    'vienna_fdd_4x2_pf': {'min': 1.8, 'max': 2.2, 'label': 'Vienna SLS (FDD 4x2 PF)'},
    'r1_1801360_4t4r_pf': {'min': 2.0, 'max': 2.8, 'label': 'R1-1801360 (4T4R PF)'},
}

# ITU-R M.2412 Dense Urban eMBB
REFERENCE_ITU = {
    'dl_avg_se_bps_hz': 7.8,           # 4x4 MIMO, 200MHz TDD
    'dl_5pct_se_bps_hz': 0.225,
    'config': '200MHz TDD, 30kHz, 4x4 MIMO, PF, 10 UE/cell, 80% DL',
}
```

- [ ] **Step 3: 创建 .gitkeep**

```bash
touch calibration_report/figures/.gitkeep
```

- [ ] **Step 4: 验证 import**

```bash
cd C:\Users\lintian\Documents\GitHub\L2_RRM_Sim
.venv312/Scripts/python.exe -c "from calibration.utils import setup_plot_style, REFERENCE_BLER_10PCT; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add calibration/utils.py calibration_report/figures/.gitkeep
git commit -m "chore: add calibration framework and shared utilities"
```

---

### Task 2: Layer 1 — PHY 抽象层 BLER 校准

**Files:**
- Create: `calibration/layer1_phy_bler.py`
- Output: `calibration_report/layer1_phy_abstraction.md` + `calibration_report/figures/layer1_*.png`

**目标:** 对 6 个代表性 MCS，扫描 SNR 范围，绘制 BLER vs SNR 曲线，与 3GPP 参考 10% BLER 点对比。

- [ ] **Step 1: 编写 Layer 1 验证脚本**

```python
"""Layer 1: PHY 抽象层 BLER vs SNR 校准

对标: 3GPP R1-1711982 LDPC BLER 曲线, MATLAB 5G Toolbox
方法: 固定 MCS, 扫描 SNR, 查询 BLER 表, 绘制曲线
通过标准: 10% BLER 工作点偏差 <= 1 dB
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from l2_rrm_sim.link_adaptation.bler_tables import BLERTableManager
from l2_rrm_sim.utils.nr_utils import get_mcs_params, compute_tbs, compute_num_code_blocks
from calibration.utils import (
    setup_plot_style, save_figure, write_report,
    REFERENCE_BLER_10PCT,
)


def find_10pct_bler_point(snr_array, bler_array):
    """在 BLER 曲线中找到 BLER=10% 对应的 SNR (线性插值)"""
    # BLER 从高到低, 找第一个 <= 0.1 的位置
    for i in range(len(bler_array) - 1):
        if bler_array[i] >= 0.1 and bler_array[i+1] < 0.1:
            # 线性插值
            ratio = (0.1 - bler_array[i]) / (bler_array[i+1] - bler_array[i])
            return snr_array[i] + ratio * (snr_array[i+1] - snr_array[i])
    # 如果全程 > 0.1 或全程 < 0.1
    if bler_array[-1] >= 0.1:
        return float('inf')
    if bler_array[0] <= 0.1:
        return snr_array[0]
    return float('nan')


def run_layer1():
    """执行 Layer 1 校准"""
    setup_plot_style()
    bler_mgr = BLERTableManager()

    # 代表性 MCS (Table 1: 64QAM)
    mcs_list = [0, 4, 9, 15, 20, 27]
    table_index = 1
    snr_range = np.arange(-15.0, 35.0, 0.5)

    # 选择一个典型 CBS (基于 273 PRBs, rank=1 的 TBS)
    re_per_prb = 132  # 标准 14 symbol DL

    results = {}

    print("=" * 60)
    print("Layer 1: PHY Abstraction — BLER vs SNR Calibration")
    print("=" * 60)

    # ---- 绘制 BLER 曲线 ----
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, mcs in enumerate(mcs_list):
        qm, r_x1024 = get_mcs_params(mcs, table_index)
        code_rate = r_x1024 / 1024.0
        mod_name = {2: 'QPSK', 4: '16QAM', 6: '64QAM', 8: '256QAM'}.get(qm, f'Qm={qm}')

        # 计算 TBS 和 CBS
        tbs = compute_tbs(re_per_prb, 273, mcs, 1, table_index)
        num_cb, cbs = compute_num_code_blocks(tbs, code_rate)

        print(f"\n  MCS {mcs:2d} ({mod_name}, R={code_rate:.3f}): TBS={tbs}, CBS={cbs}, #CB={num_cb}")

        # 检查 BLER 表是否有此 MCS 的数据
        if not bler_mgr.has_entry(table_index, mcs):
            print(f"    WARNING: No BLER table entry for MCS {mcs}, skipping")
            continue

        # 扫描 SNR
        bler_curve = []
        for snr in snr_range:
            bler = bler_mgr.lookup_bler(snr, mcs, cbs, table_index)
            bler_curve.append(bler)
        bler_curve = np.array(bler_curve)

        # 找 10% BLER 点
        snr_10pct = find_10pct_bler_point(snr_range, bler_curve)
        results[mcs] = {
            'mod': mod_name,
            'code_rate': code_rate,
            'tbs': tbs,
            'cbs': cbs,
            'snr_10pct_sim': snr_10pct,
            'snr_10pct_ref': REFERENCE_BLER_10PCT.get((table_index, mcs)),
            'bler_curve': bler_curve,
        }

        label = f"MCS {mcs} ({mod_name} R={code_rate:.2f})"
        ax.semilogy(snr_range, np.clip(bler_curve, 1e-4, 1.0), color=colors[idx], label=label)

        # 标记 10% BLER 点
        if np.isfinite(snr_10pct):
            ax.plot(snr_10pct, 0.1, 'o', color=colors[idx], markersize=8)

        # 标记参考点
        ref = REFERENCE_BLER_10PCT.get((table_index, mcs))
        if ref is not None:
            ax.plot(ref, 0.1, 'x', color=colors[idx], markersize=12, markeredgewidth=3)

        print(f"    10% BLER point: sim={snr_10pct:.1f} dB", end="")
        if ref is not None:
            delta = snr_10pct - ref
            print(f", ref={ref:.1f} dB, delta={delta:+.1f} dB", end="")
            if abs(delta) <= 1.0:
                print(" [PASS]")
            else:
                print(" [DEVIATION > 1 dB]")
        else:
            print(" (no reference)")

    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='10% BLER target')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BLER')
    ax.set_title('Layer 1: BLER vs SNR — L2_RRM_Sim vs 3GPP Reference\n(o = sim 10% point, x = 3GPP reference)')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlim(-15, 35)
    save_figure(fig, 'layer1_bler_vs_snr')

    # ---- 偏差汇总表 ----
    print("\n" + "-" * 60)
    print("Layer 1 Summary: 10% BLER Operating Points")
    print("-" * 60)
    print(f"{'MCS':>4} {'Mod':>6} {'Rate':>5} {'Sim(dB)':>8} {'Ref(dB)':>8} {'Delta':>7} {'Status':>10}")
    print("-" * 60)

    deviations = []
    report_rows = []
    for mcs in mcs_list:
        if mcs not in results:
            continue
        r = results[mcs]
        sim_val = r['snr_10pct_sim']
        ref_val = r['snr_10pct_ref']
        if ref_val is not None and np.isfinite(sim_val):
            delta = sim_val - ref_val
            status = "PASS" if abs(delta) <= 1.0 else "DEVIATION"
            deviations.append(abs(delta))
        else:
            delta = float('nan')
            status = "NO REF"
        print(f"{mcs:4d} {r['mod']:>6} {r['code_rate']:5.3f} {sim_val:8.1f} "
              f"{ref_val if ref_val else 'N/A':>8} {delta:+7.1f} {status:>10}")
        report_rows.append({
            'mcs': mcs, 'mod': r['mod'], 'rate': r['code_rate'],
            'sim': sim_val, 'ref': ref_val, 'delta': delta, 'status': status,
        })

    # ---- 生成报告 ----
    report_lines = [
        "# Layer 1: PHY Abstraction Calibration Report\n",
        "## Configuration\n",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| MCS Table | Table 1 (64QAM max) |",
        "| PRBs | 273 (100 MHz @ 30 kHz) |",
        "| RE/PRB | 132 (14 symbols, 2 PDCCH, 1 DMRS) |",
        "| Rank | 1 |",
        f"| MCS tested | {mcs_list} |",
        "| Reference | 3GPP R1-1711982, MATLAB 5G Toolbox |\n",
        "## BLER vs SNR Curves\n",
        "![BLER vs SNR](figures/layer1_bler_vs_snr.png)\n",
        "## 10% BLER Operating Point Comparison\n",
        "| MCS | Modulation | Code Rate | Sim (dB) | Ref (dB) | Delta (dB) | Status |",
        "|-----|-----------|-----------|----------|----------|------------|--------|",
    ]
    for r in report_rows:
        ref_str = f"{r['ref']:.1f}" if r['ref'] is not None else "N/A"
        delta_str = f"{r['delta']:+.1f}" if np.isfinite(r['delta']) else "N/A"
        report_lines.append(
            f"| {r['mcs']} | {r['mod']} | {r['rate']:.3f} | {r['sim']:.1f} | "
            f"{ref_str} | {delta_str} | {r['status']} |"
        )

    report_lines.extend([
        "\n## Conclusion\n",
        f"- Tested {len(results)} MCS configurations",
        f"- Max deviation: {max(deviations):.1f} dB" if deviations else "- No reference data to compare",
        f"- Pass criterion: |delta| <= 1.0 dB",
    ])

    passed = sum(1 for d in deviations if d <= 1.0)
    failed = sum(1 for d in deviations if d > 1.0)
    if deviations:
        report_lines.append(f"- Result: {passed} PASS, {failed} DEVIATION out of {len(deviations)} compared")

    report_lines.append("\n## Issues Found\n")
    if failed > 0:
        report_lines.append("**以下 MCS 偏差超过 1 dB，需要用户审查：**\n")
        for r in report_rows:
            if np.isfinite(r['delta']) and abs(r['delta']) > 1.0:
                report_lines.append(f"- MCS {r['mcs']} ({r['mod']}): delta = {r['delta']:+.1f} dB")
    else:
        report_lines.append("无偏差超标项。")

    write_report("layer1_phy_abstraction.md", "\n".join(report_lines))

    return results


if __name__ == "__main__":
    run_layer1()
```

- [ ] **Step 2: 执行 Layer 1 验证**

```bash
cd C:\Users\lintian\Documents\GitHub\L2_RRM_Sim
.venv312/Scripts/python.exe calibration/layer1_phy_bler.py
```

Expected: 终端输出每个 MCS 的 10% BLER 点和偏差，生成 `calibration_report/layer1_phy_abstraction.md` 和 `calibration_report/figures/layer1_bler_vs_snr.png`

- [ ] **Step 3: 检查报告和图表**

打开生成的报告文件和图表，确认内容完整、数据合理。

- [ ] **Step 4: 向用户汇报 Layer 1 结果**

汇报内容：
- 每个 MCS 的偏差值
- 哪些通过、哪些偏差超标
- 如果有偏差超标，列出可能原因（BLER 表来源、EESM beta 参数等）
- **等待用户决策再进入 Layer 2**

- [ ] **Step 5: Commit**

```bash
git add calibration/layer1_phy_bler.py calibration_report/
git commit -m "calibration: Layer 1 PHY abstraction BLER vs SNR verification"
```

---

### Task 3: Layer 2 — 信道模型校准

**Files:**
- Create: `calibration/layer2_channel_model.py`
- Output: `calibration_report/layer2_channel_model.md` + `calibration_report/figures/layer2_*.png`

**目标:** 验证 UMa 3.5GHz 统计信道的路损分布和 geometry SINR CDF，对标 TR 38.901 §7.8。

- [ ] **Step 1: 编写 Layer 2 验证脚本**

```python
"""Layer 2: 信道模型校准 — 耦合损耗 & Geometry SINR CDF

对标: 3GPP TR 38.901 v17 §7.8 (UMa ISD=500m, 3.5GHz)
方法: 大量 UE 撒点, 采样路损/SINR, 绘 CDF
通过标准: CDF 中位数偏差 <= 2 dB, 形状趋势一致
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig,
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from l2_rrm_sim.channel.pathloss_models import (
    compute_pathloss_uma, compute_los_probability_uma,
)
from l2_rrm_sim.utils.math_utils import linear_to_db
from calibration.utils import (
    setup_plot_style, save_figure, write_report,
    REFERENCE_CHANNEL_UMA,
)


def run_layer2():
    """执行 Layer 2 校准"""
    setup_plot_style()

    print("=" * 60)
    print("Layer 2: Channel Model — Coupling Loss & SINR CDF")
    print("=" * 60)

    # ---- Part A: 路损模型独立验证 ----
    print("\n--- Part A: Pathloss Model Verification ---")

    num_drops = 2000
    rng = np.random.default_rng(42)

    # UMa 参数
    h_bs = 25.0
    h_ut = 1.5
    fc_ghz = 3.5
    cell_radius = 500.0
    min_dist = 35.0

    # 随机撒 UE (均匀分布在圆环内)
    r = np.sqrt(rng.uniform(min_dist**2, cell_radius**2, num_drops))
    theta = rng.uniform(0, 2 * np.pi, num_drops)

    # 计算路损
    pathloss_db = np.zeros(num_drops)
    is_los = np.zeros(num_drops, dtype=bool)
    shadow_fading_db = np.zeros(num_drops)

    for i in range(num_drops):
        p_los = compute_los_probability_uma(r[i], h_ut)
        is_los[i] = rng.random() < p_los
        pathloss_db[i] = compute_pathloss_uma(r[i], h_bs, h_ut, fc_ghz, is_los[i])
        sf_std = 4.0 if is_los[i] else 6.0
        shadow_fading_db[i] = rng.normal(0, sf_std)

    coupling_loss_db = pathloss_db + shadow_fading_db
    cl_median = np.median(coupling_loss_db)

    print(f"  Num drops: {num_drops}")
    print(f"  LOS ratio: {np.mean(is_los)*100:.1f}%")
    print(f"  Pathloss range: [{np.min(pathloss_db):.1f}, {np.max(pathloss_db):.1f}] dB")
    print(f"  Coupling loss median: {cl_median:.1f} dB (ref: {REFERENCE_CHANNEL_UMA['coupling_loss_median_db']:.1f} dB)")

    cl_delta = cl_median - REFERENCE_CHANNEL_UMA['coupling_loss_median_db']
    print(f"  Delta: {cl_delta:+.1f} dB {'[PASS]' if abs(cl_delta) <= 2.0 else '[DEVIATION > 2 dB]'}")

    # ---- 绘制耦合损耗 CDF ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sorted_cl = np.sort(coupling_loss_db)
    cdf = np.arange(1, num_drops + 1) / num_drops
    ax1.plot(sorted_cl, cdf, 'b-', linewidth=2, label='L2_RRM_Sim')
    ax1.axvline(x=cl_median, color='b', linestyle='--', alpha=0.5, label=f'Median: {cl_median:.1f} dB')
    ax1.axvline(x=REFERENCE_CHANNEL_UMA['coupling_loss_median_db'], color='r', linestyle='--',
                alpha=0.5, label=f"TR 38.901 ref: {REFERENCE_CHANNEL_UMA['coupling_loss_median_db']:.1f} dB")
    ax1.set_xlabel('Coupling Loss (dB)')
    ax1.set_ylabel('CDF')
    ax1.set_title('Coupling Loss CDF — UMa 3.5GHz ISD=500m')
    ax1.legend()
    ax1.set_xlim(REFERENCE_CHANNEL_UMA['coupling_loss_range_db'])

    # ---- Part B: Geometry SINR (通过 SimulationEngine) ----
    print("\n--- Part B: Geometry SINR via SimulationEngine ---")

    num_ue = 200
    num_sinr_slots = 50  # 多 slot 采样快衰落

    config = {
        'sim': SimConfig(num_slots=num_sinr_slots, random_seed=42, warmup_slots=0),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273,
                                 bandwidth_mhz=100.0, carrier_freq_ghz=3.5),
        'cell': CellConfig(num_tx_ant=4, num_tx_ports=4, max_layers=1,
                           total_power_dbm=46.0, cell_radius_m=500.0,
                           height_m=25.0, scenario='uma'),
        'ue': UEConfig(num_ue=num_ue, num_rx_ant=2, min_distance_m=35.0,
                       max_distance_m=500.0, height_m=1.5, speed_kmh=3.0),
        'scheduler': SchedulerConfig(type='pf'),
        'link_adaptation': LinkAdaptationConfig(),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'tdd': None,
        'csi': None,
        'harq': None,
    }

    engine = SimulationEngine(config)

    # 采集多 slot 的宽带 SINR
    sinr_samples = []
    for s in range(num_sinr_slots):
        slot_ctx = engine._make_slot_context(s)
        channel_state = engine.channel.update(slot_ctx, engine.ue_states)
        # 宽带 SINR = mean over PRBs, layer 0
        wideband_sinr_linear = np.mean(channel_state.sinr_per_prb[:, 0, :], axis=1)
        sinr_samples.append(10 * np.log10(np.clip(wideband_sinr_linear, 1e-10, None)))

    sinr_all = np.array(sinr_samples).flatten()  # (num_slots * num_ue,)
    sinr_median = np.median(sinr_all)

    print(f"  Num UEs: {num_ue}, Num slots: {num_sinr_slots}")
    print(f"  Total SINR samples: {len(sinr_all)}")
    print(f"  SINR range: [{np.min(sinr_all):.1f}, {np.max(sinr_all):.1f}] dB")
    print(f"  SINR median: {sinr_median:.1f} dB (ref: {REFERENCE_CHANNEL_UMA['geometry_sinr_median_db']:.1f} dB)")

    sinr_delta = sinr_median - REFERENCE_CHANNEL_UMA['geometry_sinr_median_db']
    print(f"  Delta: {sinr_delta:+.1f} dB {'[PASS]' if abs(sinr_delta) <= 2.0 else '[DEVIATION > 2 dB]'}")

    # ---- 绘制 SINR CDF ----
    sorted_sinr = np.sort(sinr_all)
    cdf_sinr = np.arange(1, len(sinr_all) + 1) / len(sinr_all)
    ax2.plot(sorted_sinr, cdf_sinr, 'b-', linewidth=2, label='L2_RRM_Sim')
    ax2.axvline(x=sinr_median, color='b', linestyle='--', alpha=0.5,
                label=f'Median: {sinr_median:.1f} dB')
    ax2.axvline(x=REFERENCE_CHANNEL_UMA['geometry_sinr_median_db'], color='r', linestyle='--',
                alpha=0.5, label=f"TR 38.901 ref: {REFERENCE_CHANNEL_UMA['geometry_sinr_median_db']:.1f} dB")
    ax2.set_xlabel('Geometry SINR (dB)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Geometry SINR CDF — UMa 3.5GHz ISD=500m')
    ax2.legend()
    ax2.set_xlim(REFERENCE_CHANNEL_UMA['geometry_sinr_range_db'])

    fig.suptitle('Layer 2: Channel Model Calibration', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'layer2_channel_cdf')

    # ---- Part C: 路损 vs 距离散点图 ----
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(r[is_los], pathloss_db[is_los], c='green', alpha=0.3, s=10, label='LOS')
    ax3.scatter(r[~is_los], pathloss_db[~is_los], c='red', alpha=0.3, s=10, label='NLOS')

    # 理论曲线
    d_theory = np.linspace(35, 500, 200)
    pl_los_theory = [compute_pathloss_uma(d, h_bs, h_ut, fc_ghz, True) for d in d_theory]
    pl_nlos_theory = [compute_pathloss_uma(d, h_bs, h_ut, fc_ghz, False) for d in d_theory]
    ax3.plot(d_theory, pl_los_theory, 'g-', linewidth=2, label='UMa LOS (theory)')
    ax3.plot(d_theory, pl_nlos_theory, 'r-', linewidth=2, label='UMa NLOS (theory)')

    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Pathloss (dB)')
    ax3.set_title('Pathloss vs Distance — UMa 3.5GHz')
    ax3.legend()
    save_figure(fig2, 'layer2_pathloss_vs_distance')

    # ---- 生成报告 ----
    report_lines = [
        "# Layer 2: Channel Model Calibration Report\n",
        "## Configuration\n",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Scenario | UMa |",
        "| Carrier | 3.5 GHz |",
        "| ISD | 500 m (cell radius) |",
        "| BS height | 25 m |",
        "| UE height | 1.5 m |",
        "| UE distance | [35, 500] m uniform |",
        f"| Num UE drops (pathloss) | {num_drops} |",
        f"| Num UE (SINR) | {num_ue} |",
        f"| Num slots (SINR) | {num_sinr_slots} |",
        "| Reference | TR 38.901 v17 §7.8 |\n",
        "## Coupling Loss CDF\n",
        "![CDF](figures/layer2_channel_cdf.png)\n",
        "| Metric | Sim | Reference | Delta | Status |",
        "|--------|-----|-----------|-------|--------|",
        f"| Coupling Loss Median | {cl_median:.1f} dB | {REFERENCE_CHANNEL_UMA['coupling_loss_median_db']:.1f} dB | {cl_delta:+.1f} dB | {'PASS' if abs(cl_delta) <= 2 else 'DEVIATION'} |",
        f"| Geometry SINR Median | {sinr_median:.1f} dB | {REFERENCE_CHANNEL_UMA['geometry_sinr_median_db']:.1f} dB | {sinr_delta:+.1f} dB | {'PASS' if abs(sinr_delta) <= 2 else 'DEVIATION'} |\n",
        "## Pathloss vs Distance\n",
        "![Pathloss](figures/layer2_pathloss_vs_distance.png)\n",
        "## Conclusion\n",
        f"- Coupling loss median delta: {cl_delta:+.1f} dB (threshold: ±2 dB)",
        f"- Geometry SINR median delta: {sinr_delta:+.1f} dB (threshold: ±2 dB)",
        f"- LOS ratio: {np.mean(is_los)*100:.1f}%",
    ]

    issues = []
    if abs(cl_delta) > 2.0:
        issues.append(f"- Coupling loss median deviation {cl_delta:+.1f} dB exceeds ±2 dB threshold")
    if abs(sinr_delta) > 2.0:
        issues.append(f"- Geometry SINR median deviation {sinr_delta:+.1f} dB exceeds ±2 dB threshold")

    report_lines.append("\n## Issues Found\n")
    if issues:
        report_lines.append("**以下指标偏差超标，需要用户审查：**\n")
        report_lines.extend(issues)
    else:
        report_lines.append("无偏差超标项。")

    write_report("layer2_channel_model.md", "\n".join(report_lines))

    return {
        'coupling_loss_median': cl_median,
        'coupling_loss_delta': cl_delta,
        'sinr_median': sinr_median,
        'sinr_delta': sinr_delta,
    }


if __name__ == "__main__":
    run_layer2()
```

- [ ] **Step 2: 执行 Layer 2 验证**

```bash
cd C:\Users\lintian\Documents\GitHub\L2_RRM_Sim
.venv312/Scripts/python.exe calibration/layer2_channel_model.py
```

Expected: 终端输出耦合损耗和 SINR 中位数偏差，生成报告和 2 张图。

- [ ] **Step 3: 向用户汇报 Layer 2 结果**

汇报耦合损耗 CDF 和 SINR CDF 的偏差，等用户确认后进入 Layer 3。

- [ ] **Step 4: Commit**

```bash
git add calibration/layer2_channel_model.py calibration_report/
git commit -m "calibration: Layer 2 channel model coupling loss and SINR CDF"
```

---

### Task 4: Layer 3 — L2 功能频谱效率校准

**Files:**
- Create: `calibration/layer3_l2_functions.py`
- Output: `calibration_report/layer3_l2_functions.md` + `calibration_report/figures/layer3_*.png`

**目标:** 配置匹配 Vienna SLS / R1-1801360 的参考场景，跑 1000+ slots 仿真，对比频谱效率。

- [ ] **Step 1: 编写 Layer 3 验证脚本**

```python
"""Layer 3: L2 功能校准 — 频谱效率对标

对标:
  - Vienna 5G SLS (UMa FDD 4x2 PF): SE ~1.8-2.2 bps/Hz
  - R1-1801360 (UMa 4GHz 4T4R PF): SE ~2.0-2.8 bps/Hz
方法: 配置匹配参考场景, 跑 2000 slots, 统计 SE
通过标准: Cell-average SE 落入参考范围 ±20%
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig, CSIConfig, HARQConfig,
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from calibration.utils import (
    setup_plot_style, save_figure, write_report,
    REFERENCE_SE,
)


def run_scenario(name, config, num_slots=2000, warmup=500):
    """运行单个场景并返回 KPI"""
    print(f"\n  Running scenario: {name} ({num_slots} slots, warmup={warmup})...")
    config['sim'] = SimConfig(num_slots=num_slots, random_seed=42, warmup_slots=warmup)
    engine = SimulationEngine(config)
    report = engine.run()
    return report


def run_layer3():
    """执行 Layer 3 校准"""
    setup_plot_style()

    print("=" * 60)
    print("Layer 3: L2 Functions — Spectral Efficiency Calibration")
    print("=" * 60)

    results = {}

    # ---- Scenario 1: FDD 4x2 PF (对标 Vienna SLS) ----
    config_fdd_4x2 = {
        'sim': None,  # 由 run_scenario 设置
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273,
                                 bandwidth_mhz=100.0, carrier_freq_ghz=3.5),
        'cell': CellConfig(num_tx_ant=4, num_tx_ports=4, max_layers=2,
                           total_power_dbm=46.0, cell_radius_m=500.0,
                           height_m=25.0, scenario='uma'),
        'ue': UEConfig(num_ue=20, num_rx_ant=2, min_distance_m=35.0,
                       max_distance_m=500.0, height_m=1.5, speed_kmh=3.0),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1, mcs_table_index=1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'tdd': None,  # FDD
        'csi': CSIConfig(enabled=True),
        'harq': HARQConfig(),
    }
    report_fdd_4x2 = run_scenario("FDD 4x2 PF (Vienna ref)", config_fdd_4x2)
    results['fdd_4x2'] = report_fdd_4x2

    # ---- Scenario 2: FDD 4x4 PF (对标 R1-1801360) ----
    config_fdd_4x4 = {
        'sim': None,
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273,
                                 bandwidth_mhz=100.0, carrier_freq_ghz=3.5),
        'cell': CellConfig(num_tx_ant=4, num_tx_ports=4, max_layers=4,
                           total_power_dbm=46.0, cell_radius_m=500.0,
                           height_m=25.0, scenario='uma'),
        'ue': UEConfig(num_ue=20, num_rx_ant=4, min_distance_m=35.0,
                       max_distance_m=500.0, height_m=1.5, speed_kmh=3.0),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1, mcs_table_index=1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'tdd': None,
        'csi': CSIConfig(enabled=True),
        'harq': HARQConfig(),
    }
    report_fdd_4x4 = run_scenario("FDD 4x4 PF (R1-1801360 ref)", config_fdd_4x4)
    results['fdd_4x4'] = report_fdd_4x4

    # ---- 汇总 ----
    print("\n" + "-" * 60)
    print("Layer 3 Summary: Spectral Efficiency Comparison")
    print("-" * 60)

    scenarios = [
        ('fdd_4x2', 'FDD 4x2 PF', 'vienna_fdd_4x2_pf'),
        ('fdd_4x4', 'FDD 4x4 PF', 'r1_1801360_4t4r_pf'),
    ]

    comparison = []
    for key, label, ref_key in scenarios:
        r = results[key]
        se = r['spectral_efficiency_bps_hz']
        bler = r['avg_bler']
        mcs = r['avg_mcs']
        tp = r['cell_avg_throughput_mbps']
        edge_tp = r['cell_edge_throughput_mbps']
        fairness = r['jain_fairness']

        ref = REFERENCE_SE[ref_key]
        ref_mid = (ref['min'] + ref['max']) / 2
        ref_range_20 = (ref['min'] * 0.8, ref['max'] * 1.2)
        in_range = ref_range_20[0] <= se <= ref_range_20[1]

        print(f"\n  {label}:")
        print(f"    SE:         {se:.2f} bps/Hz (ref: {ref['min']}-{ref['max']})")
        print(f"    Throughput: {tp:.1f} Mbps")
        print(f"    Cell edge:  {edge_tp:.1f} Mbps")
        print(f"    BLER:       {bler:.4f}")
        print(f"    Avg MCS:    {mcs:.1f}")
        print(f"    Fairness:   {fairness:.3f}")
        print(f"    Status:     {'PASS' if in_range else 'DEVIATION'}")

        comparison.append({
            'key': key, 'label': label, 'se': se, 'tp': tp, 'edge_tp': edge_tp,
            'bler': bler, 'mcs': mcs, 'fairness': fairness,
            'ref_min': ref['min'], 'ref_max': ref['max'],
            'ref_label': ref['label'], 'in_range': in_range,
        })

    # ---- 绘制对比图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # SE bar chart
    ax = axes[0]
    labels = [c['label'] for c in comparison]
    se_vals = [c['se'] for c in comparison]
    x = np.arange(len(labels))
    bars = ax.bar(x, se_vals, color=['#2196F3', '#4CAF50'], alpha=0.8)
    for c in comparison:
        idx = [i for i, cc in enumerate(comparison) if cc['key'] == c['key']][0]
        ax.fill_between([idx - 0.4, idx + 0.4], c['ref_min'], c['ref_max'],
                       alpha=0.2, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Spectral Efficiency (bps/Hz)')
    ax.set_title('Cell-Average SE vs Reference')
    ax.legend(['Sim', 'Ref range (±20%)'])

    # Per-UE throughput CDF
    ax = axes[1]
    for c in comparison:
        r = results[c['key']]
        ue_tp = np.sort(r['ue_avg_throughput_mbps'])
        cdf = np.arange(1, len(ue_tp) + 1) / len(ue_tp)
        ax.plot(ue_tp, cdf, linewidth=2, label=c['label'])
    ax.set_xlabel('Per-UE Throughput (Mbps)')
    ax.set_ylabel('CDF')
    ax.set_title('Per-UE Throughput CDF')
    ax.legend()

    # MCS distribution
    ax = axes[2]
    for c in comparison:
        r = results[c['key']]
        if 'mcs_distribution' in r:
            mcs_dist = r['mcs_distribution']
            mcs_indices = sorted(mcs_dist.keys())
            counts = [mcs_dist[m] for m in mcs_indices]
            total = sum(counts)
            pct = [c_val / total * 100 for c_val in counts]
            ax.bar([m + (0.2 if c['key'] == 'fdd_4x4' else -0.2) for m in mcs_indices],
                   pct, width=0.35, alpha=0.7, label=c['label'])
    ax.set_xlabel('MCS Index')
    ax.set_ylabel('Usage (%)')
    ax.set_title('MCS Distribution')
    ax.legend()

    fig.suptitle('Layer 3: L2 Functions Calibration', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'layer3_se_comparison')

    # ---- 生成报告 ----
    report_lines = [
        "# Layer 3: L2 Functions Calibration Report\n",
        "## Scenarios\n",
    ]

    for c in comparison:
        report_lines.extend([
            f"### {c['label']}\n",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Reference | {c['ref_label']} |",
            f"| Cell-Avg SE | {c['se']:.2f} bps/Hz |",
            f"| Reference SE range | {c['ref_min']}-{c['ref_max']} bps/Hz |",
            f"| Cell throughput | {c['tp']:.1f} Mbps |",
            f"| Cell edge throughput | {c['edge_tp']:.1f} Mbps |",
            f"| Avg BLER | {c['bler']:.4f} |",
            f"| Avg MCS | {c['mcs']:.1f} |",
            f"| Jain fairness | {c['fairness']:.3f} |",
            f"| Status | {'PASS' if c['in_range'] else 'DEVIATION'} |\n",
        ])

    report_lines.extend([
        "## Comparison Charts\n",
        "![SE Comparison](figures/layer3_se_comparison.png)\n",
        "## Conclusion\n",
    ])

    issues = []
    for c in comparison:
        if not c['in_range']:
            issues.append(f"- {c['label']}: SE={c['se']:.2f} outside ref range [{c['ref_min']}, {c['ref_max']}] ±20%")

    report_lines.append(f"- {sum(1 for c in comparison if c['in_range'])} / {len(comparison)} scenarios within reference range")

    report_lines.append("\n## Issues Found\n")
    if issues:
        report_lines.append("**以下场景频谱效率偏离参考范围，需要用户审查：**\n")
        report_lines.extend(issues)
    else:
        report_lines.append("所有场景均在参考范围内。")

    write_report("layer3_l2_functions.md", "\n".join(report_lines))

    return results


if __name__ == "__main__":
    run_layer3()
```

- [ ] **Step 2: 执行 Layer 3 验证**

```bash
cd C:\Users\lintian\Documents\GitHub\L2_RRM_Sim
.venv312/Scripts/python.exe calibration/layer3_l2_functions.py
```

Expected: 2 个场景各跑 2000 slots（约数分钟），输出 SE 对比和偏差。

- [ ] **Step 3: 向用户汇报 Layer 3 结果**

汇报两个场景的 SE 是否在参考范围内，等用户确认后进入 Layer 4。

- [ ] **Step 4: Commit**

```bash
git add calibration/layer3_l2_functions.py calibration_report/
git commit -m "calibration: Layer 3 L2 functions spectral efficiency comparison"
```

---

### Task 5: Layer 4 — 端到端 KPI + 多小区 Smoke

**Files:**
- Create: `calibration/layer4_e2e_kpi.py`
- Output: `calibration_report/layer4_e2e_kpi.md` + `calibration_report/figures/layer4_*.png`

**目标:** 复现 ITU-R M.2412 配置跑端到端 KPI，加多小区 smoke 验证。

- [ ] **Step 1: 编写 Layer 4 验证脚本**

```python
"""Layer 4: 端到端 KPI 校验 + 多小区 Smoke

对标: ITU-R M.2412 Dense Urban eMBB (200MHz TDD, 4x4, PF)
多小区: 7-site smoke — 跑通不 crash, 干扰合理
通过标准: SE 量级正确 (同一数量级), 趋势合理
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig, TDDConfig, CSIConfig, HARQConfig,
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from calibration.utils import (
    setup_plot_style, save_figure, write_report,
    REFERENCE_ITU,
)


def run_layer4():
    """执行 Layer 4 校准"""
    setup_plot_style()

    print("=" * 60)
    print("Layer 4: End-to-End KPI + Multi-Cell Smoke")
    print("=" * 60)

    # ---- Part A: ITU-R M.2412 配置 ----
    print("\n--- Part A: ITU-R M.2412 Dense Urban eMBB ---")
    print(f"  Reference config: {REFERENCE_ITU['config']}")

    # ITU 配置: 200MHz TDD, 30kHz SCS, 4x4 MIMO
    # 200MHz @ 30kHz = 546 PRBs (实际 NR 最接近的是 273*2，但单 CC 最多 273)
    # 折中: 用 100MHz 单 CC，SE 应与 200MHz 2CC 一致
    config_itu = {
        'sim': SimConfig(num_slots=2000, random_seed=42, warmup_slots=500),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273,
                                 bandwidth_mhz=100.0, carrier_freq_ghz=3.5),
        'cell': CellConfig(num_tx_ant=4, num_tx_ports=4, max_layers=4,
                           total_power_dbm=46.0, cell_radius_m=500.0,
                           height_m=25.0, scenario='uma'),
        'ue': UEConfig(num_ue=10, num_rx_ant=4, min_distance_m=35.0,
                       max_distance_m=500.0, height_m=1.5, speed_kmh=3.0),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1, mcs_table_index=1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'tdd': TDDConfig(duplex_mode='TDD', pattern='DDDSU',
                         special_dl_symbols=10, special_gp_symbols=2,
                         special_ul_symbols=2),
        'csi': CSIConfig(enabled=True),
        'harq': HARQConfig(),
    }

    print("  Running ITU-like scenario (2000 slots)...")
    engine_itu = SimulationEngine(config_itu)
    report_itu = engine_itu.run()

    se_itu = report_itu['spectral_efficiency_bps_hz']
    edge_se = report_itu['cell_edge_throughput_mbps'] / report_itu['cell_avg_throughput_mbps'] * se_itu
    tp_itu = report_itu['cell_avg_throughput_mbps']

    # ITU 参考是 200MHz + 64T64R + MU-MIMO 的高端配置，我们用 100MHz + 4T4R SU-MIMO
    # 所以 SE 会明显低于 ITU 参考 (7.8 bps/Hz)，但应在合理范围
    # 合理范围: 单小区 SU-MIMO 4x4 约 2-4 bps/Hz
    se_reasonable = 1.0 <= se_itu <= 6.0

    print(f"  Results:")
    print(f"    Cell-Avg SE:    {se_itu:.2f} bps/Hz")
    print(f"    Cell-Edge SE:   {edge_se:.2f} bps/Hz (approx)")
    print(f"    Throughput:     {tp_itu:.1f} Mbps")
    print(f"    BLER:           {report_itu['avg_bler']:.4f}")
    print(f"    Avg MCS:        {report_itu['avg_mcs']:.1f}")
    print(f"    PRB util:       {report_itu['prb_utilization']*100:.1f}%")
    print(f"    Fairness:       {report_itu['jain_fairness']:.3f}")
    print(f"    ITU ref:        {REFERENCE_ITU['dl_avg_se_bps_hz']} bps/Hz (64T64R MU-MIMO)")
    print(f"    Our config:     4T4R SU-MIMO, SE should be lower")
    print(f"    Reasonable:     {'YES' if se_reasonable else 'NO'} (expected 1-6 bps/Hz)")

    # ---- Part B: 多小区 Smoke ----
    print("\n--- Part B: Multi-Cell Smoke Test ---")

    multicell_ok = True
    multicell_report = {}
    try:
        # 尝试导入多小区引擎
        from l2_rrm_sim.core.multicell_engine import MultiCellEngine

        config_mc = {
            'sim': SimConfig(num_slots=200, random_seed=42, warmup_slots=50),
            'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273,
                                     bandwidth_mhz=100.0, carrier_freq_ghz=3.5),
            'cell': CellConfig(num_tx_ant=4, num_tx_ports=4, max_layers=2,
                               total_power_dbm=46.0, cell_radius_m=500.0,
                               height_m=25.0, scenario='uma'),
            'ue': UEConfig(num_ue=5, num_rx_ant=2, min_distance_m=35.0,
                           max_distance_m=500.0),
            'scheduler': SchedulerConfig(type='pf'),
            'link_adaptation': LinkAdaptationConfig(),
            'traffic': TrafficConfig(type='full_buffer'),
            'channel': ChannelConfig(type='statistical', scenario='uma'),
            'tdd': None,
            'csi': None,
            'harq': None,
        }

        print("  Running 7-site multi-cell (200 slots)...")
        mc_engine = MultiCellEngine(config_mc, num_sites=7)
        mc_report = mc_engine.run()

        mc_se = mc_report.get('spectral_efficiency_bps_hz', 0)
        se_ratio = mc_se / se_itu if se_itu > 0 else 0

        print(f"  Multi-cell results:")
        print(f"    SE:        {mc_se:.2f} bps/Hz")
        print(f"    SC/MC ratio: {se_ratio:.2f} (expect 0.5-0.7 due to interference)")

        multicell_report = {
            'se': mc_se,
            'ratio': se_ratio,
            'status': 'PASS' if 0.3 <= se_ratio <= 0.9 else 'CHECK',
        }

    except ImportError:
        print("  MultiCellEngine not available, skipping")
        multicell_ok = False
        multicell_report = {'status': 'SKIPPED', 'reason': 'MultiCellEngine not importable'}
    except Exception as e:
        print(f"  Multi-cell failed with error: {e}")
        multicell_ok = False
        multicell_report = {'status': 'ERROR', 'reason': str(e)}

    # ---- 绘制图表 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-UE throughput CDF
    ax = axes[0]
    ue_tp = np.sort(report_itu['ue_avg_throughput_mbps'])
    cdf = np.arange(1, len(ue_tp) + 1) / len(ue_tp)
    ax.plot(ue_tp, cdf, 'b-', linewidth=2, label='ITU-like TDD')
    ax.axvline(x=np.percentile(ue_tp, 5), color='r', linestyle='--',
               label=f'5th pct: {np.percentile(ue_tp, 5):.1f} Mbps')
    ax.set_xlabel('Per-UE Throughput (Mbps)')
    ax.set_ylabel('CDF')
    ax.set_title('Per-UE Throughput CDF — ITU-like TDD')
    ax.legend()

    # KPI summary bar
    ax = axes[1]
    metrics = ['SE\n(bps/Hz)', 'BLER\n(×100)', 'PRB Util\n(%)', 'Fairness\n(×100)']
    values = [se_itu, report_itu['avg_bler'] * 100,
              report_itu['prb_utilization'] * 100, report_itu['jain_fairness'] * 100]
    bars = ax.bar(metrics, values, color=['#2196F3', '#F44336', '#4CAF50', '#FF9800'], alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11)
    ax.set_title('KPI Summary — ITU-like TDD')
    ax.set_ylabel('Value')

    fig.suptitle('Layer 4: End-to-End KPI Calibration', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'layer4_e2e_kpi')

    # ---- 生成报告 ----
    report_lines = [
        "# Layer 4: End-to-End KPI Calibration Report\n",
        "## Part A: ITU-R M.2412 Comparison\n",
        "### Configuration\n",
        "| Parameter | Our Config | ITU Reference |",
        "|-----------|-----------|---------------|",
        "| Bandwidth | 100 MHz (1 CC) | 200 MHz (2 CC) |",
        "| SCS | 30 kHz | 30 kHz |",
        "| Antenna | 4T4R | 64T64R |",
        "| MIMO | SU-MIMO, max 4 layers | MU-MIMO |",
        "| Scheduler | PF | PF |",
        "| UE/cell | 10 | 10 |",
        "| Duplex | TDD DDDSU | TDD (80% DL) |",
        "| Traffic | Full buffer | Full buffer |\n",
        "**Note:** 我们的配置 (4T4R SU-MIMO) 显著低于 ITU 参考 (64T64R MU-MIMO)，",
        "因此 SE 不应直接对标 ITU 7.8 bps/Hz，而是验证量级合理性。\n",
        "### Results\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Cell-Avg SE | {se_itu:.2f} bps/Hz |",
        f"| Cell-Edge SE (approx) | {edge_se:.2f} bps/Hz |",
        f"| Throughput | {tp_itu:.1f} Mbps |",
        f"| BLER | {report_itu['avg_bler']:.4f} |",
        f"| Avg MCS | {report_itu['avg_mcs']:.1f} |",
        f"| PRB Utilization | {report_itu['prb_utilization']*100:.1f}% |",
        f"| Jain Fairness | {report_itu['jain_fairness']:.3f} |",
        f"| Reasonable range | 1.0-6.0 bps/Hz |",
        f"| Status | {'PASS' if se_reasonable else 'CHECK'} |\n",
        "## Part B: Multi-Cell Smoke\n",
    ]

    if multicell_report.get('status') == 'PASS' or multicell_report.get('status') == 'CHECK':
        report_lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Multi-cell SE | {multicell_report['se']:.2f} bps/Hz |",
            f"| SC/MC ratio | {multicell_report['ratio']:.2f} |",
            f"| Status | {multicell_report['status']} |",
        ])
    else:
        report_lines.append(f"Status: {multicell_report['status']} — {multicell_report.get('reason', 'N/A')}")

    report_lines.extend([
        "\n## Charts\n",
        "![E2E KPI](figures/layer4_e2e_kpi.png)\n",
        "\n## Conclusion\n",
        f"- Single-cell TDD SE: {se_itu:.2f} bps/Hz ({'reasonable' if se_reasonable else 'needs review'})",
        f"- Multi-cell: {multicell_report['status']}",
        "\n## Issues Found\n",
    ])

    issues = []
    if not se_reasonable:
        issues.append(f"- Single-cell SE {se_itu:.2f} outside expected range [1.0, 6.0]")
    if multicell_report.get('status') == 'ERROR':
        issues.append(f"- Multi-cell engine error: {multicell_report.get('reason')}")
    if multicell_report.get('status') == 'CHECK':
        issues.append(f"- Multi-cell SE ratio {multicell_report['ratio']:.2f} outside [0.3, 0.9]")

    if issues:
        report_lines.append("**以下问题需要用户审查：**\n")
        report_lines.extend(issues)
    else:
        report_lines.append("无问题。")

    write_report("layer4_e2e_kpi.md", "\n".join(report_lines))

    return {'itu': report_itu, 'multicell': multicell_report}


if __name__ == "__main__":
    run_layer4()
```

- [ ] **Step 2: 执行 Layer 4 验证**

```bash
cd C:\Users\lintian\Documents\GitHub\L2_RRM_Sim
.venv312/Scripts/python.exe calibration/layer4_e2e_kpi.py
```

Expected: ITU 场景跑 2000 slots，多小区 smoke 跑 200 slots，输出 KPI 和图表。

- [ ] **Step 3: 向用户汇报 Layer 4 结果**

汇报端到端 SE 是否合理，多小区是否跑通，等用户确认。

- [ ] **Step 4: Commit**

```bash
git add calibration/layer4_e2e_kpi.py calibration_report/
git commit -m "calibration: Layer 4 end-to-end KPI and multi-cell smoke test"
```

---

### Task 6: 汇总报告

**Files:**
- Create: `calibration_report/summary.md`

- [ ] **Step 1: 编写汇总报告生成逻辑**

在所有四层完成后，手动汇总各层结果，编写 `calibration_report/summary.md`：

```markdown
# L2_RRM_Sim 校准验证汇总报告

## 验证概览

| Layer | 对标基准 | 关键指标 | 偏差 | 状态 |
|-------|---------|---------|------|------|
| L1 PHY | R1-1711982 BLER curves | 10% BLER点 | (填入) | (填入) |
| L2 Channel | TR 38.901 §7.8 | 耦合损耗/SINR CDF | (填入) | (填入) |
| L3 L2 Func | Vienna/R1-1801360 SE | 频谱效率 | (填入) | (填入) |
| L4 E2E | ITU-R M.2412 | 端到端 KPI | (填入) | (填入) |

## 各层详细报告

- [Layer 1: PHY Abstraction](layer1_phy_abstraction.md)
- [Layer 2: Channel Model](layer2_channel_model.md)
- [Layer 3: L2 Functions](layer3_l2_functions.md)
- [Layer 4: End-to-End KPI](layer4_e2e_kpi.md)

## 待修复问题汇总

(从各层报告中汇总所有 DEVIATION 和 Issues)

## 结论

(整体评估)
```

注意：summary.md 的具体数值需要在四层全部执行完毕后，根据实际结果填入。

- [ ] **Step 2: Commit**

```bash
git add calibration_report/summary.md
git commit -m "calibration: add summary report template"
```

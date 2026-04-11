"""Layer 3 校准: L2 功能频谱效率 — TDD Massive MIMO (LegacyPHY)

使用 LegacyPHY (EESM + OLLA + BLER 查找表) 路径验证。

Scenario 1: TDD 64T4R SU-MIMO (单小区, 无 ICI)
  - TDD DDDSU, 100MHz, 30kHz SCS, 273 PRB, 3.5GHz
  - 64 TX ant, 4 ports, max 4 layers, 20 UE, 4 RX ant

Scenario 2: TDD 64T2R SU-MIMO (降 RX 对比)
  - 同上但 2 RX ant, max 2 layers

Pass criteria: SE 落入预期范围, BLER 在 5%-15%, PRB util > 70%
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calibration.utils import setup_plot_style, save_figure, write_report

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig, CSIConfig, HARQConfig, TDDConfig,
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from l2_rrm_sim.link_adaptation.legacy_phy_adapter import LegacyPHYAdapter

# ---- Configuration ----
NUM_SLOTS = 2000
WARMUP_SLOTS = 500
SEED = 42

SCENARIOS = [
    {
        'name': 'TDD 64T4R 4-layer',
        'num_tx_ant': 64,
        'num_tx_ports': 4,
        'max_layers': 4,
        'num_rx_ant': 4,
        'num_ue': 20,
        'se_range': (2.0, 6.0),  # 单小区无ICI SU-MIMO 预期
    },
    {
        'name': 'TDD 64T2R 2-layer',
        'num_tx_ant': 64,
        'num_tx_ports': 4,
        'max_layers': 2,
        'num_rx_ant': 2,
        'num_ue': 20,
        'se_range': (1.5, 4.0),
    },
]


def make_config(s: dict) -> dict:
    return {
        'sim': SimConfig(num_slots=NUM_SLOTS, random_seed=SEED, warmup_slots=WARMUP_SLOTS),
        'carrier': CarrierConfig(
            subcarrier_spacing=30, num_prb=273,
            bandwidth_mhz=100.0, carrier_freq_ghz=3.5,
        ),
        'cell': CellConfig(
            num_tx_ant=s['num_tx_ant'],
            num_tx_ports=s['num_tx_ports'],
            max_layers=s['max_layers'],
            total_power_dbm=46.0,
            cell_radius_m=500.0,
            height_m=25.0,
            scenario='uma',
        ),
        'ue': UEConfig(
            num_ue=s['num_ue'],
            num_rx_ant=s['num_rx_ant'],
            min_distance_m=35.0,
            max_distance_m=500.0,
            height_m=1.5,
            speed_kmh=3.0,
        ),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1, mcs_table_index=1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'csi': CSIConfig(enabled=False),  # CSI bug 未修, 记录在报告中
        'harq': HARQConfig(),
        'tdd': TDDConfig(
            duplex_mode='TDD',
            pattern='DDDSU',
            special_dl_symbols=10,
            special_gp_symbols=2,
            special_ul_symbols=2,
        ),
    }


def run_scenario(s: dict, idx: int) -> dict:
    se_lo, se_hi = s['se_range']
    print(f"\n{'='*60}")
    print(f"Scenario {idx}: {s['name']}")
    print(f"  {s['num_tx_ant']}T{s['num_rx_ant']}R, max {s['max_layers']} layers, "
          f"TDD DDDSU, {s['num_ue']} UEs")
    print(f"  Expected SE range: [{se_lo}, {se_hi}] bps/Hz")
    print(f"{'='*60}")

    config = make_config(s)
    t0 = time.time()
    engine = SimulationEngine(config)

    # 强制使用 LegacyPHY（我们自己写的 EESM+OLLA 路径）
    if engine._use_sionna_phy:
        engine.phy = LegacyPHYAdapter(
            num_ue=s['num_ue'],
            bler_target=0.1,
            delta_up=0.5,
            offset_min=-10.0,
            offset_max=10.0,
            mcs_table_index=1,
            num_re_per_prb=engine.resource_grid.num_re_per_prb,
            rng=engine.rng.phy,
        )
        engine._use_sionna_phy = False
        engine.sionna_phy = None

    report = engine.run()
    elapsed = time.time() - t0

    se = report['spectral_efficiency_bps_hz']
    tput = report['cell_avg_throughput_mbps']
    edge_tput = report['cell_edge_throughput_mbps']
    bler = report['avg_bler']
    mcs = report['avg_mcs']
    fairness = report['jain_fairness']
    prb_util = report['prb_utilization']
    avg_rank = report.get('avg_rank', 0)

    se_ok = se_lo <= se <= se_hi
    bler_ok = 0.05 <= bler <= 0.15
    util_ok = prb_util > 0.70

    passed = se_ok and bler_ok and util_ok

    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Spectral Efficiency: {se:.3f} bps/Hz  "
          f"[{se_lo}, {se_hi}] {'OK' if se_ok else 'OUT OF RANGE'}")
    print(f"    Cell throughput:     {tput:.1f} Mbps")
    print(f"    Cell edge (5%):      {edge_tput:.1f} Mbps")
    print(f"    Avg BLER:            {bler:.4f}  "
          f"[0.05, 0.15] {'OK' if bler_ok else 'OUT OF RANGE'}")
    print(f"    Avg MCS:             {mcs:.1f}")
    print(f"    Avg Rank:            {avg_rank:.2f}")
    print(f"    Jain fairness:       {fairness:.4f}")
    print(f"    PRB utilization:     {prb_util:.1%}  "
          f">70% {'OK' if util_ok else 'LOW'}")
    print(f"    Status:              {'PASS' if passed else 'FAIL'}")

    return {
        'name': s['name'], 'idx': idx, 'config': s,
        'se': se, 'tput': tput, 'edge_tput': edge_tput,
        'bler': bler, 'mcs': mcs, 'avg_rank': avg_rank,
        'fairness': fairness, 'prb_util': prb_util,
        'se_range': s['se_range'], 'se_ok': se_ok,
        'bler_ok': bler_ok, 'util_ok': util_ok, 'passed': passed,
        'elapsed': elapsed,
        'ue_tput': report['ue_avg_throughput_mbps'],
        'mcs_dist': report.get('mcs_distribution', {}),
    }


def plot_results(results: list):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: SE bar chart
    ax = axes[0]
    names = [r['name'] for r in results]
    se_vals = [r['se'] for r in results]
    colors = ['#2ecc71' if r['se_ok'] else '#e74c3c' for r in results]
    bars = ax.bar(names, se_vals, color=colors, alpha=0.8, width=0.5)
    for i, r in enumerate(results):
        lo, hi = r['se_range']
        ax.plot([i - 0.25, i + 0.25], [lo, lo], 'b--', linewidth=2)
        ax.plot([i - 0.25, i + 0.25], [hi, hi], 'b--', linewidth=2)
        ax.fill_between([i - 0.25, i + 0.25], lo, hi, color='blue', alpha=0.1)
        ax.text(i, se_vals[i] + 0.05, f"{se_vals[i]:.2f}",
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Spectral Efficiency (bps/Hz)')
    ax.set_title('Cell-Average SE — TDD Massive MIMO')
    ax.set_ylim(0, max(se_vals) * 1.3 + 0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Per-UE throughput CDF
    ax = axes[1]
    for r in results:
        tput_sorted = np.sort(r['ue_tput'])
        cdf = np.arange(1, len(tput_sorted) + 1) / len(tput_sorted)
        ax.plot(tput_sorted, cdf, linewidth=2, label=r['name'])
    ax.set_xlabel('Per-UE Throughput (Mbps)')
    ax.set_ylabel('CDF')
    ax.set_title('Per-UE Throughput CDF')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel 3: MCS distribution
    ax = axes[2]
    width = 0.35
    for i, r in enumerate(results):
        if not r['mcs_dist']:
            continue
        mcs_keys = sorted(r['mcs_dist'].keys())
        mcs_counts = [r['mcs_dist'][k] for k in mcs_keys]
        total = sum(mcs_counts)
        if total == 0:
            continue
        mcs_pct = [100.0 * c / total for c in mcs_counts]
        offset = (i - 0.5) * width
        ax.bar([k + offset for k in mcs_keys], mcs_pct,
               width=width, alpha=0.7, label=r['name'])
    ax.set_xlabel('MCS Index')
    ax.set_ylabel('Usage (%)')
    ax.set_title('MCS Distribution')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Layer 3: TDD Massive MIMO L2 Calibration', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'layer3_se_comparison')


def generate_report(results: list):
    all_pass = all(r['passed'] for r in results)

    sections = []
    for r in results:
        lo, hi = r['se_range']
        s = r['config']
        sections.append(f"""### Scenario {r['idx']}: {r['name']}

| Parameter | Value |
|-----------|-------|
| TX antennas | {s['num_tx_ant']} |
| TX ports | {s['num_tx_ports']} |
| Max layers | {s['max_layers']} |
| RX antennas | {s['num_rx_ant']} |
| UEs | {s['num_ue']} |
| Duplex | TDD DDDSU |
| DL ratio | ~80% (3D + 0.7S out of 5 slots) |

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| Spectral Efficiency | {r['se']:.3f} bps/Hz | [{lo}, {hi}] | {'OK' if r['se_ok'] else 'OUT'} |
| Cell throughput | {r['tput']:.1f} Mbps | — | — |
| Cell edge (5%) | {r['edge_tput']:.1f} Mbps | — | — |
| Avg BLER | {r['bler']:.4f} | [0.05, 0.15] | {'OK' if r['bler_ok'] else 'OUT'} |
| Avg MCS | {r['mcs']:.1f} | — | — |
| Avg Rank | {r['avg_rank']:.2f} | — | — |
| Jain fairness | {r['fairness']:.4f} | — | — |
| PRB utilization | {r['prb_util']:.1%} | >70% | {'OK' if r['util_ok'] else 'LOW'} |
| **Overall** | | | **{'PASS' if r['passed'] else 'FAIL'}** |
""")

    content = f"""# Layer 3: L2 Functions — TDD Massive MIMO Calibration

## Overview

Verify the L2 stack (PF scheduler, link adaptation, HARQ) spectral efficiency
in TDD Massive MIMO configuration (64T, DDDSU pattern).

## Common Configuration

| Parameter | Value |
|-----------|-------|
| Duplex | TDD DDDSU |
| Bandwidth | 100 MHz |
| SCS | 30 kHz |
| PRBs | 273 |
| Carrier freq | 3.5 GHz |
| TX antennas | 64 |
| BS power | 46 dBm |
| Traffic | Full buffer |
| Channel | Statistical UMa |
| Slots | {NUM_SLOTS} (warmup: {WARMUP_SLOTS}) |
| Scheduler | PF (beta=0.98) |
| BLER target | 0.1 |

## PHY Backend

**LegacyPHY (EESM + OLLA + BLER lookup)** — 自研链路自适应路径。
SionnaPHY 存在 MCS 严重欠选问题 (MCS ~7 @ SINR 24 dB)，不用于校准。

## Known Deviations

1. **CSI feedback disabled**: 避免 Sionna tensor 兼容性问题。
2. **Single-cell, no ICI**: SE is higher than multi-cell deployment.
3. **SU-MIMO only**: No MU-MIMO pairing, max layers limited by min(TX ports, RX ant).

## Results

{''.join(sections)}

## Figures

![SE Comparison](figures/layer3_se_comparison.png)

## Conclusion

{'**ALL PASS** — SE, BLER, PRB utilization all within expected ranges.' if all_pass
 else '**SOME METRICS OUT OF RANGE** — see individual scenario results.'}
"""
    write_report('layer3_l2_functions.md', content)


def main():
    print("=" * 60)
    print("Layer 3: TDD Massive MIMO L2 Calibration")
    print("=" * 60)

    setup_plot_style()

    results = []
    for idx, s in enumerate(SCENARIOS, 1):
        r = run_scenario(s, idx)
        results.append(r)

    plot_results(results)
    generate_report(results)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        lo, hi = r['se_range']
        print(f"  {r['name']}: SE={r['se']:.3f} bps/Hz "
              f"(expect [{lo}, {hi}])  "
              f"BLER={r['bler']:.4f}  "
              f"PRB={r['prb_util']:.0%}  "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    all_pass = all(r['passed'] for r in results)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    main()

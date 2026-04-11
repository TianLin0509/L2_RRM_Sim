"""Layer 3 校准: L2 功能频谱效率 — 对标 Vienna SLS / R1-1801360

Scenario 1: FDD 4x2 PF (Vienna SLS 参考)
  - FDD, 100MHz, 30kHz SCS, 273 PRB, 3.5GHz
  - 4 TX ant, 4 ports, max 2 layers, 20 UE, 2 RX ant
  - PF scheduler (beta=0.98), BLER target 0.1, MCS table 1
  - Full buffer, statistical channel (UMa), HARQ enabled
  - Reference SE: 1.8-2.2 bps/Hz

Scenario 2: FDD 4x4 PF (R1-1801360 参考)
  - Same as above but: max 4 layers, 4 RX ant
  - Reference SE: 2.0-2.8 bps/Hz

Pass criteria: Cell-average SE within reference range +/-20%
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

# Force CPU — Sionna PHYAbstraction does not support cuda:0 device string
import torch
torch.cuda.is_available = lambda: False

from calibration.utils import (
    setup_plot_style, save_figure, write_report, REFERENCE_SE,
)
from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig, CSIConfig, HARQConfig, TDDConfig,
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine

# ---- Configuration ----
NUM_SLOTS = 2000
WARMUP_SLOTS = 500
SEED = 42


def make_config(scenario: int) -> dict:
    """Build config dict for scenario 1 (4x2) or 2 (4x4)."""
    max_layers = 2 if scenario == 1 else 4
    num_rx_ant = 2 if scenario == 1 else 4

    return {
        'sim': SimConfig(
            num_slots=NUM_SLOTS,
            random_seed=SEED,
            warmup_slots=WARMUP_SLOTS,
        ),
        'carrier': CarrierConfig(
            subcarrier_spacing=30,
            num_prb=273,
            bandwidth_mhz=100.0,
            carrier_freq_ghz=3.5,
        ),
        'cell': CellConfig(
            num_tx_ant=4,
            num_tx_ports=4,
            max_layers=max_layers,
            total_power_dbm=46.0,
            cell_radius_m=500.0,
            height_m=25.0,
            scenario='uma',
        ),
        'ue': UEConfig(
            num_ue=20,
            num_rx_ant=num_rx_ant,
            min_distance_m=35.0,
            max_distance_m=500.0,
            height_m=1.5,
            speed_kmh=3.0,
        ),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(
            bler_target=0.1,
            mcs_table_index=1,
        ),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        # NOTE: CSI disabled due to existing bug (Sionna OLLA _offset tensor
        # incompatible with sinr_to_cqi scalar interface). Deviation recorded.
        'csi': CSIConfig(enabled=False),
        'harq': HARQConfig(),
        'tdd': TDDConfig(duplex_mode='FDD'),
    }


def run_scenario(name: str, scenario: int, ref_key: str) -> dict:
    """Run one simulation scenario and return results dict."""
    ref = REFERENCE_SE[ref_key]
    print(f"\n{'='*60}")
    print(f"Scenario {scenario}: {name}")
    print(f"  Reference: {ref['label']}  SE = [{ref['min']}, {ref['max']}] bps/Hz")
    print(f"{'='*60}")

    config = make_config(scenario)
    t0 = time.time()
    engine = SimulationEngine(config)
    report = engine.run()
    elapsed = time.time() - t0

    se = report['spectral_efficiency_bps_hz']
    tput = report['cell_avg_throughput_mbps']
    bler = report['avg_bler']
    mcs = report['avg_mcs']
    fairness = report['jain_fairness']
    prb_util = report['prb_utilization']

    # Pass criteria: within reference range +/-20%
    se_min = ref['min'] * 0.8
    se_max = ref['max'] * 1.2
    passed = se_min <= se <= se_max

    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Spectral Efficiency: {se:.3f} bps/Hz")
    print(f"    Reference range:     [{ref['min']}, {ref['max']}] bps/Hz")
    print(f"    Acceptable range:    [{se_min:.2f}, {se_max:.2f}] bps/Hz (+/-20%)")
    print(f"    Cell throughput:     {tput:.1f} Mbps")
    print(f"    Avg BLER:            {bler:.4f}")
    print(f"    Avg MCS:             {mcs:.1f}")
    print(f"    Jain fairness:       {fairness:.4f}")
    print(f"    PRB utilization:     {prb_util:.1%}")
    print(f"    Status:              {'PASS' if passed else 'FAIL'}")

    return {
        'name': name,
        'scenario': scenario,
        'ref_key': ref_key,
        'ref': ref,
        'se': se,
        'tput': tput,
        'bler': bler,
        'mcs': mcs,
        'fairness': fairness,
        'prb_util': prb_util,
        'se_min': se_min,
        'se_max': se_max,
        'passed': passed,
        'elapsed': elapsed,
        'ue_tput': report['ue_avg_throughput_mbps'],
        'mcs_dist': report['mcs_distribution'],
        'report': report,
    }


def plot_results(results: list):
    """Generate bar chart + per-UE throughput CDF + MCS distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: SE comparison bar chart ---
    ax = axes[0]
    names = [r['name'] for r in results]
    se_vals = [r['se'] for r in results]
    colors = ['#2ecc71' if r['passed'] else '#e74c3c' for r in results]

    bars = ax.bar(names, se_vals, color=colors, alpha=0.8, width=0.5)
    for i, r in enumerate(results):
        ax.plot([i - 0.25, i + 0.25], [r['ref']['min'], r['ref']['min']],
                'b--', linewidth=2)
        ax.plot([i - 0.25, i + 0.25], [r['ref']['max'], r['ref']['max']],
                'b--', linewidth=2)
        ax.fill_between([i - 0.25, i + 0.25],
                        r['ref']['min'], r['ref']['max'],
                        color='blue', alpha=0.1)
        ax.text(i, se_vals[i] + 0.05, f"{se_vals[i]:.2f}",
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Spectral Efficiency (bps/Hz)')
    ax.set_title('Cell-Average SE vs Reference')
    ax.legend(['Reference range'], loc='upper right')
    ax.set_ylim(0, max(se_vals) * 1.5 + 0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 2: Per-UE throughput CDF ---
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

    # --- Panel 3: MCS distribution ---
    ax = axes[2]
    width = 0.35
    for i, r in enumerate(results):
        mcs_keys = sorted(r['mcs_dist'].keys())
        mcs_counts = [r['mcs_dist'][k] for k in mcs_keys]
        total = sum(mcs_counts)
        mcs_pct = [100.0 * c / total for c in mcs_counts]
        offset = (i - 0.5) * width
        ax.bar([k + offset for k in mcs_keys], mcs_pct,
               width=width, alpha=0.7, label=r['name'])
    ax.set_xlabel('MCS Index')
    ax.set_ylabel('Usage (%)')
    ax.set_title('MCS Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_figure(fig, 'layer3_se_comparison')


def generate_report(results: list):
    """Generate markdown report."""
    all_pass = all(r['passed'] for r in results)

    scenario_sections = []
    for r in results:
        section = f"""### Scenario {r['scenario']}: {r['name']}

| Parameter | Value |
|-----------|-------|
| TX antennas | 4 |
| TX ports | 4 |
| Max layers | {'2' if r['scenario'] == 1 else '4'} |
| RX antennas | {'2' if r['scenario'] == 1 else '4'} |
| UEs | 20 |
| Scheduler | PF (beta=0.98) |
| BLER target | 0.1 |
| MCS table | 1 |
| Channel | Statistical UMa |
| Slots | {NUM_SLOTS} (warmup: {WARMUP_SLOTS}) |

| Metric | Value |
|--------|-------|
| Spectral Efficiency | {r['se']:.3f} bps/Hz |
| Reference range | [{r['ref']['min']}, {r['ref']['max']}] bps/Hz |
| Acceptable range (+/-20%) | [{r['se_min']:.2f}, {r['se_max']:.2f}] bps/Hz |
| Cell throughput | {r['tput']:.1f} Mbps |
| Avg BLER | {r['bler']:.4f} |
| Avg MCS | {r['mcs']:.1f} |
| Jain fairness | {r['fairness']:.4f} |
| PRB utilization | {r['prb_util']:.1%} |
| Elapsed time | {r['elapsed']:.1f}s |
| **Status** | **{'PASS' if r['passed'] else 'FAIL'}** |
"""
        scenario_sections.append(section)

    content = f"""# Layer 3: L2 Functions Spectral Efficiency Calibration

## Overview

Verify that the complete L2 stack (PF scheduler, link adaptation, HARQ)
produces reasonable spectral efficiency compared to published references.

## Common Configuration

| Parameter | Value |
|-----------|-------|
| Duplex | FDD |
| Bandwidth | 100 MHz |
| SCS | 30 kHz |
| PRBs | 273 |
| Carrier freq | 3.5 GHz |
| Traffic | Full buffer |
| Channel | Statistical UMa |
| BS height | 25 m |
| UE height | 1.5 m |
| UE distance | [35, 500] m |
| Cell radius | 500 m |
| BS power | 46 dBm |

## Deviations

1. **CSI feedback disabled**: Sionna OLLA `_offset` (torch.Tensor) is incompatible
   with `sinr_to_cqi` (expects scalar float). The engine code at
   `simulation_engine.py:473` subtracts a torch tensor from a numpy scalar,
   producing a tensor that downstream `sinr_to_cqi` cannot handle.
   CSI was disabled to allow simulation to complete. This may affect SE
   (typically CSI improves MCS selection accuracy).

2. **CPU-only execution**: Sionna PHYAbstraction rejects `cuda:0` device string.
   Forced CPU via `torch.cuda.is_available` monkey-patch.

3. **No inter-cell interference (ICI)**: Single-cell simulation without neighboring
   cells. Reference values (Vienna SLS, R1-1801360) include 7-site 21-cell
   deployment with full ICI. Missing ICI significantly inflates SINR and SE.
   This is the primary cause of SE exceeding reference ranges.

## Results

{''.join(scenario_sections)}

## Figures

![SE Comparison](figures/layer3_se_comparison.png)

## Conclusion

Pass criterion: Cell-average SE within reference range +/-20%

**{'ALL PASS' if all_pass else 'SOME SCENARIOS FAIL'}**
"""
    write_report('layer3_l2_functions.md', content)


def main():
    print("=" * 60)
    print("Layer 3: L2 Functions Spectral Efficiency Calibration")
    print("=" * 60)

    setup_plot_style()

    results = []

    # Scenario 1: FDD 4x2 PF (Vienna SLS)
    r1 = run_scenario(
        name='FDD 4x2 PF',
        scenario=1,
        ref_key='vienna_fdd_4x2_pf',
    )
    results.append(r1)

    # Scenario 2: FDD 4x4 PF (R1-1801360)
    r2 = run_scenario(
        name='FDD 4x4 PF',
        scenario=2,
        ref_key='r1_1801360_4t4r_pf',
    )
    results.append(r2)

    # Plot
    plot_results(results)

    # Report
    generate_report(results)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f"  {r['name']}: SE={r['se']:.3f} bps/Hz "
              f"(ref [{r['ref']['min']}, {r['ref']['max']}], "
              f"accept [{r['se_min']:.2f}, {r['se_max']:.2f}])  {status}")
    all_pass = all(r['passed'] for r in results)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    main()

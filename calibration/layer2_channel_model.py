"""Layer 2 校准: 信道模型 — 耦合损耗 CDF + 几何 SINR CDF

Part A: 独立验证路径损耗模型
  - 2000 UE 均匀撒在 UMa 小区 (r=[35m, 500m])
  - 计算 LOS 概率、路径损耗、阴影衰落 → 耦合损耗 CDF
  - 参考中位数: 110 dB (TR 38.901 §7.8)

Part B: 通过 SimulationEngine 验证几何 SINR
  - 200 UE, UMa 3.5GHz, 4Tx/2Rx, FDD, statistical channel
  - 50 slots, 收集宽带 SINR CDF
  - 参考中位数: 6.0 dB (TR 38.901 §7.8)

Pass criteria: CDF 中位数偏差 <= 2 dB
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calibration.utils import (
    setup_plot_style, save_figure, write_report, REFERENCE_CHANNEL_UMA,
)
from l2_rrm_sim.channel.pathloss_models import (
    compute_pathloss_uma, compute_los_probability_uma,
)

# ---- 配置 ----
NUM_UE_PARTA = 2000
R_MIN = 35.0    # m
R_MAX = 500.0   # m
H_BS = 25.0     # m
H_UT = 1.5      # m
FC_GHZ = 3.5
SHADOW_STD_LOS = 4.0   # dB
SHADOW_STD_NLOS = 6.0  # dB
SEED = 42

# Part B
NUM_UE_PARTB = 200
NUM_SLOTS_PARTB = 50


def run_part_a(rng: np.random.Generator):
    """Part A: 路径损耗模型独立验证"""
    print("\n--- Part A: Pathloss Model Independent Verification ---")
    print(f"  UEs: {NUM_UE_PARTA}, r=[{R_MIN}, {R_MAX}]m, fc={FC_GHZ}GHz")

    # 均匀撒点 (annulus): r = sqrt(U * (R_max^2 - R_min^2) + R_min^2)
    u = rng.uniform(0, 1, NUM_UE_PARTA)
    distances = np.sqrt(u * (R_MAX**2 - R_MIN**2) + R_MIN**2)
    angles = rng.uniform(0, 2 * np.pi, NUM_UE_PARTA)

    pathloss_db = np.zeros(NUM_UE_PARTA)
    shadow_db = np.zeros(NUM_UE_PARTA)
    is_los_arr = np.zeros(NUM_UE_PARTA, dtype=bool)

    for i in range(NUM_UE_PARTA):
        d = distances[i]
        p_los = compute_los_probability_uma(d, H_UT)
        is_los = rng.random() < p_los
        is_los_arr[i] = is_los

        pl = compute_pathloss_uma(d, H_BS, H_UT, FC_GHZ, is_los)
        pathloss_db[i] = pl

        sf_std = SHADOW_STD_LOS if is_los else SHADOW_STD_NLOS
        shadow_db[i] = rng.normal(0, sf_std)

    coupling_loss_db = pathloss_db + shadow_db

    median_cl = float(np.median(coupling_loss_db))
    ref_cl = REFERENCE_CHANNEL_UMA['coupling_loss_median_db']
    dev_cl = median_cl - ref_cl

    print(f"  LOS ratio: {is_los_arr.sum()}/{NUM_UE_PARTA} ({100*is_los_arr.mean():.1f}%)")
    print(f"  Coupling loss median: {median_cl:.1f} dB  (ref: {ref_cl:.1f} dB)")
    print(f"  Deviation: {dev_cl:+.1f} dB")
    print(f"  Status: {'PASS' if abs(dev_cl) <= 2.0 else 'FAIL'}")

    return {
        'distances': distances,
        'pathloss_db': pathloss_db,
        'shadow_db': shadow_db,
        'coupling_loss_db': coupling_loss_db,
        'is_los': is_los_arr,
        'median_cl': median_cl,
        'ref_cl': ref_cl,
        'dev_cl': dev_cl,
    }


def run_part_b():
    """Part B: SimulationEngine 几何 SINR 验证"""
    print("\n--- Part B: Geometry SINR via SimulationEngine ---")
    print(f"  UEs: {NUM_UE_PARTB}, slots: {NUM_SLOTS_PARTB}")

    from l2_rrm_sim.config.sim_config import (
        SimConfig, CarrierConfig, CellConfig, UEConfig,
        SchedulerConfig, LinkAdaptationConfig, TrafficConfig, ChannelConfig,
        TDDConfig, CSIConfig, HARQConfig,
    )
    from l2_rrm_sim.core.simulation_engine import SimulationEngine

    config = {
        'sim': SimConfig(num_slots=NUM_SLOTS_PARTB, random_seed=SEED, warmup_slots=0),
        'carrier': CarrierConfig(
            subcarrier_spacing=30,
            num_prb=273,
            bandwidth_mhz=100.0,
            carrier_freq_ghz=FC_GHZ,
        ),
        'cell': CellConfig(
            num_tx_ant=4,
            num_tx_ports=4,
            max_layers=2,
            total_power_dbm=46.0,
            cell_radius_m=R_MAX,
            height_m=H_BS,
            scenario='uma',
        ),
        'ue': UEConfig(
            num_ue=NUM_UE_PARTB,
            num_rx_ant=2,
            min_distance_m=R_MIN,
            max_distance_m=R_MAX,
            height_m=H_UT,
            speed_kmh=3.0,
        ),
        'scheduler': SchedulerConfig(type='pf'),
        'link_adaptation': LinkAdaptationConfig(),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(
            type='statistical',
            scenario='uma',
            shadow_fading_std_db=SHADOW_STD_LOS,
        ),
        'tdd': TDDConfig(duplex_mode='FDD'),
        'csi': CSIConfig(enabled=False),
        'harq': HARQConfig(),
    }

    engine = SimulationEngine(config)
    print(f"  Engine initialized, {engine.num_ue} UEs")

    # 收集每 slot 每 UE 的宽带 SINR
    sinr_all = []  # (num_slots, num_ue)

    for s in range(NUM_SLOTS_PARTB):
        slot_ctx = engine._make_slot_context(s)
        channel_state = engine.channel.update(slot_ctx, engine.ue_states)
        # sinr_per_prb: (num_ue, max_layers, num_prb) linear
        sinr_linear = channel_state.sinr_per_prb[:, 0, :]  # layer 0
        wb_sinr_db = 10.0 * np.log10(np.mean(sinr_linear, axis=1) + 1e-30)
        sinr_all.append(wb_sinr_db)

    sinr_all = np.array(sinr_all)  # (num_slots, num_ue)
    # 平均每 UE 跨 slot
    sinr_per_ue = np.mean(sinr_all, axis=0)  # (num_ue,)

    median_sinr = float(np.median(sinr_per_ue))
    ref_sinr = REFERENCE_CHANNEL_UMA['geometry_sinr_median_db']
    dev_sinr = median_sinr - ref_sinr

    print(f"  SINR median: {median_sinr:.1f} dB  (ref: {ref_sinr:.1f} dB)")
    print(f"  Deviation: {dev_sinr:+.1f} dB")
    print(f"  Status: {'PASS' if abs(dev_sinr) <= 2.0 else 'FAIL'}")

    return {
        'sinr_per_ue': sinr_per_ue,
        'sinr_all': sinr_all,
        'median_sinr': median_sinr,
        'ref_sinr': ref_sinr,
        'dev_sinr': dev_sinr,
    }


def plot_cdf(result_a, result_b):
    """并排绘制耦合损耗 CDF + SINR CDF"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Coupling Loss CDF ---
    cl = np.sort(result_a['coupling_loss_db'])
    cdf = np.arange(1, len(cl) + 1) / len(cl)
    ax1.plot(cl, cdf, 'b-', linewidth=2, label='Simulated')
    ax1.axvline(result_a['median_cl'], color='b', linestyle='--', alpha=0.7,
                label=f"Median = {result_a['median_cl']:.1f} dB")
    ax1.axvline(result_a['ref_cl'], color='r', linestyle='--', alpha=0.7,
                label=f"Ref = {result_a['ref_cl']:.1f} dB")
    ax1.set_xlabel('Coupling Loss (dB)')
    ax1.set_ylabel('CDF')
    ax1.set_title('Coupling Loss CDF (UMa 3.5GHz)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(REFERENCE_CHANNEL_UMA['coupling_loss_range_db'])
    ax1.grid(True, alpha=0.3)

    # --- SINR CDF ---
    sinr = np.sort(result_b['sinr_per_ue'])
    cdf2 = np.arange(1, len(sinr) + 1) / len(sinr)
    ax2.plot(sinr, cdf2, 'b-', linewidth=2, label='Simulated')
    ax2.axvline(result_b['median_sinr'], color='b', linestyle='--', alpha=0.7,
                label=f"Median = {result_b['median_sinr']:.1f} dB")
    ax2.axvline(result_b['ref_sinr'], color='r', linestyle='--', alpha=0.7,
                label=f"Ref = {result_b['ref_sinr']:.1f} dB")
    ax2.set_xlabel('Geometry SINR (dB)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Geometry SINR CDF (UMa 3.5GHz)')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(REFERENCE_CHANNEL_UMA['geometry_sinr_range_db'])
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'layer2_channel_cdf')


def plot_pathloss_vs_distance(result_a):
    """路径损耗 vs 距离散点图 + 理论曲线"""
    fig, ax = plt.subplots(figsize=(10, 7))

    d = result_a['distances']
    pl = result_a['pathloss_db']
    is_los = result_a['is_los']

    # 散点 (含阴影衰落的耦合损耗用浅色, 纯路径损耗用深色)
    ax.scatter(d[is_los], pl[is_los], c='green', alpha=0.3, s=8, label='LOS (sim)')
    ax.scatter(d[~is_los], pl[~is_los], c='red', alpha=0.3, s=8, label='NLOS (sim)')

    # 理论曲线
    d_theory = np.linspace(R_MIN, R_MAX, 500)
    pl_los_theory = np.array([compute_pathloss_uma(dd, H_BS, H_UT, FC_GHZ, True)
                              for dd in d_theory])
    pl_nlos_theory = np.array([compute_pathloss_uma(dd, H_BS, H_UT, FC_GHZ, False)
                               for dd in d_theory])

    ax.plot(d_theory, pl_los_theory, 'g-', linewidth=2.5, label='LOS (theory)')
    ax.plot(d_theory, pl_nlos_theory, 'r-', linewidth=2.5, label='NLOS (theory)')

    ax.set_xlabel('2D Distance (m)')
    ax.set_ylabel('Pathloss (dB)')
    ax.set_title('Pathloss vs Distance — UMa 3.5GHz (TR 38.901)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, R_MAX + 20)

    fig.tight_layout()
    save_figure(fig, 'layer2_pathloss_vs_distance')


def generate_report(result_a, result_b):
    """生成校准报告"""
    pass_a = abs(result_a['dev_cl']) <= 2.0
    pass_b = abs(result_b['dev_sinr']) <= 2.0
    all_pass = pass_a and pass_b

    los_pct = 100 * result_a['is_los'].mean()

    content = f"""# Layer 2: Channel Model Calibration

## Overview

Verify the statistical channel model's pathloss distribution and geometry SINR CDF
against TR 38.901 section 7.8 calibration references for UMa 3.5 GHz.

## Configuration

| Parameter | Value |
|-----------|-------|
| Scenario | UMa |
| Carrier frequency | {FC_GHZ} GHz |
| BS height | {H_BS} m |
| UE height | {H_UT} m |
| Cell radius | {R_MAX} m |
| Min distance | {R_MIN} m |
| Shadow fading (LOS) | {SHADOW_STD_LOS} dB |
| Shadow fading (NLOS) | {SHADOW_STD_NLOS} dB |
| Part A UEs | {NUM_UE_PARTA} |
| Part B UEs | {NUM_UE_PARTB} |
| Part B slots | {NUM_SLOTS_PARTB} |
| Part B config | 4Tx/2Rx, FDD, statistical channel |

## Part A: Coupling Loss CDF

LOS ratio: {los_pct:.1f}%

![Coupling Loss CDF](figures/layer2_channel_cdf.png)

![Pathloss vs Distance](figures/layer2_pathloss_vs_distance.png)

## Part B: Geometry SINR CDF

## CDF Comparison

| Metric | Simulated Median | Reference | Deviation | Status |
|--------|-----------------|-----------|-----------|--------|
| Coupling Loss (dB) | {result_a['median_cl']:.1f} | {result_a['ref_cl']:.1f} | {result_a['dev_cl']:+.1f} | {'PASS' if pass_a else 'FAIL'} |
| Geometry SINR (dB) | {result_b['median_sinr']:.1f} | {result_b['ref_sinr']:.1f} | {result_b['dev_sinr']:+.1f} | {'PASS' if pass_b else 'FAIL'} |

Pass criterion: CDF median deviation <= 2 dB

## Conclusion

**{'ALL PASS' if all_pass else 'SOME METRICS EXCEED THRESHOLD'}**

{'All metrics within 2 dB of TR 38.901 reference values.' if all_pass else 'One or more metrics deviate by more than 2 dB from reference.'}
"""
    write_report('layer2_channel_model.md', content)


def main():
    print("=" * 60)
    print("Layer 2: Channel Model Calibration")
    print("=" * 60)

    setup_plot_style()
    rng = np.random.default_rng(SEED)

    # Part A
    result_a = run_part_a(rng)

    # Part B
    result_b = run_part_b()

    # Plots
    plot_cdf(result_a, result_b)
    plot_pathloss_vs_distance(result_a)

    # Report
    generate_report(result_a, result_b)

    # Summary
    print("\n" + "=" * 60)
    pass_a = abs(result_a['dev_cl']) <= 2.0
    pass_b = abs(result_b['dev_sinr']) <= 2.0
    print(f"  Coupling Loss: median={result_a['median_cl']:.1f} dB, "
          f"ref={result_a['ref_cl']:.1f} dB, "
          f"dev={result_a['dev_cl']:+.1f} dB  {'PASS' if pass_a else 'FAIL'}")
    print(f"  Geometry SINR: median={result_b['median_sinr']:.1f} dB, "
          f"ref={result_b['ref_sinr']:.1f} dB, "
          f"dev={result_b['dev_sinr']:+.1f} dB  {'PASS' if pass_b else 'FAIL'}")
    overall = "ALL PASS" if (pass_a and pass_b) else "SOME FAIL"
    print(f"  Overall: {overall}")
    print("=" * 60)


if __name__ == '__main__':
    main()

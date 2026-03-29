#!/usr/bin/env python3
"""Monte Carlo 多种子批跑框架

用法:
  .venv312/Scripts/python.exe examples/run_batch.py --seeds 5 --slots 2000
  .venv312/Scripts/python.exe examples/run_batch.py --seeds 10 --slots 5000 --workers 4
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def run_single_seed(seed: int, num_slots: int, channel_type: str,
                    num_ue: int, scenario: str) -> dict:
    """运行单次仿真 (独立进程)"""
    from l2_rrm_sim.config.sim_config import (
        SimConfig, CarrierConfig, CellConfig, UEConfig,
        SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
        ChannelConfig, CSIConfig,
    )
    from l2_rrm_sim.core.simulation_engine import SimulationEngine

    config = {
        'sim': SimConfig(num_slots=num_slots, random_seed=seed, warmup_slots=min(500, num_slots//5)),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273,
                                  bandwidth_mhz=100, carrier_freq_ghz=3.5),
        'cell': CellConfig(num_tx_ant=32, num_tx_ports=4, max_layers=4,
                            cell_radius_m=500, scenario=scenario),
        'ue': UEConfig(num_ue=num_ue, num_rx_ant=2, speed_kmh=3),
        'scheduler': SchedulerConfig(),
        'link_adaptation': LinkAdaptationConfig(),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type=channel_type, scenario=scenario),
        'csi': CSIConfig(enabled=False),
    }

    engine = SimulationEngine(config)
    report = engine.run()

    # 只返回可序列化的标量 KPI
    return {
        'seed': seed,
        'cell_avg_throughput_mbps': report['cell_avg_throughput_mbps'],
        'cell_edge_throughput_mbps': report['cell_edge_throughput_mbps'],
        'spectral_efficiency_bps_hz': report['spectral_efficiency_bps_hz'],
        'avg_bler': report['avg_bler'],
        'avg_mcs': report['avg_mcs'],
        'avg_sinr_db': report['avg_sinr_db'],
        'jain_fairness': report['jain_fairness'],
        'prb_utilization': report['prb_utilization'],
        'cell_experienced_rate_mbps': report['cell_experienced_rate_mbps'],
    }


def aggregate_results(results: list) -> dict:
    """聚合多次仿真结果: 均值、标准差、置信区间"""
    if not results:
        return {}

    keys = [k for k in results[0].keys() if k != 'seed']
    agg = {}

    for key in keys:
        values = np.array([r[key] for r in results])
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0

        agg[key] = {
            'mean': mean,
            'std': std,
            'ci95': ci95,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values.tolist(),
        }

    return agg


def print_summary(agg: dict, num_seeds: int):
    """打印汇总报告"""
    print("\n" + "=" * 70)
    print(f"  Monte Carlo 汇总 ({num_seeds} seeds)")
    print("=" * 70)
    fmt = "  {:<35s}  {:>8s} +/- {:>6s}  [{:>8s}, {:>8s}]"
    print(fmt.format("KPI", "Mean", "CI95", "Min", "Max"))
    print("-" * 70)

    display_keys = [
        ('cell_avg_throughput_mbps', 'Mbps'),
        ('cell_edge_throughput_mbps', 'Mbps'),
        ('spectral_efficiency_bps_hz', 'bps/Hz'),
        ('avg_bler', ''),
        ('avg_mcs', ''),
        ('avg_sinr_db', 'dB'),
        ('jain_fairness', ''),
        ('cell_experienced_rate_mbps', 'Mbps'),
    ]

    for key, unit in display_keys:
        if key not in agg:
            continue
        d = agg[key]
        label = f"{key} ({unit})" if unit else key
        print(f"  {label:<35s}  {d['mean']:>8.2f} +/- {d['ci95']:>6.2f}  "
              f"[{d['min']:>8.2f}, {d['max']:>8.2f}]")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo 多种子批跑')
    parser.add_argument('--seeds', type=int, default=5, help='种子数量')
    parser.add_argument('--seed-start', type=int, default=0, help='起始种子')
    parser.add_argument('--slots', type=int, default=2000, help='每次仿真 slots')
    parser.add_argument('--workers', type=int, default=1, help='并行进程数')
    parser.add_argument('--channel', type=str, default='statistical',
                        choices=['statistical', 'sionna'])
    parser.add_argument('--num-ue', type=int, default=10)
    parser.add_argument('--scenario', type=str, default='uma')
    parser.add_argument('--output', type=str, default=None, help='JSON 输出路径')
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print(f"Monte Carlo 批跑: {args.seeds} seeds, {args.slots} slots/seed, "
          f"{args.workers} workers")
    print(f"  Channel: {args.channel}, UEs: {args.num_ue}, Scenario: {args.scenario}")

    t0 = time.time()
    results = []

    if args.workers <= 1:
        # 串行
        for i, seed in enumerate(seeds):
            print(f"\n--- Seed {seed} ({i+1}/{len(seeds)}) ---")
            r = run_single_seed(seed, args.slots, args.channel,
                                args.num_ue, args.scenario)
            results.append(r)
            print(f"  TP={r['cell_avg_throughput_mbps']:.1f} Mbps, "
                  f"BLER={r['avg_bler']:.3f}")
    else:
        # 并行
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for seed in seeds:
                f = executor.submit(run_single_seed, seed, args.slots,
                                    args.channel, args.num_ue, args.scenario)
                futures[f] = seed

            for f in as_completed(futures):
                seed = futures[f]
                try:
                    r = f.result()
                    results.append(r)
                    print(f"  Seed {seed} done: TP={r['cell_avg_throughput_mbps']:.1f} Mbps")
                except Exception as e:
                    print(f"  Seed {seed} FAILED: {e}")

    elapsed = time.time() - t0

    # 聚合
    agg = aggregate_results(results)
    print_summary(agg, len(results))
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/len(results):.1f}s/seed)")

    # 保存
    if args.output:
        out_path = Path(args.output)
        with open(out_path, 'w') as f:
            json.dump({'seeds': seeds, 'results': results, 'aggregate': agg},
                      f, indent=2, default=str)
        print(f"结果保存: {out_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""多场景综合测试

测试用例:
  Case 1: 单小区 Full Buffer, UMa, 20 UE, 均匀撒点
  Case 2: 单小区 Full Buffer, UMa, 10 UE 近点 + 10 UE 远点
  Case 3: 单小区 FTP Model 3, 不同到达率
  Case 4: 单小区 混合流量 (10 UE Full Buffer + 10 UE Poisson)
  Case 5: 多小区 7站, Full Buffer, 含干扰
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from l2_rrm_sim.config import load_config
from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig, ChannelConfig
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from l2_rrm_sim.core.data_types import UEState
from l2_rrm_sim.traffic.ftp_model import FTPModel3
from l2_rrm_sim.traffic.bursty_traffic import PoissonTraffic
from l2_rrm_sim.traffic.full_buffer import FullBufferTraffic, INFINITE_BUFFER
from l2_rrm_sim.kpi.kpi_reporter import KPIReporter


def make_default_config(num_slots=5000, num_ue=20, scenario='uma',
                         cell_radius=500, speed_kmh=3.0):
    """构建默认配置"""
    return {
        'sim': SimConfig(num_slots=num_slots, random_seed=42, warmup_slots=500),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273,
                                  bandwidth_mhz=100, carrier_freq_ghz=3.5),
        'cell': CellConfig(cell_radius_m=cell_radius, scenario=scenario,
                            height_m=25.0, total_power_dbm=46.0),
        'ue': UEConfig(num_ue=num_ue, min_distance_m=35, max_distance_m=cell_radius,
                        speed_kmh=speed_kmh),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario=scenario),
    }


def print_case_header(case_num, description):
    print("\n" + "=" * 70)
    print(f"  Case {case_num}: {description}")
    print("=" * 70)


def run_case1():
    """单小区 Full Buffer, UMa, 20 UE, 均匀撒点"""
    print_case_header(1, "单小区 Full Buffer, UMa, 20 UE, 均匀撒点")
    config = make_default_config(num_slots=5000, num_ue=20)
    engine = SimulationEngine(config)
    report = engine.run()
    return report


def run_case2():
    """单小区 Full Buffer, UMa, 近远点混合"""
    print_case_header(2, "单小区 Full Buffer, UMa, 10近+10远")
    config = make_default_config(num_slots=5000, num_ue=20)
    engine = SimulationEngine(config)

    # 重新设置 UE 位置: 10 近点 (35-100m), 10 远点 (300-500m)
    rng = engine.rng.general
    for i in range(10):
        r = np.sqrt(rng.uniform(35**2, 100**2))
        theta = rng.uniform(0, 2 * np.pi)
        engine.ue_states[i].position = np.array([r*np.cos(theta), r*np.sin(theta), 1.5])
    for i in range(10, 20):
        r = np.sqrt(rng.uniform(300**2, 500**2))
        theta = rng.uniform(0, 2 * np.pi)
        engine.ue_states[i].position = np.array([r*np.cos(theta), r*np.sin(theta), 1.5])

    # 重新初始化信道
    engine.channel.initialize(engine.cell_config, engine.carrier_config, engine.ue_states)
    report = engine.run()

    # 打印近/远点对比
    ue_tp = report['ue_avg_throughput_mbps']
    near_tp = ue_tp[:10]
    far_tp = ue_tp[10:]
    print(f"  近点 UE (35-100m): avg={np.mean(near_tp):.2f} Mbps")
    print(f"  远点 UE (300-500m): avg={np.mean(far_tp):.2f} Mbps")
    return report


def run_case3():
    """单小区 FTP Model 3, 不同到达率"""
    print_case_header(3, "单小区 FTP Model 3, 到达率=5 files/s")
    config = make_default_config(num_slots=8000, num_ue=20)
    engine = SimulationEngine(config)

    ftp = FTPModel3(
        file_size_bytes=256_000,     # 250 KB
        arrival_rate=5.0,             # 5 files/s/UE
        slot_duration_s=engine.carrier_config.slot_duration_s,
        num_ue=engine.num_ue,
        rng=engine.rng.traffic,
    )
    engine.traffic = ftp

    # 包装 run_slot 以跟踪 FTP 文件完成
    original_run_slot = engine.run_slot
    def run_slot_with_ftp(slot_idx):
        result = original_run_slot(slot_idx)
        for ue_idx in range(engine.num_ue):
            decoded_bytes = int(result.ue_decoded_bits[ue_idx]) // 8
            if decoded_bytes > 0:
                ftp.dequeue_bytes(ue_idx, decoded_bytes, slot_idx)
        return result
    engine.run_slot = run_slot_with_ftp

    report = engine.run()

    latency = ftp.get_file_latency_stats()
    print(f"  文件完成数: {latency['count']}")
    print(f"  平均时延: {latency['mean_ms']:.1f} ms, "
          f"50%ile: {latency['p50_ms']:.1f} ms, "
          f"95%ile: {latency['p95_ms']:.1f} ms")
    return report


def run_case4():
    """单小区 混合流量: 10 UE Full Buffer + 10 UE FTP"""
    print_case_header(4, "单小区 混合流量 (10 FullBuffer + 10 FTP)")
    config = make_default_config(num_slots=5000, num_ue=20)
    engine = SimulationEngine(config)

    ftp = FTPModel3(
        file_size_bytes=256_000,     # 250 KB
        arrival_rate=10.0,            # 10 files/s/UE (高负载)
        slot_duration_s=engine.carrier_config.slot_duration_s,
        num_ue=10,                    # 只有 UE 10-19 使用 FTP
        rng=engine.rng.traffic,
    )

    # 自定义流量: UE 0-9 Full Buffer, UE 10-19 FTP
    class MixedTraffic:
        def generate(self, slot_ctx, ue_states):
            for i, ue in enumerate(ue_states):
                if i < 10:
                    ue.buffer_bytes = INFINITE_BUFFER
                else:
                    pass  # FTP 在下面处理
            # FTP UE 10-19
            ftp.generate(slot_ctx, ue_states[10:])

    engine.traffic = MixedTraffic()

    # 包装 run_slot 以跟踪 FTP 文件完成
    original_run_slot = engine.run_slot
    def run_slot_wrapped(slot_idx):
        result = original_run_slot(slot_idx)
        for ue_idx in range(10, engine.num_ue):
            decoded_bytes = int(result.ue_decoded_bits[ue_idx]) // 8
            if decoded_bytes > 0:
                ftp.dequeue_bytes(ue_idx - 10, decoded_bytes, slot_idx)
        return result
    engine.run_slot = run_slot_wrapped

    report = engine.run()

    ue_tp = report['ue_avg_throughput_mbps']
    print(f"  Full Buffer UE (0-9):  avg={np.mean(ue_tp[:10]):.2f} Mbps")
    print(f"  FTP UE (10-19):        avg={np.mean(ue_tp[10:]):.2f} Mbps")
    latency = ftp.get_file_latency_stats()
    if latency['count'] > 0:
        print(f"  FTP 文件完成: {latency['count']}, 均时延: {latency['mean_ms']:.0f}ms")
    return report


def run_case5():
    """多小区 7站 Full Buffer"""
    print_case_header(5, "多小区 7站21小区, Full Buffer, ICI=0.8")
    from l2_rrm_sim.core.multicell_engine import MultiCellSimulationEngine

    config = make_default_config(num_slots=2000, num_ue=5, cell_radius=250)
    engine = MultiCellSimulationEngine(
        config, num_rings=1, num_ue_per_cell=5, ici_load_factor=0.8
    )
    report = engine.run()
    return report


def main():
    results = {}
    print("\n" + "#" * 70)
    print("  L2 RRM 仿真平台 — 多场景综合测试")
    print("#" * 70)

    results['case1'] = run_case1()
    results['case2'] = run_case2()
    results['case3'] = run_case3()
    results['case4'] = run_case4()
    results['case5'] = run_case5()

    # 汇总
    print("\n" + "#" * 70)
    print("  测试汇总")
    print("#" * 70)
    summary_fmt = "  {:<45s}  {:>8s}  {:>8s}  {:>6s}  {:>6s}"
    print(summary_fmt.format("场景", "小区TP", "体验速率", "BLER", "Jain"))
    print(summary_fmt.format("", "(Mbps)", "(Mbps)", "", ""))
    print("-" * 80)

    labels = {
        'case1': 'Full Buffer, 均匀撒点',
        'case2': 'Full Buffer, 近远混合',
        'case3': 'FTP Model 3, rate=5/s',
        'case4': '混合流量 (FB+FTP)',
        'case5': '多小区 7站 ICI=0.8',
    }

    for key, label in labels.items():
        r = results[key]
        cell_tp = r.get('cell_avg_throughput_mbps', r.get('avg_cell_throughput_mbps', 0))
        exp_rate = r.get('cell_experienced_rate_mbps', r.get('avg_ue_throughput_mbps', 0))
        bler = r.get('avg_bler', 0)
        jain = r.get('jain_fairness', 0)
        print(f"  {label:<45s}  {cell_tp:>8.1f}  {exp_rate:>8.2f}  {bler:>6.3f}  {jain:>6.3f}")

    print("#" * 70)
    print("  所有测试完成!")


if __name__ == '__main__':
    main()

"""SRS 性能对比实验脚本

对比:
1. Ideal CSI: 全带宽更新, 无延迟 (Upper Bound)
2. Realistic SRS: 17倍跳频, 5ms周期, 4 slots处理时延 (Realistic TDD)
"""

import numpy as np
import matplotlib.pyplot as plt
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig, ChannelConfig,
    CSIConfig
)

def run_experiment(name, srs_mode_params):
    print(f"\n>>> 开始实验: {name} ...")
    
    # 基础配置 (30kHz, 100MHz, 20 UEs)
    config = {
        'sim': SimConfig(num_slots=100, warmup_slots=0, random_seed=42),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273),
        'cell': CellConfig(num_tx_ant=64, num_tx_ports=4, max_layers=4, total_power_dbm=46),
        'ue': UEConfig(num_ue=20, speed_kmh=30.0), # 30km/h 移动速度增强老化效应
        'scheduler': SchedulerConfig(type="pf"),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1),
        'traffic': TrafficConfig(type="full_buffer"),
        'channel': ChannelConfig(type="sionna"), # 只有 Sionna 支持矩阵信道
        'csi': CSIConfig(
            enabled=True,
            mode="srs_tdd",
            **srs_mode_params
        )
    }

    engine = SimulationEngine(config)
    report = engine.run()
    return report

if __name__ == "__main__":
    # 1. 理想参数: 无跳频, 无时延
    ideal_params = {
        'srs_period_slots': 1,
        'srs_hopping_subbands': 1,
        'srs_processing_delay': 0,
        'estimation_error_std': 0.01
    }
    
    # 2. 真实参数: 17倍跳频, 4 slots时延, 5ms周期
    real_params = {
        'srs_period_slots': 10,
        'srs_hopping_subbands': 17,
        'srs_processing_delay': 4,
        'estimation_error_std': 0.05
    }

    res_ideal = run_experiment("Ideal_CSI", ideal_params)
    res_real = run_experiment("Realistic_SRS", real_params)

    # --- 打印对比结果 ---
    print("\n" + "="*50)
    print(f"{'KPI':<25} | {'Ideal CSI':<12} | {'Realistic SRS':<12} | {'Loss (%)'}")
    print("-" * 65)
    
    kpis = [
        ('cell_avg_throughput_mbps', 'Cell Avg TP (Mbps)'),
        ('spectral_efficiency_bps_hz', 'Spec Eff (bps/Hz)'),
        ('avg_bler', 'Avg BLER'),
        ('jain_fairness', 'Jain Fairness'),
    ]

    for key, label in kpis:
        v_i = res_ideal[key]
        v_r = res_real[key]
        loss = (v_i - v_r) / v_i * 100 if v_i > 0 else 0
        print(f"{label:<25} | {v_i:>12.4f} | {v_r:>12.4f} | {loss:>8.1f}%")
    print("="*50)

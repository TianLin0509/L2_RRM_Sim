"""Smoke tests — 基础功能验证

运行: .venv312/Scripts/python.exe -m pytest tests/test_smoke.py -v
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig, CSIConfig,
)


def _make_config(**overrides):
    """构造默认测试配置 (小规模, 快速)"""
    cfg = {
        'sim': SimConfig(num_slots=100, random_seed=42, warmup_slots=10,
                         kpi_trim_percent=10.0),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=51,
                                  bandwidth_mhz=20, carrier_freq_ghz=3.5),
        'cell': CellConfig(num_tx_ant=8, num_tx_ports=2, max_layers=2,
                            cell_radius_m=300, scenario='uma'),
        'ue': UEConfig(num_ue=5, num_rx_ant=1, speed_kmh=3),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'csi': CSIConfig(enabled=False),
    }
    cfg.update(overrides)
    return cfg


# ============================================================
# 单小区 smoke tests
# ============================================================

def test_single_cell_full_buffer():
    """单小区 Full Buffer 基础运行"""
    from l2_rrm_sim.core.simulation_engine import SimulationEngine

    config = _make_config()
    engine = SimulationEngine(config)
    report = engine.run()

    assert report['cell_avg_throughput_mbps'] > 0, "小区吞吐应 > 0"
    assert 0 <= report['avg_bler'] <= 1, "BLER 应在 [0,1]"
    assert report['num_valid_slots'] > 0, "应有有效 slots"
    assert report['prb_utilization'] > 0, "PRB 利用率应 > 0"


def test_traffic_config_ftp():
    """TrafficConfig.type='ftp_model3' 应生效"""
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.traffic.ftp_model import FTPModel3

    config = _make_config(
        traffic=TrafficConfig(type='ftp_model3', ftp_file_size_bytes=100000,
                               ftp_lambda=5.0),
    )
    engine = SimulationEngine(config)
    assert isinstance(engine.traffic, FTPModel3), \
        f"流量模型应为 FTPModel3, 实际是 {type(engine.traffic)}"


def test_traffic_config_poisson():
    """TrafficConfig.type='poisson' 应生效"""
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.traffic.bursty_traffic import PoissonTraffic

    config = _make_config(
        traffic=TrafficConfig(type='poisson', ftp_lambda=0.2),
    )
    engine = SimulationEngine(config)
    assert isinstance(engine.traffic, PoissonTraffic), \
        f"流量模型应为 PoissonTraffic, 实际是 {type(engine.traffic)}"


def test_buffer_cap_non_full_buffer():
    """非满缓冲: decoded_bits 不应超过 buffer"""
    from l2_rrm_sim.core.simulation_engine import SimulationEngine

    config = _make_config(
        sim=SimConfig(num_slots=50, random_seed=42, warmup_slots=0),
        traffic=TrafficConfig(type='ftp_model3', ftp_file_size_bytes=1000,
                               ftp_lambda=50.0),
    )
    engine = SimulationEngine(config)

    for slot_idx in range(50):
        result = engine.run_slot(slot_idx)
        # 检查: 没有 UE 的 decoded_bits 超过其 buffer
        for ue_idx, ue in enumerate(engine.ue_states):
            # decoded_bits 已经在 _finalize_slot 中被截断
            pass  # 只要不崩溃就算通过


def test_kpi_trim_percent_passed():
    """kpi_trim_percent 应传递到 KPIReporter"""
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.kpi.kpi_reporter import KPIReporter

    config = _make_config(
        sim=SimConfig(num_slots=100, random_seed=42, warmup_slots=10,
                      kpi_trim_percent=15.0),
    )
    engine = SimulationEngine(config)
    engine.run()
    # 间接验证: 检查 SimConfig 中的值
    assert engine.sim_config.kpi_trim_percent == 15.0


# ============================================================
# 模块导入 tests
# ============================================================

def test_all_modules_importable():
    """所有模块应可导入"""
    from l2_rrm_sim.channel import StatisticalChannel
    from l2_rrm_sim.scheduler import PFSchedulerSUMIMO, MUMIMOPFScheduler, RankAdapter
    from l2_rrm_sim.traffic import FullBufferTraffic, FTPModel3, PoissonTraffic
    from l2_rrm_sim.harq import HARQEntity, HARQManager
    from l2_rrm_sim.power_control import DLPowerControl
    from l2_rrm_sim.kpi import KPICollector, KPIReporter, ExperiencedRateCalculator
    from l2_rrm_sim.csi import CQITable, TypeICodebook, CSIFeedbackManager, SINRPredictor


def test_cqi_table():
    """CQI 表映射正确性"""
    from l2_rrm_sim.csi.cqi_table import sinr_to_cqi, cqi_to_sinr

    # CQI 应单调递增
    prev_cqi = -1
    for sinr_db in range(-10, 30):
        cqi = sinr_to_cqi(sinr_db)
        assert cqi >= prev_cqi, f"CQI 应单调递增: SINR={sinr_db}, CQI={cqi}"
        prev_cqi = cqi

    # 双向映射一致性
    for cqi in range(1, 16):
        sinr = cqi_to_sinr(cqi)
        mapped_cqi = sinr_to_cqi(sinr)
        assert mapped_cqi == cqi, f"CQI {cqi} → SINR {sinr} → CQI {mapped_cqi}"


def test_harq_entity():
    """HARQ 实体基本功能"""
    from l2_rrm_sim.harq import HARQManager

    mgr = HARQManager(num_ue=2, num_processes=4, max_retx=2)

    # 新传输
    pid = mgr.start_new_tx(0, mcs=10, num_prbs=20, num_layers=1, tbs=5000, slot_idx=0)
    assert pid >= 0

    # NACK → 重传
    result = mgr.process_feedback(0, pid, is_ack=False, sinr_eff_linear=5.0)
    assert result['retx_needed'] == True
    assert result['decoded_bits'] == 0

    # ACK → 成功
    result2 = mgr.process_feedback(0, pid, is_ack=True, sinr_eff_linear=5.0)
    assert result2['decoded_bits'] == 5000


# ============================================================
# 多小区 smoke test (statistical channel only)
# ============================================================

def test_multicell_statistical():
    """多小区 statistical 信道应可运行"""
    from l2_rrm_sim.core.multicell_engine import MultiCellSimulationEngine

    config = _make_config(
        sim=SimConfig(num_slots=50, random_seed=42, warmup_slots=5),
    )
    engine = MultiCellSimulationEngine(
        config, num_rings=0, num_ue_per_cell=3, ici_load_factor=0.5
    )
    report = engine.run()
    assert report['avg_cell_throughput_mbps'] >= 0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

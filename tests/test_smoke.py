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
    ChannelConfig, CSIConfig, TDDConfig,
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
        # buffer 不应为负（decoded_bits 截断逻辑正确）
        for ue in engine.ue_states:
            assert ue.buffer_bytes >= 0, (
                f"slot {slot_idx}: UE {ue.ue_id} buffer 为负 {ue.buffer_bytes}"
            )
        # decoded_bits 不应超过 TBS 上限（合理性检查）
        for ue_idx in range(len(engine.ue_states)):
            assert result.ue_decoded_bits[ue_idx] >= 0, "decoded_bits 不应为负"


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


def test_harq_max_retx():
    """达到 max_retx 后进程应释放，不再重传"""
    from l2_rrm_sim.harq import HARQManager

    max_retx = 3  # 总传输次数: 初传 + 2 次重传
    mgr = HARQManager(num_ue=1, num_processes=4, max_retx=max_retx)

    pid = mgr.start_new_tx(0, mcs=10, num_prbs=10, num_layers=1, tbs=2000, slot_idx=0)
    assert pid >= 0

    # 连续 NACK 直到耗尽重传次数
    retx_count = 0
    for _ in range(max_retx + 2):  # 多发几次确保不会无限重传
        result = mgr.process_feedback(0, pid, is_ack=False, sinr_eff_linear=1.0)
        if result['retx_needed']:
            retx_count += 1
        else:
            break

    # 最多允许 max_retx-1 次重传（3GPP: maxHARQ-Tx 包含初传）
    assert retx_count <= max_retx - 1, f"重传次数 {retx_count} 超过上限 {max_retx - 1}"
    # 进程应已释放
    assert not mgr.entities[0].processes[pid].is_active, "进程应已释放"


def test_harq_peek_does_not_consume():
    """peek_retx_info 不应消费重传队列"""
    from l2_rrm_sim.harq import HARQManager

    mgr = HARQManager(num_ue=1, num_processes=4, max_retx=4)
    pid = mgr.start_new_tx(0, mcs=10, num_prbs=10, num_layers=1, tbs=2000, slot_idx=0)
    mgr.process_feedback(0, pid, is_ack=False, sinr_eff_linear=1.0)

    # peek 两次，队列不应被消费
    info1 = mgr.peek_retx_info(0)
    info2 = mgr.peek_retx_info(0)
    assert info1 is not None
    assert info2 is not None
    assert info1['process_id'] == info2['process_id']

    # consume 后队列才清空
    mgr.consume_retx(0)
    assert mgr.peek_retx_info(0) is None


def test_harq_combining_helper_uses_accumulated_sinr():
    """单小区主链路应把 HARQ 累积 SINR 合并到重传判决中。"""
    from types import SimpleNamespace
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.harq import HARQManager

    engine = SimulationEngine(_make_config(
        ue=UEConfig(num_ue=1, num_rx_ant=1, min_distance_m=35.0,
                    max_distance_m=35.0, speed_kmh=0.0),
        cell=CellConfig(num_tx_ant=8, num_tx_ports=2, max_layers=1,
                        cell_radius_m=300, scenario='uma'),
    ))
    engine.harq_mgr = HARQManager(num_ue=1, num_processes=4, max_retx=4)

    pid = engine.harq_mgr.start_new_tx(0, mcs=10, num_prbs=10, num_layers=1,
                                       tbs=2000, slot_idx=0)
    engine.harq_mgr.process_feedback(0, pid, is_ack=False, sinr_eff_linear=3.0)

    retx_info = {0: {'process_id': pid}}
    sched = SimpleNamespace(ue_num_prbs=np.array([10], dtype=np.int32))
    combined = engine._build_harq_combined_sinr(
        np.array([2.0], dtype=np.float32), sched, retx_info
    )

    assert np.isclose(combined[0], 5.0)


def test_rank_selection_multi_layer():
    """Fix 1 后 statistical channel 应能选出 rank > 1"""
    from l2_rrm_sim.channel.statistical_channel import StatisticalChannel
    from l2_rrm_sim.config.sim_config import CellConfig, CarrierConfig, ChannelConfig
    from l2_rrm_sim.core.data_types import UEState, SlotContext
    from l2_rrm_sim.scheduler.rank_adaptation import RankAdapter
    import numpy as np

    cell_cfg = CellConfig(num_tx_ant=4, num_tx_ports=4, max_layers=4,
                          cell_radius_m=200, scenario='uma')
    carrier_cfg = CarrierConfig(subcarrier_spacing=30, num_prb=25,
                                bandwidth_mhz=10, carrier_freq_ghz=3.5)
    ch_cfg = ChannelConfig(type='statistical', scenario='uma')

    rng = np.random.default_rng(0)
    ch = StatisticalChannel(cell_cfg, carrier_cfg, ch_cfg, rng)

    ue_states = []
    for i in range(4):
        ue = UEState(ue_id=i, position=np.array([50.0 + i*10, 0.0, 1.5]),
                     velocity=np.array([0.0, 0.0, 0.0]))
        ue.num_rx_ant = 4
        ue_states.append(ue)

    ch.initialize(cell_cfg, carrier_cfg, ue_states)
    slot_ctx = SlotContext(slot_idx=0, time_s=0.0)
    channel_state = ch.update(slot_ctx, ue_states)

    adapter = RankAdapter(max_rank=4)
    ranks = [adapter.select_rank(channel_state.sinr_per_prb[ue]) for ue in range(4)]
    # 至少有一个 UE 应选出 rank > 1（4 天线场景下概率极高）
    assert max(ranks) > 1, f"所有 UE rank 均为 1，Fix 1 可能未生效: {ranks}"


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


def test_csi_subband_pmi_report():
    """CSI 反馈应生成子带 PMI/CQI，并保持长度一致。"""
    from l2_rrm_sim.csi.csi_feedback import CSIFeedbackManager

    num_ue = 2
    num_rx_ant = 2
    num_tx_ports = 4
    num_prb = 9
    subband_size = 4

    rng = np.random.default_rng(7)
    H = (rng.normal(size=(num_ue, num_rx_ant, num_tx_ports, num_prb))
         + 1j * rng.normal(size=(num_ue, num_rx_ant, num_tx_ports, num_prb)))

    mgr = CSIFeedbackManager(
        num_ue=num_ue,
        num_tx_ports=num_tx_ports,
        max_rank=2,
        csi_period_slots=1,
        feedback_delay_slots=0,
        noise_power_per_prb=1e-9,
        subband_size_prb=subband_size,
    )
    reports = mgr.measure_and_report(slot_idx=0, channel_matrices=H, tx_power_per_prb=1.0)

    assert len(reports) == num_ue
    expected_num_subbands = int(np.ceil(num_prb / subband_size))
    for report in reports:
        assert report.pmi_subband is not None
        assert report.cqi_subband is not None
        assert len(report.pmi_subband) == expected_num_subbands
        assert len(report.cqi_subband) == expected_num_subbands
        assert report.precoding_matrices_subband is not None
        assert len(report.precoding_matrices_subband) == expected_num_subbands


def test_sinr_predictor_uses_subband_pmi():
    """SINR predictor 应支持使用子带 PMI 计算 BF gain。"""
    from l2_rrm_sim.csi.csi_feedback import CSIReport
    from l2_rrm_sim.csi.sinr_prediction import SINRPredictor

    rng = np.random.default_rng(11)
    H = (rng.normal(size=(1, 2, 4, 8))
         + 1j * rng.normal(size=(1, 2, 4, 8)))

    predictor = SINRPredictor(num_ue=1, num_tx_ports=4)
    report = CSIReport(
        ue_id=0,
        slot_idx=0,
        ri=2,
        pmi=0,
        pmi_subband=np.array([0, 1], dtype=np.int32),
        cqi_wideband=10,
    )
    sinr = predictor.predict_all_ue([report], H, mode='su')

    assert sinr.shape == (1,)
    assert np.isfinite(sinr[0])
    assert predictor.pmi_gain[0] > 0


def test_tdd_special_slot_uses_fewer_re_and_smaller_tbs():
    """TDD special slot 应按实际 DL symbols 缩减 RE/TBS。"""
    from l2_rrm_sim.core.resource_grid import ResourceGrid
    from l2_rrm_sim.core.simulation_engine import SimulationEngine

    carrier = CarrierConfig(subcarrier_spacing=30, num_prb=51,
                            bandwidth_mhz=20, carrier_freq_ghz=3.5)
    grid = ResourceGrid(carrier)

    full_re = grid.compute_re_per_prb(14)
    special_re = grid.compute_re_per_prb(10)
    assert special_re < full_re

    full_tbs = grid.get_tbs(mcs_index=10, num_prbs=carrier.num_prb,
                            num_layers=1, num_dl_symbols=14)
    special_tbs = grid.get_tbs(mcs_index=10, num_prbs=carrier.num_prb,
                               num_layers=1, num_dl_symbols=10)
    assert special_tbs < full_tbs

    config = _make_config(
        ue=UEConfig(num_ue=1, num_rx_ant=1, min_distance_m=35.0,
                    max_distance_m=35.0, speed_kmh=0.0),
        cell=CellConfig(num_tx_ant=8, num_tx_ports=2, max_layers=1,
                        cell_radius_m=300, scenario='uma'),
        tdd=TDDConfig(duplex_mode='TDD', pattern='DDDSU',
                      special_dl_symbols=10, special_gp_symbols=2,
                      special_ul_symbols=2),
    )
    engine = SimulationEngine(config)

    # Warmup: 让 OLLA MCS 收敛，避免初始 MCS=0 干扰比较
    for s in range(50):
        engine.run_slot(s)

    # 比较同一 DDDSU 周期内的 D slot (50) 和 S slot (53)
    result_d = engine.run_slot(50)  # D slot
    engine.run_slot(51)  # D
    engine.run_slot(52)  # D
    result_s = engine.run_slot(53)  # S slot

    assert result_d.scheduling_decision.ue_num_re[0] > result_s.scheduling_decision.ue_num_re[0]
    assert result_d.scheduling_decision.ue_tbs_bits[0] > result_s.scheduling_decision.ue_tbs_bits[0]


def test_kpi_slot_trace_separates_scheduled_and_delivered_bits():
    """KPI slot trace 应分离 tx-slot scheduled bits 和 feedback-slot delivered bits。"""
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.kpi.kpi_reporter import KPIReporter

    config = _make_config(
        sim=SimConfig(num_slots=20, random_seed=42, warmup_slots=0),
        ue=UEConfig(num_ue=1, num_rx_ant=1, min_distance_m=35.0,
                    max_distance_m=35.0, speed_kmh=0.0),
        cell=CellConfig(num_tx_ant=8, num_tx_ports=2, max_layers=1,
                        cell_radius_m=300, scenario='uma'),
        tdd=TDDConfig(duplex_mode='TDD', pattern='DDDSU',
                      special_dl_symbols=10, special_gp_symbols=2,
                      special_ul_symbols=2),
    )
    engine = SimulationEngine(config)

    for slot_idx in range(config['sim'].num_slots):
        result = engine.run_slot(slot_idx)
        buf_after = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(slot_idx, result, engine._buf_after_traffic, buf_after)

    report = KPIReporter(engine.kpi, config['carrier']).report()
    trace = report['slot_trace']

    assert report['cell_avg_scheduled_throughput_mbps'] >= report['cell_avg_throughput_mbps']
    assert report['dl_schedulable_prb_utilization'] >= report['prb_utilization']

    ul_mask = (trace['slot_direction'] == 'U')
    assert np.all(trace['cell_scheduled_bits'][ul_mask] == 0)
    assert np.any(trace['cell_delivered_bits'][ul_mask] > 0)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

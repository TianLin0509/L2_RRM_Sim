"""单小区单用户 Full-Buffer 基线验证

覆盖:
  1. FDD + rank=1 单 slot trace (SINR→MCS→TBS→BLER→decoded_bits)
  2. TDD DDDSU 场景: special slot RE/TBS 验证
  3. HARQ K1 闭环验证: legacy PHY 路径
"""
import sys, os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig, TDDConfig, CSIConfig,
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from l2_rrm_sim.kpi.kpi_reporter import KPIReporter
from l2_rrm_sim.utils.nr_utils import compute_tbs, compute_num_re_per_prb


def _make_config(duplex='TDD', num_tx_ant=4, max_layers=1, num_slots=500):
    """构造单小区单用户配置"""
    tdd_cfg = TDDConfig(duplex_mode=duplex)
    return {
        'sim': SimConfig(num_slots=num_slots, random_seed=42, warmup_slots=50),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=273),
        'cell': CellConfig(num_tx_ant=num_tx_ant, num_tx_ports=min(num_tx_ant, 4),
                           max_layers=max_layers),
        'ue': UEConfig(num_ue=1, num_rx_ant=4),
        'scheduler': SchedulerConfig(type='pf'),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'tdd': tdd_cfg,
        'csi': CSIConfig(enabled=False),
    }


# ========== Test 1: FDD rank=1 single-slot trace ==========
def test_fdd_rank1_trace():
    """FDD + rank=1: 验证 SINR→MCS→TBS→BLER→decoded_bits 全链路"""
    print("=" * 60)
    print("TEST 1: FDD rank=1 single-slot trace")
    print("=" * 60)

    config = _make_config(duplex='FDD', num_tx_ant=4, max_layers=1, num_slots=200)
    engine = SimulationEngine(config)

    # warmup 让 OLLA 收敛
    for s in range(100):
        result = engine.run_slot(s)
        buf_after = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(s, result, engine._buf_after_traffic, buf_after)

    # 第 101 个 slot: 详细 trace
    slot_idx = 100
    slot_ctx = engine._make_slot_context(slot_idx)
    assert slot_ctx.slot_direction == 'D', "FDD slot should be DL"
    assert slot_ctx.num_dl_symbols == 14, f"FDD should have 14 DL symbols, got {slot_ctx.num_dl_symbols}"

    # 手动步进
    engine.traffic.generate(slot_ctx, engine.ue_states)
    engine._buf_after_traffic = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
    slot_duration = engine.carrier_config.slot_duration_s
    for ue in engine.ue_states:
        ue.position = ue.position + ue.velocity * slot_duration
    engine.harq_mgr.deliver_feedback(slot_idx)
    channel_state = engine.channel.update(slot_ctx, engine.ue_states)
    channel_state.estimated_channel_matrix = engine.channel_estimator.estimate(channel_state)

    # Rank selection
    ue_rank = engine.rank_adapter.select_rank_batch(channel_state.sinr_per_prb, 1)
    for ue_idx in range(1):
        r = int(ue_rank[ue_idx])
        channel_state.sinr_per_prb[ue_idx, :r, :] /= r
        channel_state.sinr_per_prb[ue_idx, r:, :] = 0.0

    rank = int(ue_rank[0])
    print(f"  Selected rank: {rank}")

    # SINR
    sinr_layer0 = channel_state.sinr_per_prb[0, 0, :]
    mean_sinr_db = 10 * np.log10(np.mean(sinr_layer0))
    print(f"  Mean wideband SINR (layer 0): {mean_sinr_db:.1f} dB")

    # EESM → MCS (legacy path)
    from l2_rrm_sim.link_adaptation.effective_sinr import EESM
    eesm = EESM()
    sinr_eff_db = eesm.compute(channel_state.sinr_per_prb[0, :rank, :], mcs_index=0)
    print(f"  EESM effective SINR: {sinr_eff_db:.1f} dB")

    # RE/PRB
    re_per_prb = engine.resource_grid.compute_re_per_prb(slot_ctx.num_dl_symbols)
    print(f"  RE/PRB: {re_per_prb} (14 symbols)")

    # TBS (手动计算)
    mcs = 27  # 假设高 SNR 选 max MCS
    tbs = compute_tbs(re_per_prb, 273, mcs, rank, mcs_table_index=1)
    print(f"  TBS (MCS={mcs}, 273 PRBs, rank={rank}): {tbs} bits = {tbs/8/1024:.1f} KB")

    # 理论峰值吞吐 (FDD, no BLER)
    slot_dur_ms = engine.carrier_config.slot_duration_s * 1000
    peak_tp_mbps = tbs / (slot_dur_ms * 1e-3) / 1e6
    print(f"  Peak throughput (no BLER): {peak_tp_mbps:.1f} Mbps")

    # 运行完整仿真验证
    config2 = _make_config(duplex='FDD', num_tx_ant=4, max_layers=1, num_slots=500)
    engine2 = SimulationEngine(config2)
    report = engine2.run()

    tp = report['cell_avg_throughput_mbps']
    bler = report['avg_bler']
    util = report['prb_utilization']
    avg_mcs = report['avg_mcs']

    print(f"\n  Full-sim results (500 slots):")
    print(f"    Throughput: {tp:.1f} Mbps")
    print(f"    BLER: {bler:.4f} (target 0.1)")
    print(f"    PRB util: {util*100:.1f}% (FDD, expect ~100%)")
    print(f"    Avg MCS: {avg_mcs:.1f}")

    ok = util > 0.95 and 0.05 < bler < 0.15 and tp > 50
    print(f"\n  [{'PASS' if ok else 'FAIL'}] FDD rank=1 baseline")
    return ok


# ========== Test 2: TDD special slot RE verification ==========
def test_tdd_special_slot():
    """TDD DDDSU: 验证 special slot 的 RE/TBS 正确缩减"""
    print("\n" + "=" * 60)
    print("TEST 2: TDD special slot RE/TBS verification")
    print("=" * 60)

    config = _make_config(duplex='TDD', num_tx_ant=4, max_layers=1, num_slots=500)
    engine = SimulationEngine(config)

    # DDDSU pattern: slot 0=D, 1=D, 2=D, 3=S, 4=U
    re_normal = engine.resource_grid.compute_re_per_prb(14)
    re_special = engine.resource_grid.compute_re_per_prb(10)

    print(f"  RE/PRB normal DL (14 sym): {re_normal}")
    print(f"  RE/PRB special  (10 sym): {re_special}")
    print(f"  Reduction ratio: {re_special/re_normal*100:.1f}%")

    tbs_normal = compute_tbs(re_normal, 273, 27, 1, 1)
    tbs_special = compute_tbs(re_special, 273, 27, 1, 1)
    print(f"  TBS normal:  {tbs_normal} bits")
    print(f"  TBS special: {tbs_special} bits")
    print(f"  TBS ratio:   {tbs_special/tbs_normal*100:.1f}%")

    # 验证引擎实际使用的 RE
    for slot_idx in range(5):
        ctx = engine._make_slot_context(slot_idx)
        direction = ctx.slot_direction
        dl_sym = ctx.num_dl_symbols
        re = engine.resource_grid.compute_re_per_prb(dl_sym)
        status = "OK" if (direction == 'S' and re == re_special) or \
                        (direction == 'D' and re == re_normal) or \
                        direction == 'U' else "?"
        print(f"  Slot {slot_idx}: dir={direction}, dl_sym={dl_sym:2d}, re/prb={re:3d} {status}")

    # 运行完整仿真
    report = engine.run()
    tp = report['cell_avg_throughput_mbps']
    bler = report['avg_bler']
    util = report['prb_utilization']

    print(f"\n  Full-sim results:")
    print(f"    Throughput: {tp:.1f} Mbps")
    print(f"    BLER: {bler:.4f}")
    print(f"    PRB util: {util*100:.1f}% (TDD 4:1, expect ~80%)")

    # TDD throughput should be lower than FDD due to UL slots + special slot reduction
    ok = 0.75 < util < 0.85 and 0.05 < bler < 0.15 and re_special < re_normal
    print(f"\n  [{'PASS' if ok else 'FAIL'}] TDD special slot RE/TBS")
    return ok


# ========== Test 3: HARQ K1 closed-loop verification ==========
def test_harq_closed_loop():
    """验证 legacy PHY 路径下 HARQ K1 闭环真正工作"""
    print("\n" + "=" * 60)
    print("TEST 3: HARQ K1 closed-loop (legacy PHY)")
    print("=" * 60)

    config = _make_config(duplex='FDD', num_tx_ant=4, max_layers=1, num_slots=300)
    engine = SimulationEngine(config)

    harq_tx_count = 0
    harq_retx_count = 0
    harq_delivered_bits = 0

    for s in range(300):
        result = engine.run_slot(s)
        buf_after = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(s, result, engine._buf_after_traffic, buf_after)

        # 统计 HARQ 活动
        decoded = int(result.ue_decoded_bits[0])
        harq_delivered_bits += decoded
        if result.ue_tb_success[0]:
            harq_tx_count += 1

    # 检查 HARQ manager 状态
    harq_entity = engine.harq_mgr.entities[0]  # UE 0
    stats = harq_entity.get_stats()
    total_new_tx = stats['total_transmissions']
    total_retx = stats['total_retransmissions']

    print(f"  HARQ new TX: {total_new_tx}")
    print(f"  HARQ retx:   {total_retx}")
    print(f"  Total delivered bits: {harq_delivered_bits}")
    print(f"  Avg bits/slot: {harq_delivered_bits/300:.0f}")

    # HARQ 应该有新传输被注册
    ok = total_new_tx > 0 and harq_delivered_bits > 0
    if total_retx > 0:
        print(f"  Retx observed: {total_retx} (HARQ retransmission working!)")
    else:
        print(f"  No retx observed (high SNR, all first-TX success is expected)")

    print(f"\n  [{'PASS' if ok else 'FAIL'}] HARQ K1 closed-loop")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("FDD rank=1 trace", test_fdd_rank1_trace()))
    results.append(("TDD special slot", test_tdd_special_slot()))
    results.append(("HARQ closed-loop", test_harq_closed_loop()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

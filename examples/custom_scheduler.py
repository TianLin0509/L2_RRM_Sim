"""示例: 自定义调度器 — Round Robin

展示如何用 registry 注册自定义组件，无需修改引擎代码。

用法:
    python examples/custom_scheduler.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from l2_rrm_sim.core.registry import register_scheduler
from l2_rrm_sim.scheduler.scheduler_interface import SchedulerBase
from l2_rrm_sim.core.data_types import SlotContext, ChannelState, SchedulingDecision
from l2_rrm_sim.utils.nr_utils import compute_tbs


# ── Step 1: 实现接口 ──
@register_scheduler("round_robin")   # ← 注册名
class RoundRobinScheduler(SchedulerBase):
    """最简单的轮询调度器 — 每个 slot 分给下一个 UE"""

    def __init__(self, num_ue, num_prb, num_re_per_prb=132,
                 mcs_table_index=1, **kwargs):
        self.num_ue = num_ue
        self.num_prb = num_prb
        self.num_re_per_prb = num_re_per_prb
        self.mcs_table_index = mcs_table_index
        self._rr_idx = 0
        self._t_avg = np.ones(num_ue)

    def schedule(self, slot_ctx, ue_states, channel_state,
                 achievable_rate_per_prb, ue_buffer_bytes,
                 ue_mcs, ue_rank, re_per_prb=None) -> SchedulingDecision:
        actual_re = re_per_prb or self.num_re_per_prb
        # 选一个有数据的 UE
        for _ in range(self.num_ue):
            ue = self._rr_idx % self.num_ue
            self._rr_idx += 1
            if ue_buffer_bytes[ue] > 0:
                break
        else:
            ue = 0

        # 全部 PRB 给这个 UE
        prb_assignment = np.full(self.num_prb, ue, dtype=np.int32)
        ue_num_prbs = np.zeros(self.num_ue, dtype=np.int32)
        ue_num_prbs[ue] = self.num_prb
        ue_tbs = np.zeros(self.num_ue, dtype=np.int64)
        ue_tbs[ue] = compute_tbs(actual_re, self.num_prb,
                                 int(ue_mcs[ue]), int(ue_rank[ue]),
                                 self.mcs_table_index)
        ue_num_re = np.zeros(self.num_ue, dtype=np.int64)
        ue_num_re[ue] = actual_re * self.num_prb

        return SchedulingDecision(
            prb_assignment=prb_assignment,
            ue_mcs=ue_mcs.copy(), ue_rank=ue_rank.copy(),
            ue_num_prbs=ue_num_prbs, ue_tbs_bits=ue_tbs, ue_num_re=ue_num_re,
        )

    def update_throughput_history(self, ue_throughput_bits):
        self._t_avg = 0.98 * self._t_avg + 0.02 * ue_throughput_bits


# ── Step 2: 在配置中指定 type="round_robin" ──
if __name__ == "__main__":
    from l2_rrm_sim.config.sim_config import (
        SimConfig, CarrierConfig, CellConfig, UEConfig,
        SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
        ChannelConfig, TDDConfig, CSIConfig,
    )
    from l2_rrm_sim.core.simulation_engine import SimulationEngine

    config = {
        'sim': SimConfig(num_slots=500, random_seed=42, warmup_slots=50),
        'carrier': CarrierConfig(subcarrier_spacing=30, num_prb=51),
        'cell': CellConfig(num_tx_ant=4, num_tx_ports=4, max_layers=1),
        'ue': UEConfig(num_ue=5, num_rx_ant=2, min_distance_m=50, max_distance_m=300),
        'scheduler': SchedulerConfig(type='round_robin'),  # ← 使用自定义调度器
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='statistical', scenario='uma'),
        'tdd': TDDConfig(duplex_mode='FDD'),
        'csi': CSIConfig(enabled=False),
    }

    engine = SimulationEngine(config)
    print(f"Scheduler: {type(engine.scheduler).__name__}")
    report = engine.run()
    print(f"\nRound Robin Results:")
    print(f"  Cell throughput: {report['cell_avg_throughput_mbps']:.1f} Mbps")
    print(f"  Fairness (Jain): {report['jain_fairness']:.3f}")
    for ue in range(5):
        print(f"  UE {ue}: {report['ue_avg_throughput_mbps'][ue]:.1f} Mbps")

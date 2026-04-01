"""突发流量模型

支持:
1. Poisson 包到达模型
2. ON/OFF 模型 (Markov 调制)
"""

import numpy as np
from .traffic_interface import TrafficModelBase
from ..core.data_types import SlotContext
from ..core.registry import register_traffic


@register_traffic("bursty")
class PoissonTraffic(TrafficModelBase):
    """Poisson 包到达流量模型

    每 slot 到达的包数量服从 Poisson 分布，
    每包大小固定。
    """

    def __init__(self, packet_size_bytes: int = 1500,
                 arrival_rate_pps: float = 100.0,
                 slot_duration_s: float = 0.0005,
                 num_ue: int = 20,
                 rng: np.random.Generator = None):
        """
        Args:
            packet_size_bytes: 包大小 (bytes)
            arrival_rate_pps: 包到达率 (packets/s per UE)
            slot_duration_s: slot 时长 (s)
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self.packet_size_bytes = packet_size_bytes
        self.arrival_rate_pps = arrival_rate_pps
        self.slot_duration_s = slot_duration_s
        self.num_ue = num_ue

        # 每 slot 的平均包数
        self._lambda_per_slot = arrival_rate_pps * slot_duration_s

    def generate(self, slot_ctx: SlotContext, ue_states: list):
        """生成 Poisson 包到达"""
        for ue in ue_states:
            num_packets = self._rng.poisson(self._lambda_per_slot)
            arriving_bytes = num_packets * self.packet_size_bytes
            ue.buffer_bytes += arriving_bytes


class OnOffTraffic(TrafficModelBase):
    """ON/OFF 流量模型 (Markov 调制 Poisson)

    两个状态:
    - ON: 高速率包到达
    - OFF: 无流量

    状态转移概率可配置。
    """

    def __init__(self, on_rate_pps: float = 500.0,
                 off_rate_pps: float = 0.0,
                 on_duration_ms: float = 50.0,
                 off_duration_ms: float = 100.0,
                 packet_size_bytes: int = 1500,
                 slot_duration_s: float = 0.0005,
                 num_ue: int = 20,
                 rng: np.random.Generator = None):
        """
        Args:
            on_rate_pps: ON 状态包到达率 (packets/s)
            off_rate_pps: OFF 状态包到达率 (通常为 0)
            on_duration_ms: 平均 ON 持续时间 (ms)
            off_duration_ms: 平均 OFF 持续时间 (ms)
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self.on_rate_pps = on_rate_pps
        self.off_rate_pps = off_rate_pps
        self.packet_size_bytes = packet_size_bytes
        self.slot_duration_s = slot_duration_s
        self.num_ue = num_ue

        # 状态转移概率 (per slot)
        on_duration_s = on_duration_ms / 1000.0
        off_duration_s = off_duration_ms / 1000.0
        self._p_on_to_off = slot_duration_s / on_duration_s
        self._p_off_to_on = slot_duration_s / off_duration_s

        # 每 UE 的当前状态 (True=ON)
        self._is_on = self._rng.random(num_ue) < 0.5
        # 每 slot 的 lambda
        self._lambda_on = on_rate_pps * slot_duration_s
        self._lambda_off = off_rate_pps * slot_duration_s

    def generate(self, slot_ctx: SlotContext, ue_states: list):
        """生成 ON/OFF 流量"""
        for ue_idx, ue in enumerate(ue_states):
            # 状态转移
            if self._is_on[ue_idx]:
                if self._rng.random() < self._p_on_to_off:
                    self._is_on[ue_idx] = False
            else:
                if self._rng.random() < self._p_off_to_on:
                    self._is_on[ue_idx] = True

            # 包到达
            lam = self._lambda_on if self._is_on[ue_idx] else self._lambda_off
            if lam > 0:
                num_packets = self._rng.poisson(lam)
                ue.buffer_bytes += num_packets * self.packet_size_bytes

"""Realistic 随机业务模型

所有 UE 同一业务类别 (同分布), 各自独立随机生成:
- 包到达: Poisson 过程 (可配置均值速率)
- 包大小: Lognormal 分布 (可配置均值/方差)

模拟现网场景: 视频流 / Web 浏览 / 混合业务
- 业务突发性由 Poisson 过程自然产生
- 包大小变异性由 Lognormal 分布建模
- 各 UE 独立随机, 但统计特性一致
"""

import numpy as np
from .traffic_interface import TrafficModelBase
from ..core.data_types import SlotContext


class RealisticTraffic(TrafficModelBase):
    """Realistic 随机业务模型

    每 slot 每 UE 独立:
    1. 从 Poisson(λ) 抽取本 slot 到达的包数
    2. 每包大小从 Lognormal(μ, σ) 抽取
    3. 累加到 UE buffer
    """

    def __init__(self, mean_arrival_rate_pps: float = 50.0,
                 packet_size_mean_bytes: float = 1500.0,
                 packet_size_std_bytes: float = 500.0,
                 slot_duration_s: float = 0.0005,
                 num_ue: int = 20,
                 rng: np.random.Generator = None):
        """
        Args:
            mean_arrival_rate_pps: 平均包到达率 (packets/s per UE)
                典型值: 50 (轻载), 200 (中载), 1000 (重载)
            packet_size_mean_bytes: 包大小均值 (bytes)
                典型值: 300 (VoIP), 1500 (Web/Video), 8000 (大文件)
            packet_size_std_bytes: 包大小标准差 (bytes)
                典型值: 100~500 (中等变异), >1000 (高变异)
            slot_duration_s: slot 时长 (s)
            num_ue: UE 数
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self.mean_arrival_rate = mean_arrival_rate_pps
        self.packet_size_mean = packet_size_mean_bytes
        self.packet_size_std = packet_size_std_bytes
        self.slot_duration_s = slot_duration_s
        self.num_ue = num_ue

        # Poisson λ per slot
        self._lambda_per_slot = mean_arrival_rate_pps * slot_duration_s

        # Lognormal 参数转换
        # X ~ Lognormal(μ, σ) => E[X] = exp(μ + σ²/2), Var[X] = (exp(σ²)-1)*exp(2μ+σ²)
        mean = max(packet_size_mean_bytes, 1.0)
        std = max(packet_size_std_bytes, 0.01)
        variance = std ** 2
        self._ln_sigma = np.sqrt(np.log(1.0 + variance / mean**2))
        self._ln_mu = np.log(mean) - 0.5 * self._ln_sigma**2

    def generate(self, slot_ctx: SlotContext, ue_states: list):
        """生成随机业务来包"""
        for ue in ue_states:
            # Poisson 到达
            num_packets = self._rng.poisson(self._lambda_per_slot)
            if num_packets > 0:
                # Lognormal 包大小
                sizes = self._rng.lognormal(self._ln_mu, self._ln_sigma, size=num_packets)
                arriving_bytes = int(np.sum(np.maximum(sizes, 1.0)))
                ue.buffer_bytes += arriving_bytes

    def get_offered_load_mbps(self) -> float:
        """计算理论 offered load (Mbps per UE)"""
        return (self.mean_arrival_rate * self.packet_size_mean * 8) / 1e6

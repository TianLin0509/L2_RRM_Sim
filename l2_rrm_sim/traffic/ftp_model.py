"""FTP Model 3 流量模型 (3GPP TR 36.889)

特征:
- 文件大小固定 (可配置)
- 文件到达间隔服从 Poisson 过程
- 跟踪文件完成时间用于时延 KPI
"""

import numpy as np
from dataclasses import dataclass, field
from .traffic_interface import TrafficModelBase
from ..core.data_types import SlotContext
from ..core.registry import register_traffic


@dataclass
class FileTransfer:
    """单个文件传输状态"""
    file_id: int
    ue_id: int
    total_bytes: int
    remaining_bytes: int
    arrival_slot: int
    completion_slot: int = -1

    @property
    def is_complete(self) -> bool:
        return self.remaining_bytes <= 0


@register_traffic("ftp_model3")
class FTPModel3(TrafficModelBase):
    """FTP Model 3 流量模型

    每个 UE 独立产生文件到达，到达间隔服从指数分布。
    """

    def __init__(self, file_size_bytes: int = 512000,
                 arrival_rate: float = 0.5,
                 slot_duration_s: float = 0.0005,
                 num_ue: int = 20,
                 rng: np.random.Generator = None):
        """
        Args:
            file_size_bytes: 文件大小 (bytes)
            arrival_rate: 文件到达率 (files/s per UE)
            slot_duration_s: slot 时长 (s)
            num_ue: UE 数
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self.file_size_bytes = file_size_bytes
        self.arrival_rate = arrival_rate
        self.slot_duration_s = slot_duration_s
        self.num_ue = num_ue

        # 每 slot 的到达概率 = 1 - exp(-lambda * T_slot)
        self._arrival_prob = 1.0 - np.exp(-arrival_rate * slot_duration_s)

        # 每 UE 的活跃文件传输队列
        self._active_transfers = {ue: [] for ue in range(num_ue)}
        self._completed_transfers = []
        self._file_counter = 0

        # 下一次到达的 slot (per UE)
        self._next_arrival_slot = np.zeros(num_ue, dtype=np.int64)
        self._init_arrivals()

    def _init_arrivals(self):
        """初始化每 UE 的第一个文件到达时间"""
        for ue in range(self.num_ue):
            # 第一个文件在 slot 0 或稍后到达
            inter_arrival_s = self._rng.exponential(1.0 / self.arrival_rate)
            self._next_arrival_slot[ue] = int(inter_arrival_s / self.slot_duration_s)

    def generate(self, slot_ctx: SlotContext, ue_states: list):
        """生成流量"""
        slot_idx = slot_ctx.slot_idx

        for ue_idx, ue in enumerate(ue_states):
            # 检查是否有新文件到达
            while self._next_arrival_slot[ue_idx] <= slot_idx:
                transfer = FileTransfer(
                    file_id=self._file_counter,
                    ue_id=ue_idx,
                    total_bytes=self.file_size_bytes,
                    remaining_bytes=self.file_size_bytes,
                    arrival_slot=slot_idx,
                )
                self._active_transfers[ue_idx].append(transfer)
                self._file_counter += 1

                # 下一个文件到达时间
                inter_arrival_s = self._rng.exponential(1.0 / self.arrival_rate)
                inter_arrival_slots = max(1, int(inter_arrival_s / self.slot_duration_s))
                self._next_arrival_slot[ue_idx] = slot_idx + inter_arrival_slots

            # 更新 buffer: 所有活跃文件的剩余字节之和
            ue.buffer_bytes = sum(
                t.remaining_bytes for t in self._active_transfers[ue_idx]
            )

    def dequeue_bytes(self, ue_id: int, transmitted_bytes: int,
                      current_slot: int):
        """扣减已传输字节 (FIFO)"""
        remaining = transmitted_bytes
        completed = []

        for transfer in self._active_transfers[ue_id]:
            if remaining <= 0:
                break
            deducted = min(remaining, transfer.remaining_bytes)
            transfer.remaining_bytes -= deducted
            remaining -= deducted

            if transfer.is_complete:
                transfer.completion_slot = current_slot
                completed.append(transfer)

        # 移除已完成的传输
        for t in completed:
            self._active_transfers[ue_id].remove(t)
            self._completed_transfers.append(t)

    def get_completed_transfers(self) -> list:
        """获取所有已完成的文件传输"""
        return self._completed_transfers.copy()

    def get_file_latency_stats(self, slot_duration_s: float = None) -> dict:
        """获取文件传输时延统计"""
        if not self._completed_transfers:
            return {'mean_ms': 0, 'p50_ms': 0, 'p95_ms': 0, 'p99_ms': 0, 'count': 0}

        if slot_duration_s is None:
            slot_duration_s = self.slot_duration_s

        latencies_ms = []
        for t in self._completed_transfers:
            latency_slots = t.completion_slot - t.arrival_slot
            latencies_ms.append(latency_slots * slot_duration_s * 1000.0)

        latencies = np.array(latencies_ms)
        return {
            'mean_ms': float(np.mean(latencies)),
            'p50_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'count': len(latencies),
        }

"""UE 缓冲区管理"""

import numpy as np
from ..core.data_types import UEState


class BufferManager:
    """UE 发送缓冲区管理器

    跟踪每 UE 的待发送字节数，支持到达和消耗。
    """

    def __init__(self, num_ue: int):
        self.num_ue = num_ue

    def dequeue(self, ue_states: list, decoded_bits: np.ndarray):
        """从缓冲区扣减已传输的字节

        Args:
            ue_states: UE 状态列表
            decoded_bits: (num_ue,) 成功解码的比特数
        """
        for ue_idx, ue in enumerate(ue_states):
            transmitted_bytes = int(decoded_bits[ue_idx]) // 8
            ue.buffer_bytes = max(0, ue.buffer_bytes - transmitted_bytes)

    def enqueue(self, ue_states: list, arriving_bytes: np.ndarray):
        """向缓冲区添加到达的字节

        Args:
            ue_states: UE 状态列表
            arriving_bytes: (num_ue,) 到达的字节数
        """
        for ue_idx, ue in enumerate(ue_states):
            ue.buffer_bytes += int(arriving_bytes[ue_idx])

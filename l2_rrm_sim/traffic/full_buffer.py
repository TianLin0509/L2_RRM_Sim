"""全缓冲流量模型

UE 始终有无限数据待发送。
"""

from .traffic_interface import TrafficModelBase
from ..core.data_types import SlotContext


INFINITE_BUFFER = 10**9  # 1 GB


class FullBufferTraffic(TrafficModelBase):
    """全缓冲流量模型"""

    def generate(self, slot_ctx: SlotContext, ue_states: list):
        for ue in ue_states:
            ue.buffer_bytes = INFINITE_BUFFER

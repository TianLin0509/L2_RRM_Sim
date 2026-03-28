"""流量模型抽象接口"""

from abc import ABC, abstractmethod
from ..core.data_types import SlotContext, UEState


class TrafficModelBase(ABC):
    """流量模型基类"""

    @abstractmethod
    def generate(self, slot_ctx: SlotContext, ue_states: list):
        """生成流量 (更新 UE 的 buffer_bytes)

        Args:
            slot_ctx: 当前 slot 上下文
            ue_states: UE 状态列表 (原地修改 buffer_bytes)
        """
        pass

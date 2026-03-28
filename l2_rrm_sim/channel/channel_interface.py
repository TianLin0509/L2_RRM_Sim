"""信道模型抽象接口"""

from abc import ABC, abstractmethod
import numpy as np
from ..core.data_types import SlotContext, UEState, ChannelState
from ..config.sim_config import CellConfig, CarrierConfig


class ChannelModelBase(ABC):
    """信道模型基类

    所有信道模型 (统计信道, 射线追踪等) 必须实现此接口。
    """

    @abstractmethod
    def initialize(self, cell_config: CellConfig,
                   carrier_config: CarrierConfig,
                   ue_states: list):
        """初始化信道模型

        Args:
            cell_config: 小区配置
            carrier_config: 载波配置
            ue_states: UE 状态列表
        """
        pass

    @abstractmethod
    def update(self, slot_ctx: SlotContext,
               ue_states: list) -> ChannelState:
        """更新信道状态 (每 TTI 调用)

        Args:
            slot_ctx: 当前 slot 上下文
            ue_states: UE 状态列表

        Returns:
            ChannelState: 更新后的信道状态
        """
        pass

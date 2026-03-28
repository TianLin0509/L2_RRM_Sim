"""调度器抽象接口"""

from abc import ABC, abstractmethod
import numpy as np
from ..core.data_types import SlotContext, UEState, ChannelState, SchedulingDecision


class SchedulerBase(ABC):
    """调度器基类

    所有调度器 (PF, RR, MaxCIR, MU-MIMO) 必须实现此接口。
    """

    @abstractmethod
    def schedule(self, slot_ctx: SlotContext,
                 ue_states: list,
                 channel_state: ChannelState,
                 achievable_rate_per_prb: np.ndarray,
                 ue_buffer_bytes: np.ndarray,
                 ue_mcs: np.ndarray,
                 ue_rank: np.ndarray) -> SchedulingDecision:
        """执行调度

        Args:
            slot_ctx: 当前 slot 上下文
            ue_states: UE 状态列表
            channel_state: 信道状态
            achievable_rate_per_prb: (num_ue, num_prb) 每 UE 每 PRB 可达速率 [bits]
            ue_buffer_bytes: (num_ue,) 每 UE 缓冲区字节数
            ue_mcs: (num_ue,) MCS 索引
            ue_rank: (num_ue,) rank

        Returns:
            SchedulingDecision
        """
        pass

    @abstractmethod
    def update_throughput_history(self, ue_throughput_bits: np.ndarray):
        """更新吞吐量历史 (用于 PF metric)

        Args:
            ue_throughput_bits: (num_ue,) 本 TTI 各 UE 解码比特数
        """
        pass

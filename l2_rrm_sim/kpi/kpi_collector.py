"""KPI 数据收集器

逐 TTI 记录仿真数据，支持 warmup 过滤。
"""

import numpy as np
from ..core.data_types import SlotResult


class KPICollector:
    """KPI 收集器"""

    def __init__(self, num_slots: int, num_ue: int,
                 warmup_slots: int = 0):
        self.num_slots = num_slots
        self.num_ue = num_ue
        self.warmup_slots = warmup_slots

        # 预分配数组
        self.ue_throughput_bits = np.zeros((num_slots, num_ue), dtype=np.float64)
        self.ue_throughput_inst = np.zeros((num_slots, num_ue), dtype=np.float64)
        self.ue_bler = np.zeros((num_slots, num_ue), dtype=np.float64)
        self.ue_mcs = np.zeros((num_slots, num_ue), dtype=np.int32)
        self.ue_rank = np.zeros((num_slots, num_ue), dtype=np.int32)
        self.ue_sinr_eff_db = np.zeros((num_slots, num_ue), dtype=np.float64)
        self.ue_tb_success = np.zeros((num_slots, num_ue), dtype=bool)
        self.ue_num_prbs = np.zeros((num_slots, num_ue), dtype=np.int32)
        self.ue_buffer_bytes = np.zeros((num_slots, num_ue), dtype=np.int64)
        self.cell_throughput_bits = np.zeros(num_slots, dtype=np.float64)

        self._collected_slots = 0

    def collect(self, slot_idx: int, result: SlotResult,
                ue_buffer_bytes: np.ndarray = None):
        """收集单 slot 数据

        Args:
            ue_buffer_bytes: (num_ue,) 本 slot 流量生成后、调度前的 buffer 状态
        """
        self.ue_throughput_bits[slot_idx] = result.ue_decoded_bits
        self.ue_throughput_inst[slot_idx] = result.ue_throughput_inst
        self.ue_bler[slot_idx] = result.ue_bler
        self.ue_mcs[slot_idx] = result.ue_mcs
        self.ue_rank[slot_idx] = result.ue_rank
        self.ue_sinr_eff_db[slot_idx] = result.ue_sinr_eff_db
        self.ue_tb_success[slot_idx] = result.ue_tb_success
        if result.scheduling_decision is not None:
            self.ue_num_prbs[slot_idx] = result.scheduling_decision.ue_num_prbs
        if ue_buffer_bytes is not None:
            self.ue_buffer_bytes[slot_idx] = ue_buffer_bytes
        self.cell_throughput_bits[slot_idx] = np.sum(result.ue_decoded_bits)
        self._collected_slots = slot_idx + 1

    def get_valid_range(self) -> slice:
        """返回有效数据范围 (排除 warmup)"""
        start = min(self.warmup_slots, self._collected_slots)
        return slice(start, self._collected_slots)

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
        self.ue_scheduled_bits = np.zeros((num_slots, num_ue), dtype=np.float64)
        self.ue_num_re = np.zeros((num_slots, num_ue), dtype=np.int32)
        self.ue_buffer_bytes = np.zeros((num_slots, num_ue), dtype=np.int64)
        self.ue_buffer_after = np.zeros((num_slots, num_ue), dtype=np.int64)
        self.cell_throughput_bits = np.zeros(num_slots, dtype=np.float64)
        self.cell_scheduled_bits = np.zeros(num_slots, dtype=np.float64)
        self.slot_num_dl_symbols = np.zeros(num_slots, dtype=np.int32)
        self.slot_direction = np.full(num_slots, '', dtype='<U1')

        self._collected_slots = 0

    def collect(self, slot_idx: int, result: SlotResult,
                ue_buffer_bytes: np.ndarray = None,
                ue_buffer_after: np.ndarray = None):
        """收集单 slot 数据

        Args:
            ue_buffer_bytes: (num_ue,) 流量生成后、调度前的 buffer
            ue_buffer_after: (num_ue,) 传输后的 buffer (可选, 用于精确体验速率)
        """
        self.ue_throughput_bits[slot_idx] = result.ue_decoded_bits
        self.ue_throughput_inst[slot_idx] = result.ue_throughput_inst
        self.ue_bler[slot_idx] = result.ue_bler
        self.ue_mcs[slot_idx] = result.ue_mcs
        self.ue_rank[slot_idx] = result.ue_rank
        self.ue_sinr_eff_db[slot_idx] = result.ue_sinr_eff_db
        self.ue_tb_success[slot_idx] = result.ue_tb_success
        self.slot_num_dl_symbols[slot_idx] = result.num_dl_symbols
        self.slot_direction[slot_idx] = result.slot_direction
        if result.scheduling_decision is not None:
            self.ue_num_prbs[slot_idx] = result.scheduling_decision.ue_num_prbs
            self.ue_scheduled_bits[slot_idx] = result.scheduling_decision.ue_tbs_bits
            self.ue_num_re[slot_idx] = result.scheduling_decision.ue_num_re
        if ue_buffer_bytes is not None:
            self.ue_buffer_bytes[slot_idx] = ue_buffer_bytes
        if ue_buffer_after is not None:
            self.ue_buffer_after[slot_idx] = ue_buffer_after
        self.cell_throughput_bits[slot_idx] = np.sum(result.ue_decoded_bits)
        self.cell_scheduled_bits[slot_idx] = np.sum(self.ue_scheduled_bits[slot_idx])
        self._collected_slots = slot_idx + 1

    def build_slot_trace(self, slot_slice: slice = None) -> dict:
        """Build an auditable per-slot trace view."""
        if slot_slice is None:
            slot_slice = self.get_valid_range()
        slot_indices = np.arange(slot_slice.start, slot_slice.stop)
        return {
            'slot_idx': slot_indices,
            'slot_direction': self.slot_direction[slot_slice].copy(),
            'num_dl_symbols': self.slot_num_dl_symbols[slot_slice].copy(),
            'ue_num_prbs': self.ue_num_prbs[slot_slice].copy(),
            'ue_num_re': self.ue_num_re[slot_slice].copy(),
            'ue_scheduled_bits': self.ue_scheduled_bits[slot_slice].copy(),
            'ue_delivered_bits': self.ue_throughput_bits[slot_slice].copy(),
            'ue_mcs': self.ue_mcs[slot_slice].copy(),
            'ue_rank': self.ue_rank[slot_slice].copy(),
            'ue_sinr_eff_db': self.ue_sinr_eff_db[slot_slice].copy(),
            'ue_tb_success': self.ue_tb_success[slot_slice].copy(),
            'cell_scheduled_bits': self.cell_scheduled_bits[slot_slice].copy(),
            'cell_delivered_bits': self.cell_throughput_bits[slot_slice].copy(),
        }

    def get_valid_range(self) -> slice:
        """返回有效数据范围 (排除 warmup)"""
        start = min(self.warmup_slots, self._collected_slots)
        return slice(start, self._collected_slots)

"""HARQ 实体与进程管理

5G NR HARQ:
- 最多 16 个 HARQ 进程 (FDD)
- RV (Redundancy Version) 循环: 0, 2, 3, 1
- 最大重传次数可配置
- Chase Combining: 重传时 SINR 累加
"""

import numpy as np
from dataclasses import dataclass


# RV 循环顺序 (TS 38.214)
RV_SEQUENCE = [0, 2, 3, 1]


@dataclass
class HARQProcess:
    """单个 HARQ 进程"""
    process_id: int
    is_active: bool = False
    ndi: int = 0                   # New Data Indicator (0 or 1)
    rv_idx: int = 0                # 当前 RV 在 RV_SEQUENCE 中的索引
    mcs_index: int = 0             # 初传的 MCS
    num_prbs: int = 0              # 初传分配的 PRB 数
    num_layers: int = 1            # 初传的层数
    tbs_bits: int = 0              # Transport Block Size
    num_retx: int = 0              # 已重传次数
    accumulated_sinr: float = 0.0  # Chase Combining 累积 SINR (linear)
    initial_slot: int = -1         # 初传 slot

    @property
    def rv(self) -> int:
        """当前 Redundancy Version"""
        return RV_SEQUENCE[self.rv_idx % len(RV_SEQUENCE)]

    def new_transmission(self, mcs: int, num_prbs: int, num_layers: int,
                         tbs: int, slot_idx: int):
        """开始新传输"""
        self.is_active = True
        self.ndi = 1 - self.ndi  # 翻转 NDI
        self.rv_idx = 0
        self.mcs_index = mcs
        self.num_prbs = num_prbs
        self.num_layers = num_layers
        self.tbs_bits = tbs
        self.num_retx = 0
        self.accumulated_sinr = 0.0
        self.initial_slot = slot_idx

    def prepare_retransmission(self):
        """准备重传"""
        self.rv_idx += 1
        self.num_retx += 1

    def release(self):
        """释放进程"""
        self.is_active = False
        self.accumulated_sinr = 0.0


class HARQEntity:
    """HARQ 实体 (per UE)

    管理多个 HARQ 进程，处理 ACK/NACK 反馈，
    维护重传队列。
    """

    def __init__(self, num_processes: int = 16,
                 max_retx: int = 4,
                 combining_gain_db: float = 3.0):
        """
        Args:
            num_processes: HARQ 进程数 (5G NR: 最多 16)
            max_retx: 最大重传次数
            combining_gain_db: Chase Combining 每次重传增益 (dB)
        """
        self.num_processes = num_processes
        self.max_retx = max_retx
        self.combining_gain_db = combining_gain_db

        self.processes = [
            HARQProcess(process_id=i)
            for i in range(num_processes)
        ]

        # 等待重传的进程 ID 队列
        self._retx_queue = []

        # 统计
        self.total_transmissions = 0
        self.total_retransmissions = 0
        self.total_acks = 0
        self.total_nacks = 0
        self.total_max_retx_reached = 0

    def has_retransmission(self) -> bool:
        """是否有等待重传的进程"""
        return len(self._retx_queue) > 0

    def get_free_process(self) -> HARQProcess:
        """获取空闲的 HARQ 进程"""
        for proc in self.processes:
            if not proc.is_active:
                return proc
        return None

    def get_next_retx_process(self) -> HARQProcess:
        """获取下一个需要重传的进程 (FIFO)"""
        while self._retx_queue:
            pid = self._retx_queue.pop(0)
            proc = self.processes[pid]
            if proc.is_active:
                return proc
        return None

    def start_new_transmission(self, mcs: int, num_prbs: int,
                               num_layers: int, tbs: int,
                               slot_idx: int) -> HARQProcess:
        """开始新传输

        Returns:
            分配的 HARQ 进程，或 None (无空闲进程)
        """
        proc = self.get_free_process()
        if proc is None:
            return None

        proc.new_transmission(mcs, num_prbs, num_layers, tbs, slot_idx)
        self.total_transmissions += 1
        return proc

    def process_feedback(self, process_id: int, is_ack: bool,
                         sinr_eff_linear: float = 0.0) -> dict:
        """处理 HARQ 反馈

        Args:
            process_id: HARQ 进程 ID
            is_ack: True=ACK, False=NACK
            sinr_eff_linear: 本次传输的有效 SINR (linear, 用于 combining)

        Returns:
            dict: {decoded_bits, is_final_ack, retx_needed}
        """
        proc = self.processes[process_id]
        if not proc.is_active:
            return {'decoded_bits': 0, 'is_final_ack': False, 'retx_needed': False}

        # 累加 SINR (Chase Combining)
        proc.accumulated_sinr += sinr_eff_linear

        if is_ack:
            self.total_acks += 1
            decoded_bits = proc.tbs_bits
            proc.release()
            return {'decoded_bits': decoded_bits, 'is_final_ack': True,
                    'retx_needed': False}
        else:
            self.total_nacks += 1
            if proc.num_retx >= self.max_retx - 1:
                # 达到最大重传次数，放弃
                self.total_max_retx_reached += 1
                proc.release()
                return {'decoded_bits': 0, 'is_final_ack': False,
                        'retx_needed': False}
            else:
                # 安排重传
                proc.prepare_retransmission()
                self._retx_queue.append(process_id)
                self.total_retransmissions += 1
                return {'decoded_bits': 0, 'is_final_ack': False,
                        'retx_needed': True}

    def get_combining_sinr(self, process_id: int,
                           current_sinr_linear: float) -> float:
        """获取 Chase Combining 后的等效 SINR

        简化模型: SINR_combined = SINR_accumulated + SINR_current
        (MRC combining 近似)

        Returns:
            Combined SINR (linear)
        """
        proc = self.processes[process_id]
        return proc.accumulated_sinr + current_sinr_linear

    def get_stats(self) -> dict:
        """获取 HARQ 统计"""
        active_count = sum(1 for p in self.processes if p.is_active)
        return {
            'active_processes': active_count,
            'retx_queue_length': len(self._retx_queue),
            'total_transmissions': self.total_transmissions,
            'total_retransmissions': self.total_retransmissions,
            'total_acks': self.total_acks,
            'total_nacks': self.total_nacks,
            'total_max_retx_reached': self.total_max_retx_reached,
            'retx_rate': (self.total_retransmissions / max(self.total_transmissions, 1)),
        }

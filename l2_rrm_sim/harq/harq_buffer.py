"""HARQ 重传调度管理

支持:
- 重传优先于新传输
- K1 反馈时延 (HARQ-ACK 在 tx_slot + K1 的 UL slot 到达)
- TDD-aware: ACK 只能在 UL/Special slot 发送
"""

import numpy as np
from collections import defaultdict
from .harq_entity import HARQEntity


class HARQManager:
    """HARQ 管理器 (所有 UE)

    支持 K1 延迟反馈: 反馈不立即处理, 而是排队到 feedback_slot 才交付。
    """

    def __init__(self, num_ue: int,
                 num_processes: int = 16,
                 max_retx: int = 4,
                 combining_gain_db: float = 3.0):
        self.num_ue = num_ue
        self.entities = [
            HARQEntity(num_processes, max_retx, combining_gain_db)
            for _ in range(num_ue)
        ]

        # K1 延迟反馈队列: {delivery_slot: [(ue_id, pid, is_ack, sinr, tbs), ...]}
        self._pending_feedback = defaultdict(list)
        # 本 slot 交付的 decoded_bits (累积到 _finalize_slot)
        self._delivered_decoded_bits = np.zeros(num_ue, dtype=np.int64)

    def has_any_retransmission(self) -> np.ndarray:
        """(num_ue,) bool: 各 UE 是否有待重传"""
        return np.array([e.has_retransmission() for e in self.entities])

    def peek_retx_info(self, ue_id: int) -> dict:
        """查看 UE 的重传信息（只读，不消费队列）

        Returns:
            dict: {has_retx, mcs, num_prbs, num_layers, tbs, process_id}
            或 None
        """
        entity = self.entities[ue_id]
        # 只读队列头，不 pop
        while entity._retx_queue:
            pid = entity._retx_queue[0]
            proc = entity.processes[pid]
            if proc.is_active:
                return {
                    'has_retx': True,
                    'process_id': proc.process_id,
                    'mcs': proc.mcs_index,
                    'num_prbs': proc.num_prbs,
                    'num_layers': proc.num_layers,
                    'tbs': proc.tbs_bits,
                    'rv': proc.rv,
                    'num_retx': proc.num_retx,
                    'accumulated_sinr': proc.accumulated_sinr,
                }
            else:
                # 进程已失效，清理并继续
                entity._retx_queue.pop(0)
        return None

    def consume_retx(self, ue_id: int):
        """消费重传队列头（调度确认后调用）"""
        entity = self.entities[ue_id]
        entity.get_next_retx_process()  # pop 并丢弃返回值（已通过 peek 获取信息）

    def get_retx_info(self, ue_id: int) -> dict:
        """获取 UE 的重传信息并消费队列（兼容旧接口）

        Returns:
            dict: {has_retx, mcs, num_prbs, num_layers, tbs, process_id}
            或 None
        """
        entity = self.entities[ue_id]
        proc = entity.get_next_retx_process()
        if proc is None:
            return None

        return {
            'has_retx': True,
            'process_id': proc.process_id,
            'mcs': proc.mcs_index,
            'num_prbs': proc.num_prbs,
            'num_layers': proc.num_layers,
            'tbs': proc.tbs_bits,
            'rv': proc.rv,
            'num_retx': proc.num_retx,
            'accumulated_sinr': proc.accumulated_sinr,
        }

    def start_new_tx(self, ue_id: int, mcs: int, num_prbs: int,
                     num_layers: int, tbs: int, slot_idx: int) -> int:
        """开始新传输

        Returns:
            HARQ 进程 ID, 或 -1 (无空闲进程)
        """
        proc = self.entities[ue_id].start_new_transmission(
            mcs, num_prbs, num_layers, tbs, slot_idx
        )
        return proc.process_id if proc else -1

    def process_feedback(self, ue_id: int, process_id: int,
                         is_ack: bool, sinr_eff_linear: float = 0.0) -> dict:
        """处理反馈"""
        return self.entities[ue_id].process_feedback(
            process_id, is_ack, sinr_eff_linear
        )

    def get_combining_sinr(self, ue_id: int, process_id: int,
                           current_sinr: float) -> float:
        """获取 combining 后的 SINR"""
        return self.entities[ue_id].get_combining_sinr(
            process_id, current_sinr
        )

    def queue_feedback(self, delivery_slot: int, ue_id: int,
                       process_id: int, is_ack: bool,
                       sinr_eff_linear: float = 0.0,
                       tbs_bits: int = 0):
        """将反馈排入延迟队列 (K1 delay)

        Args:
            delivery_slot: 反馈到达的 slot
            ue_id: UE ID
            process_id: HARQ 进程 ID
            is_ack: True=ACK
            sinr_eff_linear: 有效 SINR
            tbs_bits: TBS (ACK 时用于 decoded_bits)
        """
        self._pending_feedback[delivery_slot].append(
            (ue_id, process_id, is_ack, sinr_eff_linear, tbs_bits)
        )

    def deliver_feedback(self, current_slot: int) -> list:
        """交付到期的反馈 (每 slot 开头调用)

        Returns:
            delivered: [(ue_id, result_dict), ...]
        """
        self._delivered_decoded_bits[:] = 0
        delivered = []

        # 收集所有到期的反馈 (包括更早的, 以防 UL slot 间隔)
        slots_to_deliver = [s for s in self._pending_feedback if s <= current_slot]
        for slot in sorted(slots_to_deliver):
            for (ue_id, pid, is_ack, sinr, tbs) in self._pending_feedback.pop(slot):
                result = self.process_feedback(ue_id, pid, is_ack, sinr)
                if is_ack:
                    self._delivered_decoded_bits[ue_id] += tbs
                delivered.append((ue_id, result))

        return delivered

    def get_delivered_decoded_bits(self) -> np.ndarray:
        """获取本 slot 因 K1 到期而确认的 decoded bits"""
        return self._delivered_decoded_bits.copy()

    def get_all_stats(self) -> dict:
        """获取所有 UE 的 HARQ 统计汇总"""
        total = {
            'total_transmissions': 0,
            'total_retransmissions': 0,
            'total_acks': 0,
            'total_nacks': 0,
            'total_max_retx_reached': 0,
        }
        for e in self.entities:
            stats = e.get_stats()
            for k in total:
                total[k] += stats[k]

        total['retx_rate'] = (
            total['total_retransmissions']
            / max(total['total_transmissions'], 1)
        )
        total['effective_bler'] = (
            total['total_max_retx_reached']
            / max(total['total_transmissions'], 1)
        )
        return total

"""HARQ 重传调度管理

重传优先级:
- HARQ 重传优先于新传输
- 重传使用原始 MCS 和分配参数
"""

import numpy as np
from .harq_entity import HARQEntity


class HARQManager:
    """HARQ 管理器 (所有 UE)

    协调各 UE 的 HARQ 实体，提供统一接口。
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

    def has_any_retransmission(self) -> np.ndarray:
        """(num_ue,) bool: 各 UE 是否有待重传"""
        return np.array([e.has_retransmission() for e in self.entities])

    def get_retx_info(self, ue_id: int) -> dict:
        """获取 UE 的重传信息

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

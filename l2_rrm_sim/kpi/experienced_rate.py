"""掐头去尾体验速率计算

定义:
  一次传输 session = 从业务来包到 buffer 清空的过程

  掐头: 从首次被调度的 TTI 开始计时 (去掉来包后等待调度的时间)
  去尾: 去掉最后一个调度 TTI (防止尾包太小造成统计偏差)

  统计区间: [首次调度 TTI, buffer 清空前最后一个调度 TTI)
  - 中间即使有 TTI 未被调度，时间也要算入 (反映调度器公平性)

  用户体验速率 = Σ(去尾 session 传输 bits) / Σ(掐头去尾 session 时间)
  小区体验速率 = Σ_all_ue(去尾 bits) / Σ_all_ue(掐头去尾时间)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SessionRecord:
    """一次传输 session 的记录"""
    ue_id: int
    # 来包 slot (buffer 从 0 变为 > 0 的那个 TTI)
    arrival_slot: int
    # 首次调度 slot (掐头后的起点)
    first_sched_slot: int = -1
    # buffer 清空 slot
    clear_slot: int = -1
    # 每个被调度 TTI 的 (slot_idx, decoded_bits)
    sched_records: list = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.clear_slot >= 0 and self.first_sched_slot >= 0

    @property
    def num_sched_ttis(self) -> int:
        return len(self.sched_records)


class ExperiencedRateCalculator:
    """掐头去尾体验速率计算器

    逐 slot 分析每个 UE 的 session 状态，
    在 session 结束 (buffer 清空) 时记录完整 session。
    """

    def __init__(self, num_ue: int, slot_duration_s: float):
        self.num_ue = num_ue
        self.slot_duration_s = slot_duration_s

        # 每 UE 当前活跃的 session (None = 无活跃 session)
        self._active_sessions = [None] * num_ue
        # 已完成的 session 列表
        self.completed_sessions = []

    def process_slot(self, slot_idx: int,
                     ue_buffer_before: np.ndarray,
                     ue_decoded_bits: np.ndarray,
                     ue_num_prbs: np.ndarray,
                     ue_buffer_after: np.ndarray):
        """处理单个 slot 的数据

        Args:
            slot_idx: slot 索引
            ue_buffer_before: (num_ue,) 流量生成后、调度前的 buffer
            ue_decoded_bits: (num_ue,) 本 slot 解码比特数
            ue_num_prbs: (num_ue,) 本 slot 分配的 PRB 数
            ue_buffer_after: (num_ue,) 调度传输后的 buffer
        """
        for ue in range(self.num_ue):
            buf_before = int(ue_buffer_before[ue])
            buf_after = int(ue_buffer_after[ue])
            decoded = int(ue_decoded_bits[ue])
            is_scheduled = int(ue_num_prbs[ue]) > 0
            session = self._active_sessions[ue]

            # --- Session 状态机 ---

            if session is None:
                # 当前无活跃 session
                if buf_before > 0:
                    # 有新来包 → 创建 session
                    session = SessionRecord(
                        ue_id=ue,
                        arrival_slot=slot_idx,
                    )
                    self._active_sessions[ue] = session

            if session is not None:
                # 有活跃 session
                if is_scheduled and session.first_sched_slot < 0:
                    # 首次被调度 (掐头后的起点)
                    session.first_sched_slot = slot_idx

                if is_scheduled and session.first_sched_slot >= 0:
                    # 记录本次调度
                    session.sched_records.append((slot_idx, decoded))

                # 检查 buffer 是否清空
                if buf_after <= 0 and session.first_sched_slot >= 0:
                    # Session 结束
                    session.clear_slot = slot_idx
                    self.completed_sessions.append(session)
                    self._active_sessions[ue] = None

    def compute_experienced_rate(self) -> dict:
        """计算掐头去尾体验速率

        Returns:
            dict with:
                ue_experienced_rate_mbps: (num_ue,) 每 UE 体验速率
                cell_experienced_rate_mbps: 小区体验速率
                num_sessions_per_ue: (num_ue,) 每 UE 完成的 session 数
                session_details: list of per-session stats
        """
        ue_total_bits = np.zeros(self.num_ue)
        ue_total_time_s = np.zeros(self.num_ue)
        ue_session_count = np.zeros(self.num_ue, dtype=np.int32)
        session_details = []

        for session in self.completed_sessions:
            ue = session.ue_id
            n_sched = session.num_sched_ttis

            if n_sched < 2:
                # 只有 1 个或 0 个调度 TTI，去尾后无有效数据
                continue

            # 掐头: 从 first_sched_slot 开始
            # 去尾: 去掉最后一个调度 TTI
            # 统计时间: first_sched_slot 到 倒数第二个调度 TTI (含)
            #          即 [first_sched_slot, last_sched_slot_before_tail]
            #          但时间要连续统计到 buffer 清空前

            # 去尾: 排除最后一个调度 TTI
            trimmed_records = session.sched_records[:-1]
            last_included_slot = trimmed_records[-1][0]

            # 去尾后的总传输比特
            trimmed_bits = sum(bits for _, bits in trimmed_records)

            # 掐头去尾的时间:
            # 从 first_sched_slot 到 last_included_slot (含), 连续计时
            # 即使中间有未调度的 slot 也算时间
            trimmed_duration_slots = last_included_slot - session.first_sched_slot + 1
            trimmed_time_s = trimmed_duration_slots * self.slot_duration_s

            if trimmed_time_s <= 0:
                continue

            ue_total_bits[ue] += trimmed_bits
            ue_total_time_s[ue] += trimmed_time_s
            ue_session_count[ue] += 1

            session_details.append({
                'ue_id': ue,
                'arrival_slot': session.arrival_slot,
                'first_sched_slot': session.first_sched_slot,
                'clear_slot': session.clear_slot,
                'total_sched_ttis': n_sched,
                'trimmed_bits': trimmed_bits,
                'trimmed_duration_slots': trimmed_duration_slots,
                'trimmed_rate_mbps': trimmed_bits / trimmed_time_s / 1e6,
                'wait_slots': session.first_sched_slot - session.arrival_slot,
            })

        # 每 UE 体验速率
        ue_exp_rate = np.zeros(self.num_ue)
        for ue in range(self.num_ue):
            if ue_total_time_s[ue] > 0:
                ue_exp_rate[ue] = ue_total_bits[ue] / ue_total_time_s[ue] / 1e6

        # 小区体验速率: Σ bits / Σ time
        total_bits_all = np.sum(ue_total_bits)
        total_time_all = np.sum(ue_total_time_s)
        cell_exp_rate = (total_bits_all / total_time_all / 1e6
                         if total_time_all > 0 else 0.0)

        # 小区边缘体验速率: 5th percentile (只看有 session 的 UE)
        active_ue_rates = ue_exp_rate[ue_session_count > 0]
        cell_edge_exp_rate = (float(np.percentile(active_ue_rates, 5))
                               if len(active_ue_rates) > 0 else 0.0)

        return {
            'ue_experienced_rate_mbps': ue_exp_rate,
            'cell_experienced_rate_mbps': float(cell_exp_rate),
            'cell_edge_experienced_rate_mbps': cell_edge_exp_rate,
            'num_sessions_per_ue': ue_session_count,
            'total_completed_sessions': len(self.completed_sessions),
            'total_valid_sessions': len(session_details),
            'session_details': session_details,
        }

    def print_summary(self, result: dict = None):
        """打印体验速率汇总"""
        if result is None:
            result = self.compute_experienced_rate()

        print("-" * 65)
        print("  [掐头去尾体验速率]")
        print(f"    完成 sessions: {result['total_completed_sessions']}, "
              f"有效 sessions: {result['total_valid_sessions']}")
        print(f"    小区体验速率:       {result['cell_experienced_rate_mbps']:.2f} Mbps")
        print(f"    小区边缘体验速率:   {result['cell_edge_experienced_rate_mbps']:.2f} Mbps")

        ue_rates = result['ue_experienced_rate_mbps']
        active = ue_rates[result['num_sessions_per_ue'] > 0]
        if len(active) > 0:
            print(f"    UE 体验速率分布 (Mbps):")
            print(f"      Min: {np.min(active):.2f},  5%: {np.percentile(active, 5):.2f}, "
                  f" 50%: {np.median(active):.2f},  95%: {np.percentile(active, 95):.2f}, "
                  f" Max: {np.max(active):.2f}")

        if result['session_details']:
            wait_slots = [s['wait_slots'] for s in result['session_details']]
            print(f"    等待调度时间: avg={np.mean(wait_slots):.1f} slots, "
                  f"max={np.max(wait_slots)} slots")

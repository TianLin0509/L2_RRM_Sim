"""TDD 时隙格式配置 (3GPP TS 38.213 Section 11.1)

典型 TDD 模式 (sub-6 GHz):
  DDDSU   — 5ms 周期, 3DL + 1Special + 1UL (60% DL)
  DDDSUDDSUU — 10ms 周期, 7DL + 1S + 2UL (70% DL)
  DDSUU   — 5ms 周期, 2DL + 1S + 2UL (40% DL)
  DDDDDDDSUU — 10ms 周期, 7DL + 1S + 2UL (70% DL)

Special slot: 部分符号用于 DL, 部分 GP, 部分 UL
  例: 10DL + 2GP + 2UL (of 14 symbols)
"""

import numpy as np


class SlotDirection:
    DL = 'D'
    UL = 'U'
    SPECIAL = 'S'


class TDDConfig:
    """TDD 时隙格式

    解析 pattern 字符串, 提供 per-slot 查询。
    """

    def __init__(self, pattern: str = "DDDSU",
                 special_dl_symbols: int = 10,
                 special_gp_symbols: int = 2,
                 special_ul_symbols: int = 2):
        """
        Args:
            pattern: 时隙模式字符串, 如 "DDDSU", "DDDDDDDSUU"
                D=下行, U=上行, S=特殊时隙
            special_dl_symbols: Special slot 中 DL 符号数
            special_gp_symbols: GP 符号数
            special_ul_symbols: UL 符号数
        """
        self.pattern = pattern.upper()
        self.period = len(self.pattern)
        self.special_dl_symbols = special_dl_symbols
        self.special_gp_symbols = special_gp_symbols
        self.special_ul_symbols = special_ul_symbols

        # 解析 pattern
        self._directions = []
        for ch in self.pattern:
            if ch == 'D':
                self._directions.append(SlotDirection.DL)
            elif ch == 'U':
                self._directions.append(SlotDirection.UL)
            elif ch == 'S':
                self._directions.append(SlotDirection.SPECIAL)
            else:
                raise ValueError(f"Invalid TDD pattern char: '{ch}'")

        # 统计
        self.num_dl_slots = sum(1 for d in self._directions if d == SlotDirection.DL)
        self.num_ul_slots = sum(1 for d in self._directions if d == SlotDirection.UL)
        self.num_special_slots = sum(1 for d in self._directions if d == SlotDirection.SPECIAL)

        # DL 比例 (Special slot 按 DL 符号比例折算)
        dl_equivalent = self.num_dl_slots + self.num_special_slots * (special_dl_symbols / 14.0)
        self.dl_ratio = dl_equivalent / self.period

    def get_slot_direction(self, slot_idx: int) -> str:
        """获取 slot 的传输方向"""
        return self._directions[slot_idx % self.period]

    def is_dl_schedulable(self, slot_idx: int) -> bool:
        """该 slot 是否可调度 DL (D 或 S)"""
        d = self.get_slot_direction(slot_idx)
        return d in (SlotDirection.DL, SlotDirection.SPECIAL)

    def is_ul_slot(self, slot_idx: int) -> bool:
        """是否为 UL slot (U 或 S 的 UL 部分)"""
        d = self.get_slot_direction(slot_idx)
        return d in (SlotDirection.UL, SlotDirection.SPECIAL)

    def get_dl_symbols(self, slot_idx: int) -> int:
        """获取该 slot 可用的 DL OFDM 符号数"""
        d = self.get_slot_direction(slot_idx)
        if d == SlotDirection.DL:
            return 14  # Normal CP
        elif d == SlotDirection.SPECIAL:
            return self.special_dl_symbols
        else:
            return 0

    def get_next_ul_slot(self, from_slot: int) -> int:
        """从 from_slot 开始 (含), 找到下一个 UL slot"""
        for offset in range(self.period * 2):  # 最多搜两个周期
            s = from_slot + offset
            if self.is_ul_slot(s):
                return s
        return from_slot + self.period  # 不应该到这

    def get_feedback_slot(self, tx_slot: int, k1_min: int = 4) -> int:
        """计算 HARQ 反馈到达的 slot

        在 TDD 中, ACK 只能在 UL slot 发送。
        找 tx_slot + k1 (k1 >= k1_min) 中第一个 UL/S slot。
        """
        candidate = tx_slot + k1_min
        return self.get_next_ul_slot(candidate)

    def __repr__(self):
        return (f"TDDConfig(pattern='{self.pattern}', "
                f"S={self.special_dl_symbols}+{self.special_gp_symbols}+"
                f"{self.special_ul_symbols}, dl_ratio={self.dl_ratio:.1%})")


# 常用 TDD 模式预设
TDD_PRESETS = {
    'DDDSU': TDDConfig('DDDSU', 10, 2, 2),
    'DDDSUDDSUU': TDDConfig('DDDSUDDSUU', 10, 2, 2),
    'DDSUU': TDDConfig('DDSUU', 10, 2, 2),
    'DDDDDDDSUU': TDDConfig('DDDDDDDSUU', 10, 2, 2),
}

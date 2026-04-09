"""分段线性 SE 估算模块

对标 AirView GDlSerialScheduler 的 getLinearFitParamFromSINR / LinearFitSINR2SE。
用于调度前快速估算 UE 的 RB 需求 (不经过 EESM/BLER 表的重量级 PHY 计算)。

SE = rank × (A × dBSinr + B), 分4段线性拟合。
A/B 参数从 3GPP CQI Table 1 (64QAM) 最小二乘标定:
  Seg1 (< -0.84 dB):   低 SINR
  Seg2 (-0.84 ~ 4.69): 中低 SINR
  Seg3 (4.69 ~ 10.26): 中高 SINR
  Seg4 (>= 10.26):     高 SINR
"""

import numpy as np


class SEEstimator:
    """分段线性谱效率估算器"""

    # 分界点 (dB), 对标 AirView
    _BREAKS = (-0.84, 4.69, 10.26)

    # (A, B) per segment — 从 CQI Table 1 最小二乘拟合
    _PARAMS = (
        (0.051343, 0.489035),   # Seg1: < -0.84 dB
        (0.139660, 0.563581),   # Seg2: -0.84 ~ 4.69
        (0.198864, 0.303305),   # Seg3: 4.69 ~ 10.26
        (0.255355, -0.251428),  # Seg4: >= 10.26
    )

    def _get_params(self, sinr_db: float) -> tuple:
        """根据 SINR 选择分段参数 (A, B)"""
        if sinr_db < self._BREAKS[0]:
            return self._PARAMS[0]
        elif sinr_db < self._BREAKS[1]:
            return self._PARAMS[1]
        elif sinr_db < self._BREAKS[2]:
            return self._PARAMS[2]
        else:
            return self._PARAMS[3]

    def estimate_se(self, sinr_db: float, rank: int = 1) -> float:
        """估算谱效率 (bits/RE)

        SE = rank × (A × sinr_db + B), clamp to [0, rank × 5.55]
        """
        A, B = self._get_params(sinr_db)
        se = rank * (A * sinr_db + B)
        return max(min(se, rank * 5.5547), 0.0)

    def estimate_se_batch(self, sinr_db: np.ndarray, rank: np.ndarray) -> np.ndarray:
        """批量估算 SE

        Args:
            sinr_db: (num_ue,) 宽带 SINR (dB)
            rank: (num_ue,) 传输层数

        Returns:
            (num_ue,) SE (bits/RE)
        """
        se = np.zeros_like(sinr_db, dtype=np.float64)
        for i in range(len(sinr_db)):
            se[i] = self.estimate_se(float(sinr_db[i]), int(rank[i]))
        return se

    def estimate_rb_num(self, buffer_bits: float, sinr_db: float,
                        rank: int, re_per_prb: int = 132) -> int:
        """估算传输所需 RB 数

        RB = ceil(buffer_bits / (SE × re_per_prb))

        Args:
            buffer_bits: 待传输比特数
            sinr_db: 宽带 SINR (dB)
            rank: 传输层数
            re_per_prb: 每 PRB 有效 RE 数

        Returns:
            估算所需 RB 数 (上限 9999)
        """
        se = self.estimate_se(sinr_db, rank)
        if se <= 0.0 or buffer_bits <= 0:
            return 0 if buffer_bits <= 0 else 9999
        return min(int(np.ceil(buffer_bits / (se * re_per_prb))), 9999)

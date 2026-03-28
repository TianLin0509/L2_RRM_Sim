"""CSI 反馈延迟建模

模拟 CSI 报告延迟: 调度器使用 N 个 slot 之前的信道状态，
而 PHY 评估使用当前信道状态。

支持:
- 可配置延迟 (slots)
- CSI 量化误差 (可选)
- 信道老化模型 (可选)
"""

import numpy as np
from collections import deque
from ..core.data_types import ChannelState


class CSIFeedbackBuffer:
    """CSI 反馈延迟缓冲区

    维护最近 N 个 slot 的 ChannelState 历史。
    调度器看到的是 delay 个 slot 之前的信道。
    """

    def __init__(self, delay_slots: int = 4,
                 quantization_bits: int = 0,
                 rng: np.random.Generator = None):
        """
        Args:
            delay_slots: CSI 反馈延迟 (slot 数)
            quantization_bits: CSI 量化比特数 (0=不量化)
        """
        self.delay_slots = delay_slots
        self.quantization_bits = quantization_bits
        self._rng = rng if rng is not None else np.random.default_rng()

        # 信道状态环形缓冲区
        self._buffer = deque(maxlen=delay_slots + 1)
        self._current_slot = -1

    def push(self, channel_state: ChannelState, slot_idx: int):
        """推入当前 slot 的信道状态"""
        self._buffer.append((slot_idx, channel_state))
        self._current_slot = slot_idx

    def get_delayed_state(self) -> ChannelState:
        """获取延迟后的信道状态 (调度器使用)

        Returns:
            delay_slots 之前的 ChannelState,
            如果缓冲区不够深，返回最老的可用状态
        """
        if len(self._buffer) == 0:
            return None

        if len(self._buffer) <= self.delay_slots:
            # 缓冲区还没填满，用最老的
            _, state = self._buffer[0]
        else:
            # 正常延迟
            _, state = self._buffer[-(self.delay_slots + 1)]

        if self.quantization_bits > 0:
            return self._quantize_state(state)

        return state

    def get_current_state(self) -> ChannelState:
        """获取当前信道状态 (PHY 评估使用)"""
        if len(self._buffer) == 0:
            return None
        _, state = self._buffer[-1]
        return state

    def _quantize_state(self, state: ChannelState) -> ChannelState:
        """CSI 量化 (简化模型: 加高斯噪声)"""
        # 量化噪声功率与量化比特数相关
        # SNR_quant ≈ 6.02 * bits + 1.76 dB
        quant_snr_db = 6.02 * self.quantization_bits + 1.76
        quant_snr_linear = 10 ** (quant_snr_db / 10.0)

        # 在 SINR 上加量化噪声
        sinr_noisy = state.sinr_per_prb.copy()
        noise_power = sinr_noisy / quant_snr_linear
        noise = self._rng.standard_normal(sinr_noisy.shape) * np.sqrt(noise_power)
        sinr_noisy = np.maximum(sinr_noisy + noise, 0.0)

        return ChannelState(
            pathloss_db=state.pathloss_db.copy(),
            shadow_fading_db=state.shadow_fading_db.copy(),
            sinr_per_prb=sinr_noisy,
            wideband_sinr_db=state.wideband_sinr_db.copy(),
            channel_matrix=state.channel_matrix.copy() if state.channel_matrix is not None else None,
        )


class ChannelAgingModel:
    """信道老化模型

    由于 UE 移动，CSI 反馈延迟导致信道估计与实际信道不匹配。
    老化程度与 UE 速度和延迟时间成正比。

    自相关系数: rho = J_0(2*pi*f_d*tau)
    其中 f_d = v*f_c/c (最大 Doppler 频率), tau = delay * T_slot
    """

    @staticmethod
    def compute_correlation(speed_mps: float, carrier_freq_ghz: float,
                             delay_slots: int, slot_duration_s: float) -> float:
        """计算信道自相关系数

        Returns:
            rho ∈ [0, 1]: 信道相关系数
        """
        from scipy.special import j0

        carrier_freq_hz = carrier_freq_ghz * 1e9
        from ..core.nr_constants import SPEED_OF_LIGHT
        f_d = speed_mps * carrier_freq_hz / SPEED_OF_LIGHT
        tau = delay_slots * slot_duration_s

        rho = abs(j0(2 * np.pi * f_d * tau))
        return float(rho)

    @staticmethod
    def apply_aging(sinr_current: np.ndarray, rho: float,
                    rng: np.random.Generator = None) -> np.ndarray:
        """应用信道老化

        aged_sinr = rho^2 * sinr + (1-rho^2) * noise_equivalent_sinr
        """
        if rho >= 0.999:
            return sinr_current

        if rng is None:
            rng = np.random.default_rng()

        # 老化后的 SINR: rho^2 分量保持，(1-rho^2) 变为随机
        rho2 = rho ** 2
        aged = rho2 * sinr_current + (1 - rho2) * rng.exponential(
            np.mean(sinr_current), sinr_current.shape
        )
        return np.maximum(aged, 0.0)

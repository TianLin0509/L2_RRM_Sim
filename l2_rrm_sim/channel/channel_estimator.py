"""信道估计模块

实现从真实信道到估计信道的转换，模拟 5G NR 中的 CSI 测量和反馈。
"""

import numpy as np
from ..core.data_types import ChannelState
from ..utils.random_utils import SimRNG


class ChannelEstimator:
    """信道估计器

    模拟 LS (Least Squares) 估计，并在信道上添加估计噪声。
    公式: H_est = H_actual + Noise
    """

    def __init__(self, rng: SimRNG, estimation_error_std: float = 0.05):
        """
        Args:
            rng: 随机数生成器
            estimation_error_std: 估计误差的标准差 (相对值)
        """
        self.rng = rng
        self.estimation_error_std = estimation_error_std

    def estimate(self, channel_state: ChannelState) -> np.ndarray:
        """根据真实信道计算估计信道"""
        if channel_state.actual_channel_matrix is None:
            return None

        h_actual = channel_state.actual_channel_matrix
        
        # 模拟 LS 估计误差: E ~ CN(0, sigma^2)
        # 误差标准差与信道均方根相关，这里简化处理
        noise_std = self.estimation_error_std * np.mean(np.abs(h_actual))
        
        noise_real = self.rng.channel.normal(0, noise_std, h_actual.shape)
        noise_imag = self.rng.channel.normal(0, noise_std, h_actual.shape)
        h_noise = noise_real + 1j * noise_imag
        
        h_est = h_actual + h_noise
        
        return h_est

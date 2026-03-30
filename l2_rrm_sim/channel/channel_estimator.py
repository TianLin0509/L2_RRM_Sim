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

    _POOL_SIZE = 64  # number of noise frames in pool

    def __init__(self, rng: SimRNG, estimation_error_std: float = 0.05):
        """
        Args:
            rng: 随机数生成器
            estimation_error_std: 估计误差的标准差 (相对值)
        """
        self.rng = rng
        self.estimation_error_std = estimation_error_std
        self._noise_pool = None
        self._pool_idx = 0
        self._pool_shape = None

    def _generate_pool(self, shape):
        """Pre-generate a batch of CN(0,1) noise matrices."""
        total = self._POOL_SIZE * 2
        flat_size = 1
        for s in shape:
            flat_size *= s
        buf = self.rng.channel.normal(0, 1.0, (total, flat_size))
        real_part = buf[:self._POOL_SIZE].reshape((self._POOL_SIZE,) + shape)
        imag_part = buf[self._POOL_SIZE:].reshape((self._POOL_SIZE,) + shape)
        self._noise_pool = real_part + 1j * imag_part
        self._pool_idx = 0
        self._pool_shape = shape

    def estimate(self, channel_state: ChannelState) -> np.ndarray:
        """根据真实信道计算估计信道"""
        if channel_state.actual_channel_matrix is None:
            return None

        h_actual = channel_state.actual_channel_matrix

        noise_std = self.estimation_error_std * np.mean(np.abs(h_actual))

        # Noise pool: generate or regenerate when exhausted or shape changed
        if (self._noise_pool is None
                or self._pool_idx >= self._POOL_SIZE
                or self._pool_shape != h_actual.shape):
            self._generate_pool(h_actual.shape)

        h_noise = self._noise_pool[self._pool_idx] * noise_std
        self._pool_idx += 1

        return h_actual + h_noise

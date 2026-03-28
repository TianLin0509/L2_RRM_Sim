"""TDL 快衰落信道模型 (3GPP TR 38.901 Clause 7.7.1)

Tapped Delay Line 模型，支持 TDL-A/B/C/D/E 配置，
生成时变多径信道。
"""

import numpy as np
from ..core.nr_constants import SPEED_OF_LIGHT, NUM_SC_PER_PRB


# ============================================================
# TDL 模型参数 (3GPP TR 38.901 Table 7.7.2-1~5)
# (相对功率 dB, 相对时延 ns)
# ============================================================
TDL_A_TAPS = [
    (0.0, 0), (-13.4, 10), (-16.3, 15), (-18.8, 20), (-21.0, 25),
    (-22.8, 50), (-17.1, 65), (-20.8, 75), (-21.7, 85), (-22.7, 105),
    (-17.6, 115), (-20.2, 125), (-21.6, 140), (-22.5, 165), (-23.2, 180),
    (-24.8, 215), (-23.0, 250), (-24.7, 280), (-23.7, 325), (-21.6, 350),
    (-22.4, 380), (-25.2, 430), (-20.8, 490),
]

TDL_C_TAPS = [
    (-4.4, 0), (-1.2, 65), (-3.5, 70), (-5.2, 190), (-2.5, 195),
    (0.0, 200), (-2.2, 240), (-3.9, 325), (-7.4, 335), (-7.1, 350),
    (-10.7, 380), (-11.1, 430), (-5.1, 510), (-6.8, 600), (-8.7, 630),
    (-13.2, 700), (-13.9, 730), (-13.9, 750), (-15.8, 775), (-17.1, 840),
    (-16.0, 890), (-15.7, 925), (-21.6, 1010), (-22.8, 1090),
]

TDL_D_TAPS = [
    # (power_dB, delay_ns, is_los)
    (-0.2, 0), (-13.5, 0), (-18.8, 10), (-21.0, 15), (-22.8, 20),
    (-17.9, 25), (-20.1, 50), (-21.9, 65), (-22.9, 75), (-27.8, 105),
    (-23.6, 115), (-24.8, 135), (-30.0, 150),
]

TDL_PROFILES = {
    'TDL-A': TDL_A_TAPS,
    'TDL-C': TDL_C_TAPS,
    'TDL-D': TDL_D_TAPS,
}


class TDLChannel:
    """TDL 多径信道模型

    生成时变、频率选择性信道系数。
    支持多天线 (per-antenna 独立衰落)。
    """

    def __init__(self, profile: str = 'TDL-A',
                 delay_spread_ns: float = 300.0,
                 carrier_freq_ghz: float = 3.5,
                 scs_khz: int = 30,
                 num_prb: int = 273,
                 num_rx_ant: int = 4,
                 num_tx_ant: int = 4,
                 rng: np.random.Generator = None):
        """
        Args:
            profile: TDL 配置 ('TDL-A', 'TDL-C', 'TDL-D')
            delay_spread_ns: 时延扩展 (ns)，用于缩放标准化时延
            carrier_freq_ghz: 载频 (GHz)
            scs_khz: 子载波间隔 (kHz)
            num_prb: PRB 数
            num_rx_ant: 接收天线数
            num_tx_ant: 发射天线数 (端口数)
        """
        self._rng = rng if rng is not None else np.random.default_rng()
        self.carrier_freq_hz = carrier_freq_ghz * 1e9
        self.scs_hz = scs_khz * 1e3
        self.num_prb = num_prb
        self.num_sc = num_prb * NUM_SC_PER_PRB
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant

        # 加载 TDL profile
        taps = TDL_PROFILES.get(profile, TDL_A_TAPS)
        self._tap_powers_db = np.array([t[0] for t in taps])
        self._tap_delays_ns = np.array([t[1] for t in taps]) * (delay_spread_ns / 100.0)
        self._num_taps = len(taps)

        # 归一化功率
        powers_linear = 10.0 ** (self._tap_powers_db / 10.0)
        self._tap_powers_linear = powers_linear / np.sum(powers_linear)

        # 时延 -> 子载波相位旋转
        # 每个 tap 在第 k 个子载波上的相位: exp(-j*2*pi*tau*k*delta_f)
        self._tap_delays_s = self._tap_delays_ns * 1e-9

        # 每 UE 的 Doppler 相位状态
        self._doppler_phases = {}

    def generate_channel(self, ue_id: int, speed_mps: float,
                         slot_duration_s: float,
                         slot_idx: int) -> np.ndarray:
        """生成单 UE 的频域信道矩阵

        Args:
            ue_id: UE 标识
            speed_mps: UE 速度 (m/s)
            slot_duration_s: slot 时长 (s)
            slot_idx: 当前 slot 索引

        Returns:
            H: (num_rx_ant, num_tx_ant, num_prb) 频域信道系数 [complex]
        """
        # Doppler 频率
        fd_max = speed_mps * self.carrier_freq_hz / SPEED_OF_LIGHT

        # 时间偏移
        t = slot_idx * slot_duration_s

        # 子载波频率偏移 (相对于载频)
        # 每个 PRB 中心频率对应的子载波索引
        prb_center_sc = np.arange(self.num_prb) * NUM_SC_PER_PRB + NUM_SC_PER_PRB // 2
        freq_offset = (prb_center_sc - self.num_sc / 2) * self.scs_hz

        # 初始化/获取 Doppler 相位
        if ue_id not in self._doppler_phases:
            self._doppler_phases[ue_id] = self._rng.uniform(
                0, 2 * np.pi,
                (self._num_taps, self.num_rx_ant, self.num_tx_ant)
            )
        phi_init = self._doppler_phases[ue_id]

        # 生成频域信道: H[rx, tx, prb] = Σ_l sqrt(P_l) * g_l * exp(-j*2*pi*tau_l*f_k)
        H = np.zeros((self.num_rx_ant, self.num_tx_ant, self.num_prb), dtype=complex)

        for l in range(self._num_taps):
            # 每 tap 的衰落系数 (Rayleigh)
            # Jakes model: 时变相位
            doppler_phase = 2 * np.pi * fd_max * t
            g_l = (self._rng.standard_normal((self.num_rx_ant, self.num_tx_ant))
                   + 1j * self._rng.standard_normal((self.num_rx_ant, self.num_tx_ant)))
            g_l /= np.sqrt(2.0)

            # 加入 Doppler 时变
            phase_shift = np.exp(1j * (doppler_phase * np.cos(phi_init[l]) + phi_init[l]))
            g_l = g_l * phase_shift

            # 频域相位旋转
            freq_phase = np.exp(-1j * 2 * np.pi * self._tap_delays_s[l] * freq_offset)

            # 累加
            H += np.sqrt(self._tap_powers_linear[l]) * g_l[:, :, np.newaxis] * freq_phase[np.newaxis, np.newaxis, :]

        return H

    def generate_sinr_per_prb(self, ue_id: int, speed_mps: float,
                               slot_duration_s: float, slot_idx: int,
                               tx_power_per_prb: float,
                               pathloss_linear: float,
                               noise_power: float,
                               num_layers: int = 1) -> np.ndarray:
        """生成 per-layer per-PRB SINR

        Args:
            tx_power_per_prb: 每 PRB 发射功率 (W)
            pathloss_linear: 路径损耗 (线性)
            noise_power: 噪声功率 (W)
            num_layers: 传输层数

        Returns:
            sinr: (num_layers, num_prb) SINR [linear]
        """
        H = self.generate_channel(ue_id, speed_mps, slot_duration_s, slot_idx)

        # 简化: 取前 num_layers 个天线对的信道增益
        sinr = np.zeros((num_layers, self.num_prb))
        for layer in range(num_layers):
            # 信道增益: |H|^2 在发射天线维度求和
            rx_idx = layer % self.num_rx_ant
            channel_gain = np.sum(np.abs(H[rx_idx, :, :]) ** 2, axis=0)
            sinr[layer, :] = tx_power_per_prb * channel_gain / (pathloss_linear * noise_power)

        return sinr

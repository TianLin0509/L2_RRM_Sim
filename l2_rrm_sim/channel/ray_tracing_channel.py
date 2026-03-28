"""射线追踪信道接口

支持两种模式:
1. 加载预计算的 CIR 数据集 (推荐)
2. 接口 Sionna RT (需要 sionna 安装)

CIR 格式: (amplitude, delay, AoA_az, AoA_el, AoD_az, AoD_el)
"""

import numpy as np
from pathlib import Path
from .channel_interface import ChannelModelBase
from ..core.data_types import SlotContext, UEState, ChannelState
from ..core.nr_constants import (
    BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE, NUM_SC_PER_PRB
)
from ..config.sim_config import CellConfig, CarrierConfig
from ..utils.math_utils import db_to_linear, dbm_to_watt, linear_to_db


class RayTracingChannel(ChannelModelBase):
    """射线追踪信道模型

    从预计算的 CIR 数据生成频域信道。
    """

    def __init__(self, cir_data_path: str = None,
                 cell_config: CellConfig = None,
                 carrier_config: CarrierConfig = None,
                 rng: np.random.Generator = None):
        self._rng = rng if rng is not None else np.random.default_rng()
        self._cir_data = None
        self._cell_config = cell_config
        self._carrier_config = carrier_config
        self._noise_power_per_prb = None
        self._tx_power_per_prb = None

        if cir_data_path and Path(cir_data_path).exists():
            self._load_cir_data(cir_data_path)

    def _load_cir_data(self, path: str):
        """加载预计算的 CIR 数据

        期望格式 (npz):
            amplitudes: (num_ue, num_paths, num_rx_ant, num_tx_ant) complex
            delays: (num_ue, num_paths) float, 秒
            aoa_az: (num_ue, num_paths) float, 弧度
            aoa_el: (num_ue, num_paths) float, 弧度
            aod_az: (num_ue, num_paths) float, 弧度
            aod_el: (num_ue, num_paths) float, 弧度
        """
        data = np.load(path, allow_pickle=True)
        self._cir_data = {
            'amplitudes': data['amplitudes'],
            'delays': data['delays'],
        }
        if 'aoa_az' in data:
            self._cir_data['aoa_az'] = data['aoa_az']
            self._cir_data['aoa_el'] = data['aoa_el']
            self._cir_data['aod_az'] = data['aod_az']
            self._cir_data['aod_el'] = data['aod_el']

    def initialize(self, cell_config: CellConfig,
                   carrier_config: CarrierConfig,
                   ue_states: list):
        """初始化信道模型"""
        self._cell_config = cell_config
        self._carrier_config = carrier_config

        bw_prb_hz = carrier_config.subcarrier_spacing * 1e3 * NUM_SC_PER_PRB
        nf_linear = db_to_linear(cell_config.noise_figure_db)
        self._noise_power_per_prb = (
            BOLTZMANN_CONSTANT * STANDARD_TEMPERATURE * bw_prb_hz * nf_linear
        )
        self._tx_power_per_prb = (
            dbm_to_watt(cell_config.total_power_dbm) / carrier_config.num_prb
        )

    def update(self, slot_ctx: SlotContext,
               ue_states: list) -> ChannelState:
        """更新信道状态"""
        num_ue = len(ue_states)
        num_prb = self._carrier_config.num_prb
        max_layers = self._cell_config.max_layers
        scs_hz = self._carrier_config.subcarrier_spacing * 1e3

        sinr_per_prb = np.zeros((num_ue, max_layers, num_prb))
        wideband_sinr_db = np.zeros(num_ue)
        pathloss_db = np.zeros(num_ue)
        shadow_db = np.zeros(num_ue)
        channel_matrix = np.zeros(
            (num_ue, min(4, self._cell_config.num_tx_ports),
             self._cell_config.num_tx_ports, num_prb),
            dtype=complex
        )

        for ue_idx in range(num_ue):
            if self._cir_data is not None and ue_idx < len(self._cir_data['amplitudes']):
                H_freq = self._cir_to_freq(ue_idx, num_prb, scs_hz)
            else:
                # 没有 CIR 数据时回退到 Rayleigh
                n_rx = min(4, self._cell_config.num_tx_ports)
                n_tx = self._cell_config.num_tx_ports
                H_freq = (self._rng.standard_normal((n_rx, n_tx, num_prb))
                          + 1j * self._rng.standard_normal((n_rx, n_tx, num_prb))) / np.sqrt(2)

            n_rx = H_freq.shape[0]
            n_tx = H_freq.shape[1]
            channel_matrix[ue_idx, :n_rx, :n_tx, :] = H_freq

            # 计算 per-layer SINR
            for layer in range(min(max_layers, n_rx)):
                ch_gain = np.sum(np.abs(H_freq[layer, :, :]) ** 2, axis=0)
                sinr_per_prb[ue_idx, layer, :] = (
                    self._tx_power_per_prb * ch_gain / self._noise_power_per_prb
                )

            # 路径损耗 (从信道增益反推)
            avg_gain = np.mean(np.sum(np.abs(H_freq[0, :, :]) ** 2, axis=0))
            pathloss_db[ue_idx] = -linear_to_db(avg_gain) if avg_gain > 0 else 100.0
            wideband_sinr_db[ue_idx] = linear_to_db(
                np.mean(sinr_per_prb[ue_idx, 0, :])
            )

        return ChannelState(
            pathloss_db=pathloss_db,
            shadow_fading_db=shadow_db,
            sinr_per_prb=sinr_per_prb,
            wideband_sinr_db=wideband_sinr_db,
            channel_matrix=channel_matrix,
        )

    def _cir_to_freq(self, ue_idx: int, num_prb: int,
                     scs_hz: float) -> np.ndarray:
        """CIR 转频域信道

        H[rx, tx, prb] = Σ_p a_p * exp(-j*2*pi*tau_p*f_k)
        """
        amps = self._cir_data['amplitudes'][ue_idx]  # (num_paths, rx, tx)
        delays = self._cir_data['delays'][ue_idx]    # (num_paths,)
        num_paths = len(delays)

        num_sc = num_prb * NUM_SC_PER_PRB
        prb_center_sc = np.arange(num_prb) * NUM_SC_PER_PRB + NUM_SC_PER_PRB // 2
        freq_offset = (prb_center_sc - num_sc / 2) * scs_hz

        n_rx, n_tx = amps.shape[1], amps.shape[2]
        H = np.zeros((n_rx, n_tx, num_prb), dtype=complex)

        for p in range(num_paths):
            phase = np.exp(-1j * 2 * np.pi * delays[p] * freq_offset)
            H += amps[p, :, :, np.newaxis] * phase[np.newaxis, np.newaxis, :]

        return H

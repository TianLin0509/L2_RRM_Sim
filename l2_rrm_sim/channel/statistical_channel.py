"""统计信道模型

Phase 1: 路径损耗 + 对数正态阴影衰落 + Rayleigh 快衰落 (per-PRB per-layer)
Phase 2: + Kronecker 空间相关 + Jake's 时间演进 + 3GPP 天线增益 + 特征值持久化
单小区无干扰场景。
"""

import numpy as np
from scipy.special import j0 as bessel_j0
from .channel_interface import ChannelModelBase
from .pathloss_models import PATHLOSS_MODELS, LOS_PROBABILITY_MODELS
from .antenna_model import antenna_gain_3gpp_element
from ..core.data_types import SlotContext, UEState, ChannelState
from ..core.nr_constants import (
    BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE, NUM_SC_PER_PRB
)
from ..config.sim_config import CellConfig, CarrierConfig, ChannelConfig
from ..utils.math_utils import db_to_linear, dbm_to_watt, linear_to_db
from ..core.registry import register_channel

# 空间相关系数映射
_CORR_RHO = {"none": 0.0, "low": 0.3, "medium": 0.6, "high": 0.9}

SPEED_OF_LIGHT = 3e8


@register_channel("statistical")
class StatisticalChannel(ChannelModelBase):
    """统计信道模型

    信道增益模型:
        SINR[ue, layer, prb] = P_tx_per_prb × |H[layer, prb]|² / (PL × N_0)

    其中 |H|² ~ Exponential(1) (Rayleigh 衰落)

    Phase 2 扩展:
    - Kronecker 空间相关: H_corr = L_rx @ H_iid @ L_tx^T
    - Jake's 时间演进: H(t) = rho_t * H(t-1) + sqrt(1-rho_t^2) * H_innov
    - 3GPP 天线增益 (TR 38.901 Table 7.3-1)
    - SVD 特征值持久化到 ChannelState.eigenvalues
    """

    def __init__(self, cell_config: CellConfig,
                 carrier_config: CarrierConfig,
                 channel_config: ChannelConfig = None,
                 rng: np.random.Generator = None):
        self.cell_config = cell_config
        self.carrier_config = carrier_config
        self.channel_config = channel_config or ChannelConfig()
        self._rng = rng if rng is not None else np.random.default_rng()

        # 选择路径损耗模型
        scenario = self.channel_config.scenario or cell_config.scenario
        self._pl_func = PATHLOSS_MODELS.get(scenario, PATHLOSS_MODELS['uma'])
        self._los_prob_func = LOS_PROBABILITY_MODELS.get(scenario)

        # 噪声功率 (per PRB)
        bw_prb_hz = carrier_config.subcarrier_spacing * 1e3 * NUM_SC_PER_PRB
        noise_figure_linear = db_to_linear(cell_config.noise_figure_db)
        self._noise_power_per_prb = (
            BOLTZMANN_CONSTANT * STANDARD_TEMPERATURE * bw_prb_hz * noise_figure_linear
        )

        # 每 PRB 发射功率
        self._tx_power_per_prb = (
            dbm_to_watt(cell_config.total_power_dbm) / carrier_config.num_prb
        )

        # 缓存
        self._pathloss_db = None
        self._shadow_fading_db = None
        self._is_los = None

        # --- Phase 2: Kronecker 空间相关 ---
        self._L_tx = None  # Cholesky of R_tx, (num_tx, num_tx)
        self._L_rx = None  # Cholesky of R_rx, (num_rx, num_rx)
        self._corr_enabled = (
            self.channel_config.spatial_correlation != "none"
        )

        # --- Phase 2: Jake's 时间演进 ---
        self._h_prev = None      # 上一 slot 的信道矩阵 (含相关+路损)
        self._rho_t = 0.0        # Jake's 时间相关系数
        self._doppler_enabled = self.channel_config.doppler_enabled

        # --- Phase 2: 天线增益 ---
        self._antenna_gain_linear = None  # (num_ue,) 线性增益

    def _build_correlation_cholesky(self, n: int, rho: float) -> np.ndarray:
        """构建指数相关矩阵 R[i,j]=rho^|i-j| 的 Cholesky 分解 L"""
        if rho == 0.0 or n <= 1:
            return np.eye(n)
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                R[i, j] = rho ** abs(i - j)
        return np.linalg.cholesky(R)

    def _compute_doppler_rho(self, speed_kmh: float, freq_ghz: float,
                             slot_duration_s: float) -> float:
        """计算 Jake's 时间相关系数 rho_t = J_0(2*pi*fd*T_slot)"""
        speed_ms = speed_kmh / 3.6
        fd = speed_ms * freq_ghz * 1e9 / SPEED_OF_LIGHT
        return float(bessel_j0(2 * np.pi * fd * slot_duration_s))

    def initialize(self, cell_config: CellConfig,
                   carrier_config: CarrierConfig,
                   ue_states: list):
        """初始化: 计算路径损耗和阴影衰落 (大尺度参数, 不随 TTI 变化)"""
        num_ue = len(ue_states)
        num_tx_ant = cell_config.num_tx_ant
        num_rx_ant = getattr(cell_config, 'num_rx_ant', 4)
        if hasattr(ue_states[0], 'num_rx_ant'):
            num_rx_ant = ue_states[0].num_rx_ant

        self._pathloss_db = np.zeros(num_ue)
        self._shadow_fading_db = np.zeros(num_ue)
        self._is_los = np.zeros(num_ue, dtype=bool)

        for i, ue in enumerate(ue_states):
            # 2D 距离
            d_2d = np.sqrt(ue.position[0]**2 + ue.position[1]**2)
            d_2d = max(d_2d, 10.0)

            # LOS 概率判定
            if self._los_prob_func is not None:
                p_los = self._los_prob_func(d_2d, ue.position[2])
                self._is_los[i] = self._rng.random() < p_los
            else:
                self._is_los[i] = d_2d < 50.0

            # 路径损耗
            self._pathloss_db[i] = self._pl_func(
                d_2d, cell_config.height_m, ue.position[2],
                carrier_config.carrier_freq_ghz, self._is_los[i]
            )

            # 阴影衰落
            sf_std = self.channel_config.shadow_fading_std_db
            if not self._is_los[i]:
                sf_std = 6.0  # NLOS 标准差通常更大
            self._shadow_fading_db[i] = self._rng.normal(0, sf_std)

        # --- Phase 2: 预计算 Kronecker Cholesky ---
        if self._corr_enabled:
            rho = _CORR_RHO.get(self.channel_config.spatial_correlation, 0.0)
            self._L_tx = self._build_correlation_cholesky(num_tx_ant, rho)
            self._L_rx = self._build_correlation_cholesky(num_rx_ant, rho)

        # --- Phase 2: 预计算 Jake's rho_t ---
        if self._doppler_enabled:
            # 使用 UEConfig 中的速度 (通过 ue_states[0].velocity 推算)
            # 或直接用 carrier_config 和 channel_config
            speed_kmh = 3.0  # 默认
            if len(ue_states) > 0:
                v = np.linalg.norm(ue_states[0].velocity)
                if v > 0:
                    speed_kmh = v * 3.6
            self._rho_t = self._compute_doppler_rho(
                speed_kmh,
                carrier_config.carrier_freq_ghz,
                carrier_config.slot_duration_s
            )

        # --- Phase 2: 预计算天线增益 ---
        if self.channel_config.antenna_gain_enabled:
            self._antenna_gain_linear = np.ones(num_ue)
            for i, ue in enumerate(ue_states):
                # 水平角: UE 相对 gNB 的方位角
                phi_deg = np.degrees(np.arctan2(ue.position[1], ue.position[0]))
                # 垂直角: 俯仰角 (从天顶算, 90度=水平)
                d_2d = max(np.sqrt(ue.position[0]**2 + ue.position[1]**2), 10.0)
                height_diff = cell_config.height_m - ue.position[2]
                theta_deg = 90.0 + np.degrees(np.arctan2(height_diff, d_2d))
                gain_dbi = antenna_gain_3gpp_element(theta_deg, phi_deg)
                self._antenna_gain_linear[i] = db_to_linear(gain_dbi)

        # 重置 Jake's 缓存
        self._h_prev = None

    def update(self, slot_ctx: SlotContext,
               ue_states: list) -> ChannelState:
        """更新信道: 生成每 TTI 的快衰落"""
        num_ue = len(ue_states)
        num_prb = self.carrier_config.num_prb
        max_layers = self.cell_config.max_layers
        num_tx_ant = self.cell_config.num_tx_ant
        num_rx_ant = getattr(self.cell_config, 'num_rx_ant', 4)
        if hasattr(ue_states[0], 'num_rx_ant'):
            num_rx_ant = ue_states[0].num_rx_ant

        # 生成 iid 复高斯信道矩阵 H ~ CN(0, 1) - batch RNG
        shape = (num_ue, num_rx_ant, num_tx_ant, num_prb)
        total_elements = num_ue * num_rx_ant * num_tx_ant * num_prb
        _buf = self._rng.normal(0, 1/np.sqrt(2), 2 * total_elements)
        h_real = _buf[:total_elements].reshape(shape)
        h_imag = _buf[total_elements:].reshape(shape)
        h_iid = h_real + 1j * h_imag

        # --- Phase 2: Jake's 时间演进 (在 iid 域操作，避免 Kronecker 功率累积) ---
        if self._doppler_enabled and self._h_prev is not None:
            rho = self._rho_t
            h_iid = rho * self._h_prev + np.sqrt(1.0 - rho**2) * h_iid

        # 保存 iid 域 H 作为下一 slot 的 H_prev (Kronecker 前, 保持单位方差)
        if self._doppler_enabled:
            self._h_prev = h_iid.copy()

        # --- Phase 2: Kronecker 空间相关 (在 Jake's 之后应用) ---
        # H_corr = L_rx @ H_iid @ L_tx^T  (per UE per PRB)
        if self._corr_enabled and self._L_tx is not None:
            h_flat = h_iid.transpose(0, 3, 1, 2).reshape(-1, num_rx_ant, num_tx_ant)
            h_corr = self._L_rx @ h_flat @ self._L_tx.T
            h_iid = h_corr.reshape(num_ue, num_prb, num_rx_ant, num_tx_ant).transpose(0, 2, 3, 1)

        # --- 应用路径损耗 + 阴影衰落 ---
        total_loss_db = self._pathloss_db + self._shadow_fading_db
        total_loss_linear = db_to_linear(total_loss_db)
        sqrt_loss = np.sqrt(total_loss_linear)[:, None, None, None]
        actual_channel_matrix = h_iid / sqrt_loss

        # --- Phase 2: 天线增益 (振幅域) ---
        if (self.channel_config.antenna_gain_enabled
                and self._antenna_gain_linear is not None):
            sqrt_gain = np.sqrt(self._antenna_gain_linear)[:, None, None, None]
            actual_channel_matrix = actual_channel_matrix * sqrt_gain

        # Batch SVD on actual_channel_matrix
        H_batch = actual_channel_matrix.transpose(0, 3, 1, 2).reshape(
            -1, num_rx_ant, num_tx_ant
        )  # (num_ue*num_prb, num_rx, num_tx)
        _, S_batch, _ = np.linalg.svd(H_batch, full_matrices=False)
        # S_batch: (num_ue*num_prb, min(rx,tx))
        num_sv = S_batch.shape[1]
        S_all = S_batch.reshape(num_ue, num_prb, num_sv).transpose(0, 2, 1)
        # S_all: (num_ue, num_sv, num_prb)

        n_layers = min(max_layers, num_sv)

        # Vectorized SINR computation
        sinr_per_prb = np.zeros((num_ue, max_layers, num_prb))
        # Rank-independent SINR: P * |s|² / N0 (不含功率分摊因子 /r)
        sinr_per_prb[:, :n_layers, :] = (
            self._tx_power_per_prb * S_all[:, :n_layers, :] ** 2
            / self._noise_power_per_prb
        )

        # Wideband SINR based on layer-0 (reference layer)
        mean_sinr = np.mean(sinr_per_prb[:, 0, :], axis=1)
        wideband_sinr_db = linear_to_db(mean_sinr)

        # --- Phase 2: 特征值持久化 ---
        eigenvalues = np.zeros((num_ue, max_layers, num_prb))
        eigenvalues[:, :n_layers, :] = S_all[:, :n_layers, :]

        return ChannelState(
            pathloss_db=self._pathloss_db.copy(),
            shadow_fading_db=self._shadow_fading_db.copy(),
            sinr_per_prb=sinr_per_prb,
            wideband_sinr_db=wideband_sinr_db,
            actual_channel_matrix=actual_channel_matrix,
            eigenvalues=eigenvalues
        )

"""统计信道模型

Phase 1: 路径损耗 + 对数正态阴影衰落 + Rayleigh 快衰落 (per-PRB per-layer)
单小区无干扰场景。
"""

import numpy as np
from .channel_interface import ChannelModelBase
from .pathloss_models import PATHLOSS_MODELS, LOS_PROBABILITY_MODELS
from ..core.data_types import SlotContext, UEState, ChannelState
from ..core.nr_constants import (
    BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE, NUM_SC_PER_PRB
)
from ..config.sim_config import CellConfig, CarrierConfig, ChannelConfig
from ..utils.math_utils import db_to_linear, dbm_to_watt, linear_to_db


class StatisticalChannel(ChannelModelBase):
    """统计信道模型

    信道增益模型:
        SINR[ue, layer, prb] = P_tx_per_prb × |H[layer, prb]|² / (PL × N_0)

    其中 |H|² ~ Exponential(1) (Rayleigh 衰落)
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

    def initialize(self, cell_config: CellConfig,
                   carrier_config: CarrierConfig,
                   ue_states: list):
        """初始化: 计算路径损耗和阴影衰落 (大尺度参数, 不随 TTI 变化)"""
        num_ue = len(ue_states)
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

    def update(self, slot_ctx: SlotContext,
               ue_states: list) -> ChannelState:
        """更新信道: 生成每 TTI 的快衰落"""
        num_ue = len(ue_states)
        num_prb = self.carrier_config.num_prb
        max_layers = self.cell_config.max_layers

        # 快衰落: Rayleigh → |H|² ~ Exp(1)
        # shape: (num_ue, max_layers, num_prb)
        fading_gain = self._rng.exponential(1.0, (num_ue, max_layers, num_prb))

        # SINR 计算
        sinr_per_prb = np.zeros((num_ue, max_layers, num_prb))
        wideband_sinr_db = np.zeros(num_ue)

        for ue in range(num_ue):
            # 总路径损耗 (linear)
            total_loss_db = self._pathloss_db[ue] + self._shadow_fading_db[ue]
            total_loss_linear = db_to_linear(total_loss_db)

            # per-PRB per-layer SINR (linear)
            # SINR = P_tx_per_prb × gain / (loss × noise)
            sinr_per_prb[ue] = (
                self._tx_power_per_prb * fading_gain[ue]
                / (total_loss_linear * self._noise_power_per_prb)
            )

            # 宽带 SINR (所有 PRB 的平均, layer 0)
            wideband_sinr_db[ue] = linear_to_db(np.mean(sinr_per_prb[ue, 0, :]))

        return ChannelState(
            pathloss_db=self._pathloss_db.copy(),
            shadow_fading_db=self._shadow_fading_db.copy(),
            sinr_per_prb=sinr_per_prb,
            wideband_sinr_db=wideband_sinr_db,
            channel_matrix=None
        )

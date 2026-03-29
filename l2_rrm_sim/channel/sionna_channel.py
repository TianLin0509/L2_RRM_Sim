"""Sionna 2.0 信道适配器

调用 Sionna 的 3GPP TR 38.901 信道模型生成频域信道矩阵，
转换为 L2 RRM 引擎需要的 SINR 格式。

信道管线:
  Topology → LSP → Rays → CIR(h, tau) → OFDM H[f]
  → pathloss + post-eq SINR per PRB per layer

需要 Python 3.12 + sionna 2.0 + PyTorch (GPU 可选)
"""

import numpy as np
from .channel_interface import ChannelModelBase
from ..core.data_types import SlotContext, UEState, ChannelState
from ..core.nr_constants import (
    BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE, NUM_SC_PER_PRB
)
from ..config.sim_config import CellConfig, CarrierConfig, ChannelConfig
from ..utils.math_utils import db_to_linear, dbm_to_watt, linear_to_db

try:
    import torch
    from sionna.phy.channel.tr38901 import UMa, UMi, RMa, PanelArray
    from sionna.phy.channel import cir_to_ofdm_channel
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False


class SionnaChannel(ChannelModelBase):
    """Sionna 3GPP TR 38.901 信道模型适配器

    完整信道管线:
    1. set_topology: 设置 BS/UE 位置 → Sionna 内部生成 LSP (大尺度参数)
    2. update (每 TTI):
       a. Sionna 生成 CIR (h, tau) — 含小尺度衰落、Doppler
       b. CIR → OFDM 频域转换
       c. 从 H[f] 计算 per-PRB per-layer SINR
    """

    # 场景映射
    SCENARIO_MAP = {'uma': UMa, 'umi': UMi, 'rma': RMa} if SIONNA_AVAILABLE else {}

    def __init__(self, cell_config: CellConfig,
                 carrier_config: CarrierConfig,
                 channel_config: ChannelConfig = None,
                 device: str = None):
        """
        Args:
            device: PyTorch 设备 ('cuda:0', 'cpu', None=自动)
        """
        if not SIONNA_AVAILABLE:
            raise ImportError(
                "Sionna 2.0 未安装。请使用 .venv312 环境: "
                ".venv312/Scripts/python.exe"
            )

        self.cell_config = cell_config
        self.carrier_config = carrier_config
        self.channel_config = channel_config or ChannelConfig()

        # 设备选择
        if device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        # 天线配置
        carrier_freq = carrier_config.carrier_freq_ghz * 1e9

        # UE 天线: 单面板, 单极化
        self._ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=cell_config.num_tx_ports // 2 if cell_config.num_tx_ports > 1 else 1,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=carrier_freq,
        )

        # BS 天线: 多面板, 双极化, 3GPP 方向图
        # num_tx_ant = rows × cols × 2(dual-pol)
        n_ant = cell_config.num_tx_ant
        n_rows = max(1, int(np.sqrt(n_ant / 2)))
        n_cols = max(1, n_ant // (2 * n_rows))
        self._bs_array = PanelArray(
            num_rows_per_panel=n_rows,
            num_cols_per_panel=n_cols,
            polarization='dual',
            polarization_type='cross',
            antenna_pattern='38.901',
            carrier_frequency=carrier_freq,
        )

        self._num_tx_ant = self._bs_array.num_ant
        self._num_rx_ant = self._ut_array.num_ant

        # Sionna 信道模型
        scenario_cls = self.SCENARIO_MAP.get(
            self.channel_config.scenario or cell_config.scenario,
            UMa
        )
        self._channel = scenario_cls(
            carrier_frequency=carrier_freq,
            o2i_model='low',
            ut_array=self._ut_array,
            bs_array=self._bs_array,
            direction='downlink',
            enable_pathloss=True,
            enable_shadow_fading=True,
            precision='single',
            device=self._device,
        )

        # OFDM 参数
        self._num_prb = carrier_config.num_prb
        self._num_sc = carrier_config.num_prb * NUM_SC_PER_PRB
        self._scs_hz = carrier_config.subcarrier_spacing * 1e3
        self._sampling_freq = self._scs_hz * self._num_sc  # 近似

        # 子载波频率偏移 (用于 CIR→OFDM)
        sc_indices = torch.arange(self._num_sc, device=self._device, dtype=torch.float32)
        self._frequencies = (sc_indices - self._num_sc / 2) * self._scs_hz

        # 噪声功率 per subcarrier
        bw_sc_hz = self._scs_hz
        nf_linear = db_to_linear(cell_config.noise_figure_db)
        self._noise_power_per_sc = (
            BOLTZMANN_CONSTANT * STANDARD_TEMPERATURE * bw_sc_hz * nf_linear
        )

        # 每 PRB 发射功率
        self._tx_power_per_prb = (
            dbm_to_watt(cell_config.total_power_dbm) / carrier_config.num_prb
        )
        self._tx_power_total = dbm_to_watt(cell_config.total_power_dbm)

        # 缓存
        self._topology_set = False

    def initialize(self, cell_config: CellConfig,
                   carrier_config: CarrierConfig,
                   ue_states: list):
        """设置拓扑 → Sionna 内部生成 LSP"""
        num_ue = len(ue_states)
        batch_size = 1
        num_bs = 1

        # BS 位置 (原点, 高度来自配置)
        bs_loc = torch.zeros(batch_size, num_bs, 3, device=self._device)
        bs_loc[..., 2] = cell_config.height_m

        # UE 位置
        ut_loc = torch.zeros(batch_size, num_ue, 3, device=self._device)
        for i, ue in enumerate(ue_states):
            ut_loc[0, i, 0] = ue.position[0]
            ut_loc[0, i, 1] = ue.position[1]
            ut_loc[0, i, 2] = ue.position[2]

        # UE 速度
        ut_vel = torch.zeros(batch_size, num_ue, 3, device=self._device)
        for i, ue in enumerate(ue_states):
            ut_vel[0, i, 0] = ue.velocity[0]
            ut_vel[0, i, 1] = ue.velocity[1]
            ut_vel[0, i, 2] = ue.velocity[2]

        # 设置拓扑 (触发 LSP 生成)
        self._channel.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=torch.zeros(batch_size, num_ue, 3, device=self._device),
            bs_orientations=torch.zeros(batch_size, num_bs, 3, device=self._device),
            ut_velocities=ut_vel,
            in_state=torch.zeros(batch_size, num_ue, dtype=torch.bool, device=self._device),
        )
        self._topology_set = True
        self._num_ue = num_ue

    def update(self, slot_ctx: SlotContext,
               ue_states: list) -> ChannelState:
        """生成当前 TTI 的信道"""
        if not self._topology_set:
            raise RuntimeError("必须先调用 initialize()")

        num_ue = self._num_ue
        num_prb = self._num_prb
        max_layers = self.cell_config.max_layers

        with torch.no_grad():
            # 1. 生成 CIR
            # h: [batch, num_rx=UE, num_rx_ant, num_tx=BS, num_tx_ant, num_paths, num_time_samples]
            # tau: [batch, num_rx, num_tx, num_paths]
            h_cir, tau = self._channel(
                num_time_samples=1,  # 单时间采样点 (quasi-static per slot)
                sampling_frequency=self._sampling_freq,
            )

            # 2. CIR → OFDM 频域
            # h_freq: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time, num_sc]
            h_freq = cir_to_ofdm_channel(
                self._frequencies, h_cir, tau, normalize=False
            )

        # 3. 转换为 numpy 并计算 SINR
        # h_freq shape: [1, num_ue, rx_ant, 1, tx_ant, 1, num_sc]
        h_np = h_freq.cpu().numpy()
        h_np = h_np[0, :, :, 0, :, 0, :]
        # h_np: [num_ue, rx_ant, tx_ant, num_sc]

        # 4. 计算 per-PRB per-layer SINR
        sinr_per_prb = np.zeros((num_ue, max_layers, num_prb))
        wideband_sinr_db = np.zeros(num_ue)
        pathloss_db = np.zeros(num_ue)
        channel_matrix = np.zeros((num_ue, self._num_rx_ant,
                                   self._num_tx_ant, num_prb), dtype=complex)

        for ue in range(num_ue):
            h_ue = h_np[ue]  # (rx_ant, tx_ant, num_sc)

            # 按 PRB 聚合 (每 PRB 12 子载波取平均功率)
            h_prb = np.zeros((h_ue.shape[0], h_ue.shape[1], num_prb), dtype=complex)
            for prb in range(num_prb):
                sc_start = prb * NUM_SC_PER_PRB
                sc_end = sc_start + NUM_SC_PER_PRB
                h_prb[:, :, prb] = np.mean(h_ue[:, :, sc_start:sc_end], axis=2)

            channel_matrix[ue, :h_ue.shape[0], :h_ue.shape[1], :] = h_prb

            # SVD per PRB 计算 per-layer SINR
            # 对每个 PRB: H[rx, tx] → SVD → 奇异值 → SINR
            for prb in range(num_prb):
                H = h_prb[:, :, prb]  # (rx_ant, tx_ant)
                try:
                    U, S, Vh = np.linalg.svd(H, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue

                # 每 layer 的 SINR: P_tx × sigma_l^2 / (num_layers × N_0)
                # 等功率分配到各层
                n_layers = min(max_layers, len(S))
                noise = self._noise_power_per_sc * NUM_SC_PER_PRB  # PRB 噪声
                for l in range(n_layers):
                    signal = self._tx_power_per_prb * S[l]**2
                    sinr_per_prb[ue, l, prb] = signal / (n_layers * noise)

            # 路径损耗 (从信道能量反推)
            avg_gain = np.mean(np.sum(np.abs(h_prb[0, :, :])**2, axis=0))
            if avg_gain > 0:
                # PL = P_tx / (P_rx / gain_antenna) 近似
                pathloss_db[ue] = -linear_to_db(avg_gain)
            else:
                pathloss_db[ue] = 150.0

            # 宽带 SINR
            mean_sinr = np.mean(sinr_per_prb[ue, 0, :])
            wideband_sinr_db[ue] = linear_to_db(mean_sinr) if mean_sinr > 0 else -30.0

        return ChannelState(
            pathloss_db=pathloss_db,
            shadow_fading_db=np.zeros(num_ue),  # 已包含在 Sionna 的信道中
            sinr_per_prb=sinr_per_prb,
            wideband_sinr_db=wideband_sinr_db,
            channel_matrix=channel_matrix,
        )

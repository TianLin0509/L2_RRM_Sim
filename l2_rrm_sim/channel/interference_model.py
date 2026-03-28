"""小区间干扰建模

计算来自邻小区的干扰功率，用于多小区 SINR 计算。
"""

import numpy as np
from ..channel.pathloss_models import PATHLOSS_MODELS
from ..utils.math_utils import db_to_linear, dbm_to_watt
from ..core.nr_constants import BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE, NUM_SC_PER_PRB


class InterCellInterference:
    """小区间干扰计算器

    根据邻小区的发射功率和路径损耗，
    计算每个 UE 受到的总干扰功率。
    """

    def __init__(self, num_cells: int,
                 total_power_dbm: float = 46.0,
                 num_prb: int = 273,
                 carrier_freq_ghz: float = 3.5,
                 cell_height: float = 25.0,
                 scenario: str = 'uma',
                 noise_figure_db: float = 5.0,
                 scs_khz: int = 30,
                 load_factor: float = 1.0):
        """
        Args:
            num_cells: 总小区数
            load_factor: 邻小区负载因子 (0~1, 1=满载)
        """
        self.num_cells = num_cells
        self.tx_power_per_prb = dbm_to_watt(total_power_dbm) / num_prb
        self.carrier_freq_ghz = carrier_freq_ghz
        self.cell_height = cell_height
        self.num_prb = num_prb
        self.load_factor = load_factor
        self._pl_func = PATHLOSS_MODELS.get(scenario, PATHLOSS_MODELS['uma'])

        # 噪声功率
        bw_prb_hz = scs_khz * 1e3 * NUM_SC_PER_PRB
        nf_linear = db_to_linear(noise_figure_db)
        self.noise_power_per_prb = (
            BOLTZMANN_CONSTANT * STANDARD_TEMPERATURE * bw_prb_hz * nf_linear
        )

    def compute_interference(self, ue_pos: np.ndarray,
                              serving_cell: int,
                              cell_positions: np.ndarray,
                              topology=None) -> np.ndarray:
        """计算来自邻小区的干扰

        Args:
            ue_pos: (3,) UE 位置
            serving_cell: 服务小区索引
            cell_positions: (num_cells, 2) 各小区位置
            topology: HexGridTopology (用于 wrap-around 距离)

        Returns:
            interference_per_prb: (num_prb,) 干扰功率 (W) per PRB
        """
        total_interference = 0.0

        for cell_idx in range(self.num_cells):
            if cell_idx == serving_cell:
                continue

            # 计算距离
            if topology is not None:
                d_2d = topology.compute_distance_2d(ue_pos, cell_idx)
            else:
                cx, cy = cell_positions[cell_idx]
                d_2d = np.sqrt((ue_pos[0] - cx)**2 + (ue_pos[1] - cy)**2)

            d_2d = max(d_2d, 10.0)

            # 路径损耗 (假设 NLOS)
            pl_db = self._pl_func(
                d_2d, self.cell_height, ue_pos[2],
                self.carrier_freq_ghz, is_los=False
            )
            pl_linear = db_to_linear(pl_db)

            # 干扰功率 = P_tx_per_prb / PL × load_factor
            intf = self.tx_power_per_prb * self.load_factor / pl_linear
            total_interference += intf

        return np.full(self.num_prb, total_interference)

    def compute_sinr_with_interference(self, signal_power_per_prb: np.ndarray,
                                        interference_per_prb: np.ndarray) -> np.ndarray:
        """计算含干扰的 SINR

        Args:
            signal_power_per_prb: (num_prb,) 信号功率
            interference_per_prb: (num_prb,) 干扰功率

        Returns:
            sinr: (num_prb,) SINR [linear]
        """
        return signal_power_per_prb / (interference_per_prb + self.noise_power_per_prb)

    def compute_geometry_factor(self, ue_pos: np.ndarray,
                                 serving_cell: int,
                                 cell_positions: np.ndarray,
                                 topology=None) -> float:
        """计算几何因子 G = S / (I + N)

        Returns:
            geometry_factor_db: 几何因子 (dB)
        """
        from ..utils.math_utils import linear_to_db

        # 服务小区信号
        if topology is not None:
            d_2d_s = topology.compute_distance_2d(ue_pos, serving_cell)
        else:
            cx, cy = cell_positions[serving_cell]
            d_2d_s = np.sqrt((ue_pos[0] - cx)**2 + (ue_pos[1] - cy)**2)

        d_2d_s = max(d_2d_s, 10.0)
        pl_s_db = self._pl_func(
            d_2d_s, self.cell_height, ue_pos[2],
            self.carrier_freq_ghz, is_los=True
        )
        signal = self.tx_power_per_prb / db_to_linear(pl_s_db)

        # 干扰
        intf = np.mean(self.compute_interference(
            ue_pos, serving_cell, cell_positions, topology
        ))

        g = signal / (intf + self.noise_power_per_prb)
        return float(linear_to_db(g))

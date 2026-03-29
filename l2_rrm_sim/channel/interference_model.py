"""动态小区间干扰建模

两种模式:
1. 静态模式: 预计算平均干扰 (快速, 适合早期评估)
2. 动态模式: 每 slot 根据邻小区实际调度结果计算干扰 (精确)

动态干扰公式:
  I_ue[prb] = Σ_{cell ≠ serving} (P_tx_cell[prb] / PL_cell_ue) × load_cell[prb]
  其中:
    P_tx_cell[prb]: 邻小区在该 PRB 上的发射功率 (取决于调度)
    PL_cell_ue: UE 到邻小区的路径损耗
    load_cell[prb]: 邻小区在该 PRB 上是否有调度 (0 或 1)
"""

import numpy as np
from ..channel.pathloss_models import PATHLOSS_MODELS
from ..utils.math_utils import db_to_linear, dbm_to_watt, linear_to_db
from ..core.nr_constants import BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE, NUM_SC_PER_PRB


class InterCellInterference:
    """小区间干扰计算器"""

    def __init__(self, num_cells: int,
                 total_power_dbm: float = 46.0,
                 num_prb: int = 273,
                 carrier_freq_ghz: float = 3.5,
                 cell_height: float = 25.0,
                 scenario: str = 'uma',
                 noise_figure_db: float = 5.0,
                 scs_khz: int = 30,
                 load_factor: float = 1.0):
        self.num_cells = num_cells
        self.total_power_w = dbm_to_watt(total_power_dbm)
        self.tx_power_per_prb = self.total_power_w / num_prb
        self.carrier_freq_ghz = carrier_freq_ghz
        self.cell_height = cell_height
        self.num_prb = num_prb
        self.load_factor = load_factor
        self._pl_func = PATHLOSS_MODELS.get(scenario, PATHLOSS_MODELS['uma'])

        bw_prb_hz = scs_khz * 1e3 * NUM_SC_PER_PRB
        nf_linear = db_to_linear(noise_figure_db)
        self.noise_power_per_prb = (
            BOLTZMANN_CONSTANT * STANDARD_TEMPERATURE * bw_prb_hz * nf_linear
        )

        # 缓存: UE 到各小区的路径损耗
        self._pathloss_cache = {}  # (cell_idx, ue_global_id) → PL_linear

    def precompute_pathloss(self, cell_ue_states: dict,
                            topology=None,
                            cell_positions: np.ndarray = None):
        """预计算所有 UE 到所有小区的路径损耗 (仿真开始时调一次)

        Args:
            cell_ue_states: {cell_idx: [UEState, ...]}
            topology: HexGridTopology (可选, 用于 wrap-around 距离)
        """
        self._pathloss_cache.clear()
        for serving_cell, ue_list in cell_ue_states.items():
            for ue_idx, ue in enumerate(ue_list):
                ue_key = (serving_cell, ue_idx)
                # 到所有小区的路径损耗
                for cell_idx in range(self.num_cells):
                    if topology is not None:
                        d_2d = topology.compute_distance_2d(ue.position, cell_idx)
                    elif cell_positions is not None:
                        cx, cy = cell_positions[cell_idx]
                        d_2d = np.sqrt((ue.position[0]-cx)**2 + (ue.position[1]-cy)**2)
                    else:
                        d_2d = 100.0
                    d_2d = max(d_2d, 10.0)
                    pl_db = self._pl_func(d_2d, self.cell_height, ue.position[2],
                                          self.carrier_freq_ghz, is_los=False)
                    self._pathloss_cache[(ue_key, cell_idx)] = db_to_linear(pl_db)

    def compute_static_interference(self, ue_key: tuple,
                                     serving_cell: int) -> float:
        """静态干扰 (平均, 兼容旧接口)

        Returns:
            平均干扰功率 per PRB (W)
        """
        total_intf = 0.0
        for cell_idx in range(self.num_cells):
            if cell_idx == serving_cell:
                continue
            pl_linear = self._pathloss_cache.get((ue_key, cell_idx), 1e10)
            total_intf += self.tx_power_per_prb * self.load_factor / pl_linear
        return total_intf

    def compute_dynamic_interference(self, ue_key: tuple,
                                      serving_cell: int,
                                      cell_prb_loads: dict) -> np.ndarray:
        """动态干扰: 根据邻小区每 PRB 的实际调度状态计算

        Args:
            ue_key: (serving_cell, ue_idx_in_cell)
            serving_cell: 服务小区索引
            cell_prb_loads: {cell_idx: prb_load_array}
                prb_load_array: (num_prb,) float, 每 PRB 的负载 (0~1)
                1.0 = 该 PRB 有调度, 0.0 = 空闲

        Returns:
            interference_per_prb: (num_prb,) 干扰功率 (W)
        """
        intf = np.zeros(self.num_prb)
        for cell_idx in range(self.num_cells):
            if cell_idx == serving_cell:
                continue
            pl_linear = self._pathloss_cache.get((ue_key, cell_idx), 1e10)
            load = cell_prb_loads.get(cell_idx, np.full(self.num_prb, self.load_factor))
            intf += self.tx_power_per_prb * load / pl_linear
        return intf

    def compute_sinr_with_interference(self, signal_power_per_prb: np.ndarray,
                                        interference_per_prb: np.ndarray) -> np.ndarray:
        """SINR = signal / (interference + noise)"""
        return signal_power_per_prb / (interference_per_prb + self.noise_power_per_prb)

    # --- 兼容旧接口 ---
    def compute_interference(self, ue_pos, serving_cell, cell_positions,
                              topology=None):
        """兼容旧接口: 静态干扰计算"""
        total_intf = 0.0
        for cell_idx in range(self.num_cells):
            if cell_idx == serving_cell:
                continue
            if topology is not None:
                d_2d = topology.compute_distance_2d(ue_pos, cell_idx)
            else:
                cx, cy = cell_positions[cell_idx]
                d_2d = np.sqrt((ue_pos[0]-cx)**2 + (ue_pos[1]-cy)**2)
            d_2d = max(d_2d, 10.0)
            pl_db = self._pl_func(d_2d, self.cell_height, ue_pos[2],
                                  self.carrier_freq_ghz, is_los=False)
            total_intf += self.tx_power_per_prb * self.load_factor / db_to_linear(pl_db)
        return np.full(self.num_prb, total_intf)

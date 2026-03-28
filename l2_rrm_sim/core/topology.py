"""多小区拓扑管理

六角网格拓扑 + Wrap-around，消除边缘效应。
支持多环网格，每小区 3 扇区。
"""

import numpy as np


class HexGridTopology:
    """六角网格拓扑

    生成多环六角网格小区位置，
    支持 wrap-around 消除边缘效应。
    """

    def __init__(self, num_rings: int = 1,
                 isd: float = 500.0,
                 cell_height: float = 25.0,
                 num_sectors: int = 3,
                 wraparound: bool = True):
        """
        Args:
            num_rings: 环数 (0=仅中心, 1=7小区, 2=19小区)
            isd: Inter-Site Distance (m)
            cell_height: 基站高度 (m)
            num_sectors: 每站点扇区数
            wraparound: 是否启用 wrap-around
        """
        self.num_rings = num_rings
        self.isd = isd
        self.cell_height = cell_height
        self.num_sectors = num_sectors
        self.wraparound = wraparound

        # 生成站点位置
        self.site_positions = self._generate_hex_positions()
        self.num_sites = len(self.site_positions)
        self.num_cells = self.num_sites * num_sectors

        # 扇区方向角 (度)
        self.sector_angles = np.array([0, 120, 240])[:num_sectors]

        # 小区位置 = 站点位置 (扇区共址)
        self.cell_positions = np.repeat(self.site_positions, num_sectors, axis=0)
        self.cell_sector_ids = np.tile(np.arange(num_sectors), self.num_sites)

        # Wrap-around: 6 个镜像偏移
        if wraparound:
            self._mirror_offsets = self._compute_mirror_offsets()
        else:
            self._mirror_offsets = np.zeros((1, 2))

    def _generate_hex_positions(self) -> np.ndarray:
        """生成六角螺旋网格站点位置

        Returns:
            positions: (num_sites, 2) xy 坐标
        """
        positions = [(0.0, 0.0)]  # 中心站点

        for ring in range(1, self.num_rings + 1):
            # 六角形的 6 个方向
            hex_dirs = [
                (1, 0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2),
                (-1, 0), (-0.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)/2)
            ]

            # 起始点
            x, y = ring * self.isd, 0.0

            for d in range(6):
                dx, dy = hex_dirs[(d + 2) % 6]
                for _ in range(ring):
                    positions.append((x, y))
                    x += dx * self.isd
                    y += dy * self.isd

        return np.array(positions)

    def _compute_mirror_offsets(self) -> np.ndarray:
        """计算 wrap-around 镜像偏移

        对于六角网格，需要 6 个镜像来覆盖所有方向。
        """
        # 网格跨度
        R = self.isd * (2 * self.num_rings + 1)
        angles = np.arange(6) * 60.0 * np.pi / 180.0

        offsets = np.zeros((7, 2))
        offsets[0] = [0, 0]  # 原始位置
        for i in range(6):
            offsets[i + 1] = [R * np.cos(angles[i]),
                              R * np.sin(angles[i])]
        return offsets

    def drop_ues(self, num_ue_per_cell: int,
                 min_distance: float = 35.0,
                 max_distance: float = None,
                 ue_height: float = 1.5,
                 rng: np.random.Generator = None) -> np.ndarray:
        """在各小区内均匀撒点

        Args:
            num_ue_per_cell: 每小区 UE 数

        Returns:
            ue_positions: (num_cells, num_ue_per_cell, 3) xyz 坐标
        """
        if rng is None:
            rng = np.random.default_rng()
        if max_distance is None:
            max_distance = self.isd / 2

        positions = np.zeros((self.num_cells, num_ue_per_cell, 3))

        for cell_idx in range(self.num_cells):
            cx, cy = self.cell_positions[cell_idx]
            sector_angle = self.sector_angles[self.cell_sector_ids[cell_idx]]

            for ue_idx in range(num_ue_per_cell):
                # 在扇区内均匀分布
                while True:
                    r = np.sqrt(rng.uniform(min_distance**2, max_distance**2))
                    theta = rng.uniform(
                        (sector_angle - 60) * np.pi / 180,
                        (sector_angle + 60) * np.pi / 180
                    )
                    x = cx + r * np.cos(theta)
                    y = cy + r * np.sin(theta)

                    # 确保在扇区内
                    d = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if d >= min_distance:
                        break

                positions[cell_idx, ue_idx] = [x, y, ue_height]

        return positions

    def compute_distance(self, ue_pos: np.ndarray,
                         cell_idx: int) -> float:
        """计算 UE 到指定小区的距离 (含 wrap-around)

        Returns:
            最小距离 (m)
        """
        cx, cy = self.cell_positions[cell_idx]

        if self.wraparound:
            min_dist = float('inf')
            for offset in self._mirror_offsets:
                dx = ue_pos[0] - (cx + offset[0])
                dy = ue_pos[1] - (cy + offset[1])
                dz = ue_pos[2] - self.cell_height
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                min_dist = min(min_dist, dist)
            return min_dist
        else:
            dx = ue_pos[0] - cx
            dy = ue_pos[1] - cy
            dz = ue_pos[2] - self.cell_height
            return np.sqrt(dx**2 + dy**2 + dz**2)

    def compute_distance_2d(self, ue_pos: np.ndarray,
                             cell_idx: int) -> float:
        """计算 2D 距离 (含 wrap-around)"""
        cx, cy = self.cell_positions[cell_idx]

        if self.wraparound:
            min_dist = float('inf')
            for offset in self._mirror_offsets:
                dx = ue_pos[0] - (cx + offset[0])
                dy = ue_pos[1] - (cy + offset[1])
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = min(min_dist, dist)
            return min_dist
        else:
            dx = ue_pos[0] - cx
            dy = ue_pos[1] - cy
            return np.sqrt(dx**2 + dy**2)

    def find_serving_cell(self, ue_pos: np.ndarray,
                          pathloss_func=None,
                          carrier_freq_ghz: float = 3.5) -> int:
        """找到服务小区 (最小路径损耗 / 最大 RSRP)

        Returns:
            serving cell index
        """
        min_pl = float('inf')
        best_cell = 0

        for cell_idx in range(self.num_cells):
            d_2d = self.compute_distance_2d(ue_pos, cell_idx)
            d_2d = max(d_2d, 10.0)

            if pathloss_func:
                pl = pathloss_func(d_2d, self.cell_height, ue_pos[2],
                                   carrier_freq_ghz, True)
            else:
                # 简化: 自由空间路径损耗
                d_3d = np.sqrt(d_2d**2 + (self.cell_height - ue_pos[2])**2)
                pl = 32.4 + 20 * np.log10(d_3d) + 20 * np.log10(carrier_freq_ghz)

            if pl < min_pl:
                min_pl = pl
                best_cell = cell_idx

        return best_cell

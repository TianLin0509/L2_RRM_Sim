"""可复现随机数管理"""

import numpy as np


class SimRNG:
    """仿真随机数生成器

    基于 numpy.random.Generator，通过 SeedSequence 为每个子系统
    生成独立的子生成器，保证跨子系统的可复现性。
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._ss = np.random.SeedSequence(seed)
        # 为各子系统生成独立的子种子
        child_seeds = self._ss.spawn(5)
        self.channel = np.random.Generator(np.random.PCG64(child_seeds[0]))
        self.traffic = np.random.Generator(np.random.PCG64(child_seeds[1]))
        self.phy = np.random.Generator(np.random.PCG64(child_seeds[2]))
        self.scheduler = np.random.Generator(np.random.PCG64(child_seeds[3]))
        self.general = np.random.Generator(np.random.PCG64(child_seeds[4]))

    @property
    def seed(self) -> int:
        return self._seed

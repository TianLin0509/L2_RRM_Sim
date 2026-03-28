"""5G NR 资源网格管理

基于 TS 38.211/38.214 的资源网格和开销计算。
"""

from ..config.sim_config import CarrierConfig
from ..utils.nr_utils import compute_num_re_per_prb, compute_tbs


class ResourceGrid:
    """5G NR 资源网格

    管理 PRB/RE 级资源，计算可用 RE 数和 TBS。
    """

    def __init__(self, carrier_config: CarrierConfig):
        self.scs_khz = carrier_config.subcarrier_spacing
        self.num_prb = carrier_config.num_prb
        self.num_pdcch_symbols = carrier_config.num_pdcch_symbols
        self.dmrs_type = carrier_config.dmrs_type
        self.dmrs_cdm_groups = carrier_config.dmrs_cdm_groups
        self.num_dmrs_symbols = carrier_config.num_dmrs_symbols

        # 预计算每 PRB 可用 RE 数
        self._num_re_per_prb = compute_num_re_per_prb(
            num_pdcch_symbols=self.num_pdcch_symbols,
            dmrs_type=self.dmrs_type,
            num_dmrs_cdm_groups=self.dmrs_cdm_groups,
            num_dmrs_symbols=self.num_dmrs_symbols
        )

    @property
    def num_re_per_prb(self) -> int:
        return self._num_re_per_prb

    def get_tbs(self, mcs_index: int, num_prbs: int, num_layers: int,
                mcs_table_index: int = 1) -> int:
        """计算 TBS"""
        return compute_tbs(
            self._num_re_per_prb, num_prbs,
            mcs_index, num_layers, mcs_table_index
        )

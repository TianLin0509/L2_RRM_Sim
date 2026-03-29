"""Sionna PHY 层适配器

用 Sionna 2.0 的 PHYAbstraction + OuterLoopLinkAdaptation
替换自实现的 EESM/ILLA/OLLA/BLER 模块。

Sionna 内部管线:
  OLLA: 维护 per-UE offset, 基于 HARQ ACK/NACK 调节, 内含 EESM+ILLA
  PHYAbstraction: MCS + sinr_eff + num_allocated_re → BLER查表 → decoded_bits
"""

import numpy as np

try:
    import torch
    from sionna.sys import PHYAbstraction, OuterLoopLinkAdaptation, EESM
    SIONNA_PHY_AVAILABLE = True
except ImportError:
    SIONNA_PHY_AVAILABLE = False


class SionnaPHY:
    """Sionna PHY 层封装

    将 Sionna 的 torch 接口包装为 numpy 接口。
    封装: EESM + ILLA + OLLA + PHYAbstraction
    """

    def __init__(self, num_ue: int,
                 bler_target: float = 0.1,
                 delta_up: float = 0.5,
                 offset_min: float = -10.0,
                 offset_max: float = 10.0,
                 mcs_table_index: int = 1,
                 mcs_category: int = 0,
                 device: str = None):
        if not SIONNA_PHY_AVAILABLE:
            raise ImportError("Sionna 2.0 未安装")

        if device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        self.num_ue = num_ue
        self.mcs_table_index = mcs_table_index
        self.mcs_category = mcs_category

        # Sionna 模块
        self._phy = PHYAbstraction(precision='single', device=self._device)
        self._eesm = EESM(precision='single', device=self._device)
        self._olla = OuterLoopLinkAdaptation(
            phy_abstraction=self._phy,
            num_ut=num_ue,
            bler_target=bler_target,
            delta_up=delta_up,
            offset_min=offset_min,
            offset_max=offset_max,
            precision='single',
            device=self._device,
        )

        # 状态缓存
        self._last_sinr_eff = np.ones(num_ue, dtype=np.float32)
        self._last_harq = np.ones(num_ue, dtype=np.int32)
        self._last_scheduled = np.zeros(num_ue, dtype=bool)

    def select_mcs(self, num_allocated_re: np.ndarray,
                   harq_feedback: np.ndarray = None,
                   sinr_eff: np.ndarray = None,
                   scheduled_mask: np.ndarray = None) -> np.ndarray:
        """通过 Sionna OLLA 选择 MCS"""
        if harq_feedback is None:
            harq_feedback = self._last_harq.copy()
        if scheduled_mask is not None:
            harq_feedback = harq_feedback.copy()
            harq_feedback[~scheduled_mask.astype(bool)] = -1
        if sinr_eff is None:
            sinr_eff = self._last_sinr_eff.copy()

        t_num_re = torch.tensor(num_allocated_re.reshape(1, -1),
                                dtype=torch.int32, device=self._device)
        t_harq = torch.tensor(harq_feedback.reshape(1, -1),
                              dtype=torch.int32, device=self._device)
        t_sinr = torch.tensor(sinr_eff.reshape(1, -1),
                              dtype=torch.float32, device=self._device)

        with torch.no_grad():
            t_mcs = self._olla(
                num_allocated_re=t_num_re,
                harq_feedback=t_harq,
                sinr_eff=t_sinr,
                mcs_table_index=self.mcs_table_index,
                mcs_category=self.mcs_category,
            )
        return t_mcs[0].cpu().numpy().astype(np.int32)

    def compute_sinr_eff(self, sinr_per_prb: np.ndarray,
                         mcs_indices: np.ndarray,
                         prb_assignment: np.ndarray = None) -> np.ndarray:
        """用 Sionna EESM 计算 per-UE 有效 SINR

        Args:
            sinr_per_prb: (num_ue, max_layers, num_prb) SINR [linear]
            mcs_indices: (num_ue,) MCS
            prb_assignment: (num_prb,) PRB 分配

        Returns:
            sinr_eff: (num_ue,) 有效 SINR [linear]
        """
        num_ue = self.num_ue
        num_layers = sinr_per_prb.shape[1]
        num_prb = sinr_per_prb.shape[2]

        # 只保留分配的 PRB 的 SINR
        sinr_masked = np.zeros_like(sinr_per_prb)
        if prb_assignment is not None:
            for ue in range(num_ue):
                ue_prbs = (prb_assignment == ue)
                sinr_masked[ue, :, :] = sinr_per_prb[ue, :, :] * ue_prbs[np.newaxis, :]
        else:
            sinr_masked = sinr_per_prb.copy()

        # 转换为 Sionna 格式: [batch=1, ofdm_sym=1, subcarrier, ue, stream]
        # sinr_masked: (ue, layer, prb) → (prb, ue, layer) → (1, 1, prb, ue, layer)
        sinr_t = np.transpose(sinr_masked, (2, 0, 1))  # (prb, ue, layer)
        sinr_t = sinr_t.reshape(1, 1, num_prb, num_ue, num_layers)

        t_sinr = torch.tensor(sinr_t, dtype=torch.float32, device=self._device)
        t_mcs = torch.tensor(mcs_indices.reshape(1, -1),
                             dtype=torch.int32, device=self._device)

        with torch.no_grad():
            t_sinr_eff = self._eesm(
                sinr=t_sinr, mcs_index=t_mcs,
                mcs_table_index=self.mcs_table_index,
                mcs_category=self.mcs_category,
            )
        return t_sinr_eff[0].cpu().numpy().astype(np.float32)

    def evaluate(self, mcs_indices: np.ndarray,
                 sinr_per_re: np.ndarray,
                 num_allocated_re: np.ndarray,
                 prb_assignment: np.ndarray = None) -> dict:
        """评估 PHY 传输结果

        使用 sinr_eff 模式调用 PHYAbstraction (先算 EESM, 再查 BLER)。
        """
        num_ue = self.num_ue

        # 1. 用 Sionna EESM 计算 per-UE sinr_eff
        sinr_eff = self.compute_sinr_eff(sinr_per_re, mcs_indices, prb_assignment)

        # 2. 用 sinr_eff 模式调用 PHYAbstraction
        t_mcs = torch.tensor(mcs_indices.reshape(1, -1),
                             dtype=torch.int32, device=self._device)
        t_sinr_eff = torch.tensor(sinr_eff.reshape(1, -1),
                                  dtype=torch.float32, device=self._device)
        t_num_re = torch.tensor(num_allocated_re.reshape(1, -1),
                                dtype=torch.int32, device=self._device)

        with torch.no_grad():
            decoded_bits, harq, se, tbler, bler = self._phy(
                mcs_index=t_mcs,
                sinr_eff=t_sinr_eff,
                num_allocated_re=t_num_re,
                mcs_table_index=self.mcs_table_index,
                mcs_category=self.mcs_category,
            )

        result = {
            'decoded_bits': decoded_bits[0].cpu().numpy().astype(np.int64),
            'harq_feedback': harq[0].cpu().numpy().astype(np.int32),
            'sinr_eff': se[0].cpu().numpy().astype(np.float32),
            'tbler': tbler[0].cpu().numpy().astype(np.float32),
            'bler': bler[0].cpu().numpy().astype(np.float32),
            'is_success': (harq[0].cpu().numpy() == 1),
        }

        # 更新缓存
        self._last_sinr_eff = result['sinr_eff'].copy()
        self._last_harq = result['harq_feedback'].copy()
        self._last_scheduled = (num_allocated_re > 0)

        return result

    @property
    def olla_offsets(self) -> np.ndarray:
        return self._olla._offset.cpu().numpy().ravel()

"""SINR 预估模块

基于 CSI 反馈 (CQI) + BFGain 进行 SINR 预估:

SU 场景:
  SINR_pred = SINR_cqi + BFGain_dB
  其中:
    SINR_cqi = CQI 反馈对应的 SINR (通过 CQI→SINR 映射表)
    BFGain = SVD_gain / PMI_gain (dB)
           = 使用 SRS/SVD 权值相较于 PMI 权值的能量增益

MU 场景:
  SINR_pred_mu = SINR_pred_su - MU_penalty_dB
  MU_penalty = 10log10(num_co_ue) + inter_layer_leakage_dB
  其中 inter_layer_leakage 取决于 ZF 预编码的精度

SINR 预估结果用于:
1. MCS 选择 (替代 OLLA 的 blind 方式)
2. 调度决策 (achievable rate 估计)
"""

import numpy as np
from .cqi_table import cqi_to_sinr, sinr_to_cqi
from .codebook import TypeICodebook
from ..utils.math_utils import linear_to_db, db_to_linear


class SINRPredictor:
    """SINR 预估器

    结合 CSI 反馈和 BFGain 进行精确的 SINR 预测。
    """

    def __init__(self, num_ue: int, num_tx_ports: int = 4,
                 codebook: TypeICodebook = None,
                 cqi_table_index: int = 1):
        self.num_ue = num_ue
        self.num_tx_ports = num_tx_ports
        self.codebook = codebook or TypeICodebook(num_tx_ports)
        self.cqi_table_index = cqi_table_index

        # 缓存每 UE 的 BFGain
        self._bf_gain_db = np.zeros(num_ue)
        self._svd_gain = np.zeros(num_ue)
        self._pmi_gain = np.zeros(num_ue)

    def compute_bf_gain(self, H: np.ndarray, pmi: int,
                        num_layers: int = 1) -> dict:
        """计算 BFGain = SVD增益 / PMI增益

        BFGain 反映了使用 SRS (SVD) 权值相较于 PMI codebook 权值
        在接收端的能量增益。

        Args:
            H: (num_rx_ant, num_tx_ports) 或 (num_rx_ant, num_tx_ports, num_prb)
            pmi: PMI 索引
            num_layers: 传输层数

        Returns:
            dict: svd_gain, pmi_gain, bf_gain_db, bf_gain_linear
        """
        # 处理宽带信道
        if H.ndim == 3:
            H_wb = np.mean(H, axis=2)  # 宽带平均
        else:
            H_wb = H

        # --- SVD 权值增益 ---
        U, S, Vh = np.linalg.svd(H_wb, full_matrices=False)
        # SVD 预编码: W_svd = Vh[:num_layers, :].conj().T → (tx_ports, layers)
        W_svd = Vh[:num_layers, :].conj().T
        H_eff_svd = H_wb @ W_svd  # (rx_ant, layers)
        svd_gain = np.sum(np.abs(H_eff_svd) ** 2) / num_layers

        # --- PMI 权值增益 ---
        W_pmi = self.codebook.get_precoding_matrix(pmi, num_layers)
        H_eff_pmi = H_wb @ W_pmi  # (rx_ant, layers)
        pmi_gain = np.sum(np.abs(H_eff_pmi) ** 2) / num_layers

        # --- BFGain ---
        if pmi_gain > 0:
            bf_gain_linear = svd_gain / pmi_gain
        else:
            bf_gain_linear = 1.0

        bf_gain_db = linear_to_db(bf_gain_linear)

        return {
            'svd_gain': float(svd_gain),
            'pmi_gain': float(pmi_gain),
            'bf_gain_linear': float(bf_gain_linear),
            'bf_gain_db': float(bf_gain_db),
        }

    def compute_bf_gain_subband(self, H: np.ndarray, pmi_subband: np.ndarray,
                                num_layers: int = 1,
                                subband_size_prb: int = 4) -> dict:
        """基于子带 PMI 计算平均波束增益。"""
        if H.ndim != 3 or pmi_subband is None or len(pmi_subband) == 0:
            return self.compute_bf_gain(H, 0, num_layers)

        sb_size = max(int(subband_size_prb), 1)
        svd_gain_acc = 0.0
        pmi_gain_acc = 0.0
        count = 0

        for sb_idx, prb_start in enumerate(range(0, H.shape[2], sb_size)):
            prb_end = min(prb_start + sb_size, H.shape[2])
            H_sb = H[:, :, prb_start:prb_end]
            H_wb = np.mean(H_sb, axis=2)

            U, S, Vh = np.linalg.svd(H_wb, full_matrices=False)
            W_svd = Vh[:num_layers, :].conj().T
            H_eff_svd = H_wb @ W_svd
            svd_gain = np.sum(np.abs(H_eff_svd) ** 2) / num_layers

            pmi = int(pmi_subband[min(sb_idx, len(pmi_subband) - 1)])
            W_pmi = self.codebook.get_precoding_matrix(pmi, num_layers)
            H_eff_pmi = H_wb @ W_pmi
            pmi_gain = np.sum(np.abs(H_eff_pmi) ** 2) / num_layers

            svd_gain_acc += svd_gain
            pmi_gain_acc += pmi_gain
            count += 1

        svd_gain_avg = svd_gain_acc / max(count, 1)
        pmi_gain_avg = pmi_gain_acc / max(count, 1)
        bf_gain_linear = svd_gain_avg / pmi_gain_avg if pmi_gain_avg > 0 else 1.0
        bf_gain_db = linear_to_db(bf_gain_linear)

        return {
            'svd_gain': float(svd_gain_avg),
            'pmi_gain': float(pmi_gain_avg),
            'bf_gain_linear': float(bf_gain_linear),
            'bf_gain_db': float(bf_gain_db),
        }

    def predict_sinr_su(self, cqi: int, bf_gain_db: float) -> float:
        """SU 场景 SINR 预估

        SINR_pred = SINR_cqi + BFGain_dB

        Args:
            cqi: UE 反馈的 CQI (0-15)
            bf_gain_db: BFGain (dB)

        Returns:
            predicted SINR (dB)
        """
        sinr_cqi_db = cqi_to_sinr(cqi, self.cqi_table_index)
        if cqi == 0:
            return -30.0  # out of range
        return sinr_cqi_db + bf_gain_db

    def predict_sinr_mu(self, cqi: int, bf_gain_db: float,
                        num_co_ue: int = 2,
                        inter_layer_leakage_db: float = -15.0) -> float:
        """MU 场景 SINR 预估

        SINR_pred_mu = SINR_pred_su - MU_penalty
        MU_penalty = power_sharing_loss + residual_interference

        power_sharing_loss: 功率分摊给多个 UE → 10log10(num_co_ue) dB
        residual_interference: ZF 后残余干扰 (取决于 ZF 精度和信道正交性)

        Args:
            cqi: CQI
            bf_gain_db: BFGain (dB)
            num_co_ue: 共调度 UE 数
            inter_layer_leakage_db: ZF 残余干扰 (dB, 负值, 如 -15 dB)

        Returns:
            predicted MU-SINR (dB)
        """
        sinr_su_db = self.predict_sinr_su(cqi, bf_gain_db)

        # 功率分摊损失
        power_sharing_loss_db = 10.0 * np.log10(max(num_co_ue, 1))

        # 残余干扰引起的 SINR 损失
        # 有效 SINR = S / (I_residual + N)
        # I_residual / N = 10^(leak_db/10) 相对于噪声的干扰比
        sinr_su_linear = db_to_linear(sinr_su_db)
        i_residual_ratio = db_to_linear(inter_layer_leakage_db)  # 残余干扰/信号比

        # MU-SINR = (S/num_co_ue) / (N + I_residual)
        # = sinr_su / num_co_ue / (1 + sinr_su * i_residual_ratio)
        sinr_mu_linear = (sinr_su_linear / num_co_ue
                          / (1.0 + sinr_su_linear * i_residual_ratio))

        return linear_to_db(sinr_mu_linear) if sinr_mu_linear > 0 else -30.0

    def predict_all_ue(self, csi_reports: list,
                       channel_matrices: np.ndarray = None,
                       mode: str = 'su') -> np.ndarray:
        """批量预估所有 UE 的 SINR

        Args:
            csi_reports: list of CSIReport (每 UE 一个)
            channel_matrices: (num_ue, rx_ant, tx_ports, num_prb) 当前信道
                              (用于计算 BFGain, 可选)
            mode: 'su' 或 'mu'

        Returns:
            sinr_pred_db: (num_ue,) 预估 SINR (dB)
        """
        sinr_pred_db = np.full(self.num_ue, -30.0)

        for ue in range(self.num_ue):
            report = csi_reports[ue]
            if report is None:
                continue

            # 计算 BFGain (如果有当前信道矩阵)
            if channel_matrices is not None:
                H_ue = channel_matrices[ue]  # (rx_ant, tx_ports, num_prb)
                if getattr(report, 'pmi_subband', None) is not None:
                    num_subbands = max(len(report.pmi_subband), 1)
                    subband_size_prb = int(np.ceil(H_ue.shape[2] / num_subbands))
                    bf = self.compute_bf_gain_subband(
                        H_ue, report.pmi_subband, report.ri,
                        subband_size_prb=subband_size_prb,
                    )
                else:
                    bf = self.compute_bf_gain(H_ue, report.pmi, report.ri)
                self._bf_gain_db[ue] = bf['bf_gain_db']
                self._svd_gain[ue] = bf['svd_gain']
                self._pmi_gain[ue] = bf['pmi_gain']
            else:
                # 使用缓存的 BFGain
                pass

            if mode == 'su':
                sinr_pred_db[ue] = self.predict_sinr_su(
                    report.cqi_wideband, self._bf_gain_db[ue]
                )
            else:
                sinr_pred_db[ue] = self.predict_sinr_mu(
                    report.cqi_wideband, self._bf_gain_db[ue],
                    num_co_ue=2  # 默认 2 UE 配对
                )

        return sinr_pred_db

    @property
    def bf_gain_db(self) -> np.ndarray:
        return self._bf_gain_db.copy()

    @property
    def svd_gain(self) -> np.ndarray:
        return self._svd_gain.copy()

    @property
    def pmi_gain(self) -> np.ndarray:
        return self._pmi_gain.copy()

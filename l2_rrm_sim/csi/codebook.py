"""Type I 单面板 Codebook (3GPP TS 38.214 Section 5.2.2.2.1)

预编码矩阵结构: W = W1 × W2

W1: 宽带/长期 — 选择波束方向 (DFT 向量)
W2: 窄带/短期 — 极化间相位选择

支持: 2/4/8/16/32 CSI-RS 端口, rank 1-4
"""

import numpy as np


class TypeICodebook:
    """Type I 单面板 Codebook

    基于 DFT 向量的预编码码本。
    每个 PMI 由 (i1, i2) 索引:
    - i1 = (i1_1, i1_2): 宽带波束方向索引
    - i2: 窄带相位选择索引

    简化实现: 使用 DFT 码本 (oversampled DFT codebook)
    """

    def __init__(self, num_tx_ports: int, num_layers_max: int = 4,
                 oversampling: int = 1):
        """
        Args:
            num_tx_ports: CSI-RS 端口数 (2, 4, 8, 16, 32)
            num_layers_max: 最大层数
            oversampling: DFT oversampling factor (O1/O2)
        """
        self.num_tx_ports = num_tx_ports
        self.num_layers_max = min(num_layers_max, num_tx_ports)
        self.oversampling = oversampling

        # 生成 DFT 码本
        # 对于双极化天线: N1 × N2 × 2pol
        # 简化: 单极化 DFT 码本
        self._codebook = self._generate_dft_codebook()
        self.num_codewords = self._codebook.shape[0]

    def _generate_dft_codebook(self) -> np.ndarray:
        """生成 DFT 码本

        Returns:
            codebook: (num_codewords, num_tx_ports) complex
        """
        N = self.num_tx_ports
        O = self.oversampling
        num_beams = N * O

        codebook = np.zeros((num_beams, N), dtype=complex)
        for b in range(num_beams):
            for n in range(N):
                codebook[b, n] = np.exp(1j * 2 * np.pi * n * b / num_beams)
            codebook[b] /= np.sqrt(N)  # 功率归一化

        return codebook

    def get_precoding_matrix(self, pmi: int, num_layers: int = 1) -> np.ndarray:
        """获取预编码矩阵

        Args:
            pmi: PMI 索引 (0 ~ num_codewords-1)
            num_layers: 传输层数

        Returns:
            W: (num_tx_ports, num_layers) complex
        """
        pmi = pmi % self.num_codewords

        if num_layers == 1:
            # Rank 1: 直接使用 DFT 波束向量
            return self._codebook[pmi, :, np.newaxis]

        elif num_layers <= self.num_layers_max:
            # Rank > 1: 选择相邻的 DFT 波束组合
            W = np.zeros((self.num_tx_ports, num_layers), dtype=complex)
            for l in range(num_layers):
                beam_idx = (pmi + l) % self.num_codewords
                W[:, l] = self._codebook[beam_idx]
            return W
        else:
            raise ValueError(f"num_layers={num_layers} > max={self.num_layers_max}")

    def select_best_pmi(self, H: np.ndarray, num_layers: int = 1) -> tuple:
        """从信道矩阵选择最佳 PMI (穷搜)

        遍历所有 codeword, 选择使接收 SINR/容量最大的 PMI。

        Args:
            H: (num_rx_ant, num_tx_ports) 信道矩阵 (单 PRB 或宽带)
            num_layers: 传输层数

        Returns:
            (best_pmi, best_W, best_sinr_gain):
                best_pmi: 最佳 PMI 索引
                best_W: 最佳预编码矩阵 (num_tx_ports, num_layers)
                best_sinr_gain: ||H × W||² (线性增益)
        """
        best_pmi = 0
        best_gain = 0.0
        best_W = None

        for pmi in range(self.num_codewords):
            W = self.get_precoding_matrix(pmi, num_layers)
            # 有效信道: H_eff = H × W, shape (num_rx_ant, num_layers)
            H_eff = H @ W
            # 增益: ||H_eff||_F²
            gain = np.sum(np.abs(H_eff) ** 2)

            if gain > best_gain:
                best_gain = gain
                best_pmi = pmi
                best_W = W.copy()

        return best_pmi, best_W, best_gain

    def select_best_pmi_wideband(self, H_prb: np.ndarray,
                                  num_layers: int = 1) -> tuple:
        """宽带 PMI 选择 (跨所有 PRB 联合优化)

        Args:
            H_prb: (num_rx_ant, num_tx_ports, num_prb) 信道矩阵

        Returns:
            (best_pmi, best_W, avg_gain)
        """
        num_prb = H_prb.shape[2]
        best_pmi = 0
        best_avg_gain = 0.0
        best_W = None

        for pmi in range(self.num_codewords):
            W = self.get_precoding_matrix(pmi, num_layers)
            total_gain = 0.0
            for prb in range(num_prb):
                H = H_prb[:, :, prb]
                H_eff = H @ W
                total_gain += np.sum(np.abs(H_eff) ** 2)
            avg_gain = total_gain / num_prb

            if avg_gain > best_avg_gain:
                best_avg_gain = avg_gain
                best_pmi = pmi
                best_W = W.copy()

        return best_pmi, best_W, best_avg_gain

    def select_best_pmi_subband(self, H_prb: np.ndarray,
                                num_layers: int = 1,
                                subband_size_prb: int = 4) -> tuple:
        """按子带选择最佳 PMI。"""
        num_prb = H_prb.shape[2]
        sb_size = max(int(subband_size_prb), 1)
        subband_pmi = []
        subband_w = []
        subband_gain = []

        for prb_start in range(0, num_prb, sb_size):
            prb_end = min(prb_start + sb_size, num_prb)
            H_sb = H_prb[:, :, prb_start:prb_end]
            best_pmi, best_W, best_avg_gain = self.select_best_pmi_wideband(
                H_sb, num_layers=num_layers
            )
            subband_pmi.append(best_pmi)
            subband_w.append(best_W)
            subband_gain.append(best_avg_gain)

        return (
            np.asarray(subband_pmi, dtype=np.int32),
            subband_w,
            np.asarray(subband_gain, dtype=np.float64),
        )

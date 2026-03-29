"""CSI 反馈管理

模拟 3GPP NR CSI 测量与反馈流程:
1. 基站周期性下发 CSI-RS
2. UE 测量信道, 计算 CQI/RI/PMI
3. UE 通过 PUCCH/PUSCH 反馈 CSI 报告
4. 基站接收后用于调度和链路自适应

支持:
- 周期性 CSI 报告 (configurable period)
- 反馈延迟建模 (N slots)
- 宽带 CQI/PMI + RI
"""

import numpy as np
from dataclasses import dataclass, field
from .cqi_table import sinr_to_cqi, cqi_to_sinr, sinr_to_cqi_batch
from .codebook import TypeICodebook
from ..utils.math_utils import linear_to_db, db_to_linear


@dataclass
class CSIReport:
    """单个 UE 的 CSI 报告"""
    ue_id: int
    slot_idx: int               # 测量时的 slot
    ri: int = 1                 # Rank Indicator (1-4)
    pmi: int = 0                # Precoding Matrix Indicator
    cqi_wideband: int = 0       # 宽带 CQI (0-15)
    cqi_subband: np.ndarray = None  # 子带 CQI (可选)
    sinr_per_layer_db: np.ndarray = None  # per-layer SINR (PMI 后)
    precoding_matrix: np.ndarray = None  # PMI 对应的预编码矩阵 W


class CSIFeedbackManager:
    """CSI 反馈管理器

    管理所有 UE 的 CSI 测量、报告生成和反馈延迟。
    """

    def __init__(self, num_ue: int, num_tx_ports: int = 4,
                 max_rank: int = 4,
                 csi_period_slots: int = 10,
                 feedback_delay_slots: int = 4,
                 cqi_table_index: int = 1,
                 codebook_oversampling: int = 1,
                 noise_power_per_prb: float = 1e-13):
        """
        Args:
            num_ue: UE 数
            num_tx_ports: CSI-RS 端口数
            max_rank: 最大 rank
            csi_period_slots: CSI-RS 周期 (slots)
            feedback_delay_slots: 反馈延迟 (slots)
            noise_power_per_prb: 每 PRB 噪声功率 (W)
        """
        self.num_ue = num_ue
        self.num_tx_ports = num_tx_ports
        self.max_rank = min(max_rank, num_tx_ports)
        self.csi_period = csi_period_slots
        self.feedback_delay = feedback_delay_slots
        self.cqi_table_index = cqi_table_index
        self.noise_power_per_prb = noise_power_per_prb

        # Codebook
        self.codebook = TypeICodebook(num_tx_ports, max_rank, codebook_oversampling)

        # 每 UE 最新的 CSI 报告 (带延迟)
        self._latest_reports = [None] * num_ue  # 基站当前可用的 CSI
        self._pending_reports = {}  # {delivery_slot: [CSIReport, ...]}

        # 上一次 CSI 测量的 slot
        self._last_csi_slot = -csi_period_slots

    def should_measure(self, slot_idx: int) -> bool:
        """是否到了 CSI 测量周期"""
        return (slot_idx - self._last_csi_slot) >= self.csi_period

    def measure_and_report(self, slot_idx: int,
                           channel_matrices: np.ndarray,
                           tx_power_per_prb: float) -> list:
        """执行 CSI 测量, 生成报告

        Args:
            slot_idx: 当前 slot
            channel_matrices: (num_ue, num_rx_ant, num_tx_ports, num_prb) 信道矩阵
            tx_power_per_prb: 每 PRB 发射功率 (W)

        Returns:
            reports: list of CSIReport
        """
        self._last_csi_slot = slot_idx
        reports = []
        num_prb = channel_matrices.shape[3]

        for ue in range(self.num_ue):
            H_ue = channel_matrices[ue]  # (rx_ant, tx_ports, num_prb)
            num_rx = H_ue.shape[0]

            # --- RI 选择 ---
            # 对宽带信道做 SVD, 根据奇异值选择 rank
            H_wb = np.mean(H_ue, axis=2)  # (rx_ant, tx_ports) 宽带平均
            U, S, Vh = np.linalg.svd(H_wb, full_matrices=False)
            ri = self._select_ri(S, tx_power_per_prb, num_prb)

            # --- PMI 选择 ---
            # 宽带 PMI: 遍历 codebook 找最佳
            best_pmi, best_W, pmi_gain = self.codebook.select_best_pmi_wideband(
                H_ue, num_layers=ri
            )

            # --- CQI 计算 ---
            # 使用 PMI 后的有效信道计算 per-PRB SINR, 再映射到 CQI
            sinr_per_layer_per_prb = np.zeros((ri, num_prb))
            for prb in range(num_prb):
                H_prb = H_ue[:, :, prb]  # (rx_ant, tx_ports)
                H_eff = H_prb @ best_W   # (rx_ant, ri)
                for l in range(ri):
                    signal = tx_power_per_prb * np.sum(np.abs(H_eff[:, l]) ** 2)
                    sinr_per_layer_per_prb[l, prb] = signal / (ri * self.noise_power_per_prb)

            # 宽带 SINR (各层最低值的 PRB 平均, 保守估计)
            avg_sinr_per_layer = np.mean(sinr_per_layer_per_prb, axis=1)  # (ri,)
            # CQI 基于最差层的 SINR
            min_layer_sinr_db = linear_to_db(np.min(avg_sinr_per_layer))
            cqi_wideband = sinr_to_cqi(min_layer_sinr_db, self.cqi_table_index)

            report = CSIReport(
                ue_id=ue,
                slot_idx=slot_idx,
                ri=ri,
                pmi=best_pmi,
                cqi_wideband=cqi_wideband,
                sinr_per_layer_db=linear_to_db(avg_sinr_per_layer),
                precoding_matrix=best_W,
            )
            reports.append(report)

        # 加入延迟队列
        delivery_slot = slot_idx + self.feedback_delay
        self._pending_reports.setdefault(delivery_slot, []).extend(reports)

        return reports

    def receive_feedback(self, slot_idx: int):
        """接收到期的 CSI 反馈 (基站侧)

        Returns:
            received: list of CSIReport (本 slot 到期的报告)
        """
        received = self._pending_reports.pop(slot_idx, [])
        for report in received:
            self._latest_reports[report.ue_id] = report
        return received

    def get_latest_report(self, ue_id: int) -> CSIReport:
        """获取某 UE 最新可用的 CSI 报告"""
        return self._latest_reports[ue_id]

    def get_all_latest_reports(self) -> list:
        """获取所有 UE 的最新 CSI"""
        return self._latest_reports.copy()

    def _select_ri(self, singular_values: np.ndarray,
                   tx_power: float, num_prb: int) -> int:
        """基于 SVD 奇异值选择 RI

        选择使总容量最大的 rank。
        """
        S = singular_values
        max_rank = min(self.max_rank, len(S))
        best_ri = 1
        best_capacity = 0.0

        for r in range(1, max_rank + 1):
            # 等功率分配: 每层 P/r
            capacity = 0.0
            for l in range(r):
                sinr_l = tx_power * num_prb * S[l]**2 / (r * self.noise_power_per_prb * num_prb)
                capacity += np.log2(1.0 + max(sinr_l, 0))

            if capacity > best_capacity:
                best_capacity = capacity
                best_ri = r

        return best_ri

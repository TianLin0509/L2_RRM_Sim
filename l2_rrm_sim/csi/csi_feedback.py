"""CSI feedback management.

Models a simplified 3GPP NR CSI measurement and reporting flow:
1. gNB periodically transmits CSI-RS
2. UE measures channel and computes CQI/RI/PMI
3. UE reports CSI with configurable delay
4. gNB consumes the latest available report for scheduling/link adaptation
"""

import numpy as np
from dataclasses import dataclass
from .cqi_table import sinr_to_cqi
from .codebook import TypeICodebook
from ..utils.math_utils import linear_to_db


@dataclass
class CSIReport:
    """Single-UE CSI report."""
    ue_id: int
    slot_idx: int
    ri: int = 1
    pmi: int = 0
    pmi_subband: np.ndarray = None
    cqi_wideband: int = 0
    cqi_subband: np.ndarray = None
    sinr_per_layer_db: np.ndarray = None
    precoding_matrix: np.ndarray = None
    precoding_matrices_subband: list = None


class CSIFeedbackManager:
    """Manages CSI measurement, delayed delivery, and latest reports."""

    def __init__(self, num_ue: int, num_tx_ports: int = 4,
                 max_rank: int = 4,
                 csi_period_slots: int = 10,
                 feedback_delay_slots: int = 4,
                 cqi_table_index: int = 1,
                 codebook_oversampling: int = 1,
                 noise_power_per_prb: float = 1e-13,
                 subband_size_prb: int = 4):
        self.num_ue = num_ue
        self.num_tx_ports = num_tx_ports
        self.max_rank = min(max_rank, num_tx_ports)
        self.csi_period = csi_period_slots
        self.feedback_delay = feedback_delay_slots
        self.cqi_table_index = cqi_table_index
        self.noise_power_per_prb = noise_power_per_prb
        self.subband_size_prb = max(int(subband_size_prb), 1)

        self.codebook = TypeICodebook(num_tx_ports, max_rank, codebook_oversampling)
        self._latest_reports = [None] * num_ue
        self._pending_reports = {}
        self._last_csi_slot = -csi_period_slots

    def should_measure(self, slot_idx: int) -> bool:
        return (slot_idx - self._last_csi_slot) >= self.csi_period

    def measure_and_report(self, slot_idx: int,
                           channel_matrices: np.ndarray,
                           tx_power_per_prb: float) -> list:
        self._last_csi_slot = slot_idx
        reports = []
        num_prb = channel_matrices.shape[3]

        for ue in range(self.num_ue):
            h_ue = channel_matrices[ue]
            h_wb = np.mean(h_ue, axis=2)
            _, singular_values, _ = np.linalg.svd(h_wb, full_matrices=False)
            ri = self._select_ri(singular_values, tx_power_per_prb)

            best_pmi, best_w, _ = self.codebook.select_best_pmi_wideband(
                h_ue, num_layers=ri
            )
            pmi_subband, w_subband, _ = self.codebook.select_best_pmi_subband(
                h_ue, num_layers=ri, subband_size_prb=self.subband_size_prb
            )

            sinr_per_layer_per_prb = np.zeros((ri, num_prb))
            for prb in range(num_prb):
                h_prb = h_ue[:, :, prb]
                sb_idx = min(prb // self.subband_size_prb, len(w_subband) - 1)
                h_eff = h_prb @ w_subband[sb_idx]
                for layer in range(ri):
                    signal = tx_power_per_prb * np.sum(np.abs(h_eff[:, layer]) ** 2)
                    sinr_per_layer_per_prb[layer, prb] = (
                        signal / (ri * self.noise_power_per_prb)
                    )

            avg_sinr_per_layer = np.mean(sinr_per_layer_per_prb, axis=1)
            min_layer_sinr_db = linear_to_db(np.min(avg_sinr_per_layer))
            cqi_wideband = sinr_to_cqi(min_layer_sinr_db, self.cqi_table_index)

            cqi_subband = []
            for prb_start in range(0, num_prb, self.subband_size_prb):
                prb_end = min(prb_start + self.subband_size_prb, num_prb)
                sb_sinr = np.mean(sinr_per_layer_per_prb[:, prb_start:prb_end], axis=1)
                sb_min_layer_sinr_db = linear_to_db(np.min(sb_sinr))
                cqi_subband.append(
                    sinr_to_cqi(sb_min_layer_sinr_db, self.cqi_table_index)
                )

            reports.append(CSIReport(
                ue_id=ue,
                slot_idx=slot_idx,
                ri=ri,
                pmi=best_pmi,
                pmi_subband=pmi_subband,
                cqi_wideband=cqi_wideband,
                cqi_subband=np.asarray(cqi_subband, dtype=np.int32),
                sinr_per_layer_db=linear_to_db(avg_sinr_per_layer),
                precoding_matrix=best_w,
                precoding_matrices_subband=w_subband,
            ))

        delivery_slot = slot_idx + self.feedback_delay
        self._pending_reports.setdefault(delivery_slot, []).extend(reports)
        return reports

    def receive_feedback(self, slot_idx: int):
        received = self._pending_reports.pop(slot_idx, [])
        for report in received:
            self._latest_reports[report.ue_id] = report
        return received

    def get_latest_report(self, ue_id: int) -> CSIReport:
        return self._latest_reports[ue_id]

    def get_all_latest_reports(self) -> list:
        return self._latest_reports.copy()

    def _select_ri(self, singular_values: np.ndarray,
                   tx_power: float) -> int:
        """Select the rank that maximizes a simple equal-power capacity metric."""
        max_rank = min(self.max_rank, len(singular_values))
        best_ri = 1
        best_capacity = 0.0

        for rank in range(1, max_rank + 1):
            capacity = 0.0
            for layer in range(rank):
                sinr_l = tx_power * singular_values[layer] ** 2 / (
                    rank * self.noise_power_per_prb
                )
                capacity += np.log2(1.0 + max(sinr_l, 0.0))
            if capacity > best_capacity:
                best_capacity = capacity
                best_ri = rank

        return best_ri

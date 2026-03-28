"""KPI 统计报告

计算小区级和用户级 KPI，支持掐头去尾 (trimmed mean)。
"""

import numpy as np
from .kpi_collector import KPICollector
from ..config.sim_config import CarrierConfig


class KPIReporter:
    """KPI 报告生成器"""

    def __init__(self, collector: KPICollector,
                 carrier_config: CarrierConfig,
                 trim_percent: float = 5.0):
        self.collector = collector
        self.carrier_config = carrier_config
        self.trim_percent = trim_percent

    def report(self) -> dict:
        """生成 KPI 报告"""
        c = self.collector
        s = c.get_valid_range()
        slot_duration_s = self.carrier_config.slot_duration_s

        # ---- 小区级 KPI ----
        cell_bits = c.cell_throughput_bits[s]
        cell_throughput_bps = cell_bits / slot_duration_s
        cell_avg_throughput_mbps = np.mean(cell_throughput_bps) / 1e6

        # ---- 用户级 KPI ----
        # 每 UE 平均吞吐量
        ue_bits = c.ue_throughput_bits[s]  # (valid_slots, num_ue)
        ue_avg_bits_per_slot = np.mean(ue_bits, axis=0)  # (num_ue,)
        ue_avg_throughput_mbps = ue_avg_bits_per_slot / slot_duration_s / 1e6

        # 掐头去尾平均
        ue_trimmed_throughput_mbps = self._trimmed_mean(
            ue_avg_throughput_mbps, self.trim_percent
        )

        # 小区边缘吞吐量 (5th percentile)
        cell_edge_throughput_mbps = float(np.percentile(ue_avg_throughput_mbps, 5))

        # ---- BLER ----
        # 只统计被调度的 slot
        scheduled_mask = c.ue_num_prbs[s] > 0  # (valid_slots, num_ue)
        if np.any(scheduled_mask):
            # 实际 BLER = 1 - 成功率
            tb_success = c.ue_tb_success[s]
            actual_bler_per_ue = np.zeros(c.num_ue)
            for ue in range(c.num_ue):
                ue_sched = scheduled_mask[:, ue]
                if np.any(ue_sched):
                    actual_bler_per_ue[ue] = 1.0 - np.mean(
                        tb_success[ue_sched, ue].astype(float)
                    )
            avg_bler = np.mean(actual_bler_per_ue)
        else:
            actual_bler_per_ue = np.zeros(c.num_ue)
            avg_bler = 0.0

        # ---- MCS 分布 ----
        mcs_data = c.ue_mcs[s]
        scheduled_mcs = mcs_data[scheduled_mask] if np.any(scheduled_mask) else mcs_data
        avg_mcs = float(np.mean(scheduled_mcs)) if len(scheduled_mcs) > 0 else 0.0

        # ---- SINR ----
        sinr_data = c.ue_sinr_eff_db[s]
        avg_sinr_db = float(np.mean(sinr_data))

        # ---- 频谱效率 ----
        bw_hz = self.carrier_config.bandwidth_mhz * 1e6
        spectral_eff = cell_avg_throughput_mbps * 1e6 / bw_hz

        # ---- Jain 公平指数 ----
        if np.sum(ue_avg_throughput_mbps) > 0:
            jain_fairness = (np.sum(ue_avg_throughput_mbps) ** 2
                             / (c.num_ue * np.sum(ue_avg_throughput_mbps ** 2)))
        else:
            jain_fairness = 1.0

        return {
            'cell_avg_throughput_mbps': cell_avg_throughput_mbps,
            'cell_edge_throughput_mbps': cell_edge_throughput_mbps,
            'ue_avg_throughput_mbps': ue_avg_throughput_mbps,
            'ue_trimmed_throughput_mbps': ue_trimmed_throughput_mbps,
            'avg_bler': avg_bler,
            'bler_per_ue': actual_bler_per_ue,
            'avg_mcs': avg_mcs,
            'avg_sinr_db': avg_sinr_db,
            'spectral_efficiency_bps_hz': spectral_eff,
            'jain_fairness': jain_fairness,
            'num_valid_slots': s.stop - s.start,
        }

    def _trimmed_mean(self, data: np.ndarray, percent: float) -> float:
        """掐头去尾平均值"""
        if len(data) == 0:
            return 0.0
        n = len(data)
        k = int(n * percent / 100.0)
        if k == 0:
            return float(np.mean(data))
        sorted_data = np.sort(data)
        return float(np.mean(sorted_data[k:-k])) if k < n // 2 else float(np.mean(data))

    def print_report(self, report: dict):
        """打印 KPI 报告"""
        print("=" * 60)
        print("  L2 RRM 仿真 KPI 报告")
        print("=" * 60)
        print(f"  有效 Slots: {report['num_valid_slots']}")
        print(f"  小区平均吞吐量:     {report['cell_avg_throughput_mbps']:.2f} Mbps")
        print(f"  小区边缘吞吐量(5%): {report['cell_edge_throughput_mbps']:.2f} Mbps")
        print(f"  UE 平均吞吐量(掐头去尾): {report['ue_trimmed_throughput_mbps']:.2f} Mbps")
        print(f"  平均 BLER:           {report['avg_bler']:.4f} "
              f"(目标: {self.collector.num_ue}×0.1)")
        print(f"  平均 MCS:            {report['avg_mcs']:.1f}")
        print(f"  平均有效 SINR:       {report['avg_sinr_db']:.1f} dB")
        print(f"  频谱效率:            {report['spectral_efficiency_bps_hz']:.2f} bps/Hz")
        print(f"  Jain 公平指数:       {report['jain_fairness']:.4f}")
        print("-" * 60)
        print("  UE 吞吐量分布 (Mbps):")
        ue_tp = report['ue_avg_throughput_mbps']
        print(f"    Min: {np.min(ue_tp):.2f},  5%: {np.percentile(ue_tp, 5):.2f}, "
              f" 50%: {np.median(ue_tp):.2f},  95%: {np.percentile(ue_tp, 95):.2f}, "
              f" Max: {np.max(ue_tp):.2f}")
        print("=" * 60)

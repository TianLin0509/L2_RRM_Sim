"""KPI 统计报告

计算小区级和用户级 KPI，包含体验速率指标。

体验速率定义:
- 小区级体验速率: 所有 UE 平均吞吐量的掐头去尾均值
- 用户级体验速率: 单 UE 在仿真时段内吞吐量的掐头去尾均值
- 小区边缘体验速率: 所有 UE 平均吞吐量的第 5 百分位
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
        """生成完整 KPI 报告"""
        c = self.collector
        s = c.get_valid_range()
        num_valid = s.stop - s.start
        if num_valid <= 0:
            return self._empty_report()

        slot_duration_s = self.carrier_config.slot_duration_s
        bw_hz = self.carrier_config.bandwidth_mhz * 1e6

        # ---- 原始数据 ----
        ue_bits = c.ue_throughput_bits[s]           # (valid_slots, num_ue)
        cell_bits = c.cell_throughput_bits[s]        # (valid_slots,)
        scheduled_mask = c.ue_num_prbs[s] > 0       # (valid_slots, num_ue)

        # ============================================================
        # 小区级 KPI
        # ============================================================
        cell_tp_bps = cell_bits / slot_duration_s
        cell_avg_throughput_mbps = float(np.mean(cell_tp_bps) / 1e6)
        spectral_eff = cell_avg_throughput_mbps * 1e6 / bw_hz

        # ============================================================
        # 用户级吞吐量
        # ============================================================
        # 每 UE 在整个仿真时段的平均吞吐量
        ue_avg_bits_per_slot = np.mean(ue_bits, axis=0)      # (num_ue,)
        ue_avg_tp_mbps = ue_avg_bits_per_slot / slot_duration_s / 1e6

        # ============================================================
        # 体验速率 (User Experienced Data Rate)
        # ============================================================
        # 1. 小区体验速率: UE 平均吞吐量的掐头去尾均值
        cell_experienced_rate_mbps = self._trimmed_mean(ue_avg_tp_mbps, self.trim_percent)

        # 2. 每 UE 体验速率: 每 UE 时间维度的掐头去尾均值
        ue_experienced_tp_mbps = np.zeros(c.num_ue)
        for ue in range(c.num_ue):
            ue_slot_tp = ue_bits[:, ue] / slot_duration_s / 1e6
            # 只统计被调度的 slot
            ue_sched_slots = scheduled_mask[:, ue]
            if np.sum(ue_sched_slots) > 0:
                scheduled_tp = ue_slot_tp[ue_sched_slots]
                ue_experienced_tp_mbps[ue] = self._trimmed_mean(scheduled_tp, self.trim_percent)
            else:
                ue_experienced_tp_mbps[ue] = 0.0

        # 3. 小区边缘体验速率: 5th percentile
        cell_edge_tp_mbps = float(np.percentile(ue_avg_tp_mbps, 5))

        # ============================================================
        # BLER
        # ============================================================
        actual_bler_per_ue = np.zeros(c.num_ue)
        for ue in range(c.num_ue):
            ue_sched = scheduled_mask[:, ue]
            if np.any(ue_sched):
                tb_success = c.ue_tb_success[s][ue_sched, ue].astype(float)
                actual_bler_per_ue[ue] = 1.0 - np.mean(tb_success)
        avg_bler = float(np.mean(actual_bler_per_ue))

        # ============================================================
        # MCS 分布
        # ============================================================
        mcs_data = c.ue_mcs[s]
        scheduled_mcs = mcs_data[scheduled_mask]
        avg_mcs = float(np.mean(scheduled_mcs)) if len(scheduled_mcs) > 0 else 0.0

        # MCS 分布统计
        mcs_distribution = {}
        if len(scheduled_mcs) > 0:
            for mcs_val in range(29):
                count = np.sum(scheduled_mcs == mcs_val)
                if count > 0:
                    mcs_distribution[mcs_val] = int(count)

        # ============================================================
        # SINR
        # ============================================================
        sinr_data = c.ue_sinr_eff_db[s]
        valid_sinr = sinr_data[scheduled_mask] if np.any(scheduled_mask) else sinr_data
        avg_sinr_db = float(np.mean(valid_sinr)) if len(valid_sinr) > 0 else -30.0

        # ============================================================
        # 公平性
        # ============================================================
        tp_sum = np.sum(ue_avg_tp_mbps)
        tp_sq_sum = np.sum(ue_avg_tp_mbps ** 2)
        jain_fairness = (tp_sum ** 2 / (c.num_ue * tp_sq_sum)) if tp_sq_sum > 0 else 1.0

        # ============================================================
        # 调度统计
        # ============================================================
        # 每 UE 的调度占比
        ue_scheduling_ratio = np.mean(scheduled_mask.astype(float), axis=0)
        avg_scheduling_ratio = float(np.mean(ue_scheduling_ratio))

        # PRB 利用率
        total_prb_per_slot = np.sum(c.ue_num_prbs[s], axis=1)
        prb_utilization = float(np.mean(total_prb_per_slot) / self.carrier_config.num_prb)

        return {
            # 小区级
            'cell_avg_throughput_mbps': cell_avg_throughput_mbps,
            'cell_edge_throughput_mbps': cell_edge_tp_mbps,
            'cell_experienced_rate_mbps': cell_experienced_rate_mbps,
            'spectral_efficiency_bps_hz': spectral_eff,
            # 用户级
            'ue_avg_throughput_mbps': ue_avg_tp_mbps,
            'ue_experienced_throughput_mbps': ue_experienced_tp_mbps,
            'ue_trimmed_throughput_mbps': cell_experienced_rate_mbps,
            # BLER
            'avg_bler': avg_bler,
            'bler_per_ue': actual_bler_per_ue,
            # MCS
            'avg_mcs': avg_mcs,
            'mcs_distribution': mcs_distribution,
            # SINR
            'avg_sinr_db': avg_sinr_db,
            # 公平性
            'jain_fairness': jain_fairness,
            # 调度
            'avg_scheduling_ratio': avg_scheduling_ratio,
            'ue_scheduling_ratio': ue_scheduling_ratio,
            'prb_utilization': prb_utilization,
            # 元数据
            'num_valid_slots': num_valid,
            'num_ue': c.num_ue,
        }

    def _trimmed_mean(self, data: np.ndarray, percent: float) -> float:
        """掐头去尾平均值

        去掉最低和最高 percent% 的数据后计算均值。
        """
        data = np.asarray(data).ravel()
        n = len(data)
        if n == 0:
            return 0.0
        k = int(n * percent / 100.0)
        if k <= 0 or n <= 2:
            return float(np.mean(data))
        sorted_data = np.sort(data)
        trimmed = sorted_data[k:n - k]
        if len(trimmed) == 0:
            return float(np.mean(data))
        return float(np.mean(trimmed))

    def _empty_report(self) -> dict:
        return {
            'cell_avg_throughput_mbps': 0, 'cell_edge_throughput_mbps': 0,
            'cell_experienced_rate_mbps': 0, 'spectral_efficiency_bps_hz': 0,
            'ue_avg_throughput_mbps': np.array([]), 'ue_experienced_throughput_mbps': np.array([]),
            'ue_trimmed_throughput_mbps': 0, 'avg_bler': 0, 'bler_per_ue': np.array([]),
            'avg_mcs': 0, 'mcs_distribution': {}, 'avg_sinr_db': 0,
            'jain_fairness': 0, 'avg_scheduling_ratio': 0,
            'ue_scheduling_ratio': np.array([]), 'prb_utilization': 0,
            'num_valid_slots': 0, 'num_ue': 0,
        }

    def print_report(self, report: dict):
        """打印 KPI 报告"""
        print("=" * 65)
        print("  L2 RRM 仿真 KPI 报告")
        print("=" * 65)
        print(f"  有效 Slots: {report['num_valid_slots']}, UE 数: {report['num_ue']}")
        print("-" * 65)
        print("  [小区级]")
        print(f"    平均吞吐量:       {report['cell_avg_throughput_mbps']:.2f} Mbps")
        print(f"    频谱效率:         {report['spectral_efficiency_bps_hz']:.2f} bps/Hz")
        print(f"    PRB 利用率:       {report['prb_utilization']*100:.1f}%")
        print("-" * 65)
        print("  [体验速率]")
        print(f"    小区体验速率 (掐头去尾): {report['cell_experienced_rate_mbps']:.2f} Mbps")
        print(f"    小区边缘体验速率 (5%ile): {report['cell_edge_throughput_mbps']:.2f} Mbps")
        print("-" * 65)
        print("  [链路质量]")
        print(f"    平均 BLER:        {report['avg_bler']:.4f} (目标 0.1)")
        print(f"    平均 MCS:         {report['avg_mcs']:.1f}")
        print(f"    平均有效 SINR:    {report['avg_sinr_db']:.1f} dB")
        print("-" * 65)
        print("  [调度与公平性]")
        print(f"    Jain 公平指数:    {report['jain_fairness']:.4f}")
        print(f"    平均调度占比:     {report['avg_scheduling_ratio']*100:.1f}%")
        print("-" * 65)
        ue_tp = report['ue_avg_throughput_mbps']
        if len(ue_tp) > 0:
            print("  [UE 吞吐量分布 (Mbps)]")
            print(f"    Min: {np.min(ue_tp):.2f},  5%: {np.percentile(ue_tp, 5):.2f}, "
                  f" 50%: {np.median(ue_tp):.2f},  95%: {np.percentile(ue_tp, 95):.2f}, "
                  f" Max: {np.max(ue_tp):.2f}")
            ue_exp = report['ue_experienced_throughput_mbps']
            if len(ue_exp) > 0:
                print("  [UE 体验速率分布 (Mbps)]")
                print(f"    Min: {np.min(ue_exp):.2f},  5%: {np.percentile(ue_exp, 5):.2f}, "
                      f" 50%: {np.median(ue_exp):.2f},  95%: {np.percentile(ue_exp, 95):.2f}, "
                      f" Max: {np.max(ue_exp):.2f}")
        print("=" * 65)

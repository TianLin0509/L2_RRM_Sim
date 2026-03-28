"""KPI 可视化绘图

CDF 曲线、时间序列、OLLA 收敛等。
"""

import numpy as np
from .kpi_collector import KPICollector
from ..config.sim_config import CarrierConfig


class KPIPlotter:
    """KPI 绘图器"""

    def __init__(self, collector: KPICollector,
                 carrier_config: CarrierConfig):
        self.collector = collector
        self.carrier_config = carrier_config

    def plot_all(self, save_dir: str = None):
        """生成所有图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib 未安装，跳过绘图")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('L2 RRM 仿真 KPI', fontsize=14)

        s = self.collector.get_valid_range()
        slot_duration_s = self.carrier_config.slot_duration_s

        # 1. UE 吞吐量 CDF
        self._plot_ue_throughput_cdf(axes[0, 0], s, slot_duration_s)

        # 2. 小区吞吐量时间序列
        self._plot_cell_throughput_ts(axes[0, 1], s, slot_duration_s)

        # 3. BLER 时间序列
        self._plot_bler_ts(axes[0, 2], s)

        # 4. MCS 分布
        self._plot_mcs_distribution(axes[1, 0], s)

        # 5. SINR CDF
        self._plot_sinr_cdf(axes[1, 1], s)

        # 6. PRB 分配
        self._plot_prb_utilization(axes[1, 2], s)

        plt.tight_layout()

        if save_dir:
            from pathlib import Path
            path = Path(save_dir) / "kpi_report.png"
            plt.savefig(str(path), dpi=150, bbox_inches='tight')
            print(f"KPI 图表已保存: {path}")
        else:
            plt.savefig("kpi_report.png", dpi=150, bbox_inches='tight')
            print("KPI 图表已保存: kpi_report.png")

        plt.close()

    def _plot_ue_throughput_cdf(self, ax, s, slot_duration_s):
        """UE 平均吞吐量 CDF"""
        ue_bits = self.collector.ue_throughput_bits[s]
        ue_avg_mbps = np.mean(ue_bits, axis=0) / slot_duration_s / 1e6
        sorted_tp = np.sort(ue_avg_mbps)
        cdf = np.arange(1, len(sorted_tp) + 1) / len(sorted_tp)
        ax.plot(sorted_tp, cdf, 'b-o', markersize=4)
        ax.set_xlabel('UE Avg Throughput (Mbps)')
        ax.set_ylabel('CDF')
        ax.set_title('UE Throughput CDF')
        ax.grid(True, alpha=0.3)

    def _plot_cell_throughput_ts(self, ax, s, slot_duration_s):
        """小区吞吐量时间序列 (滑动平均)"""
        cell_mbps = self.collector.cell_throughput_bits[s] / slot_duration_s / 1e6
        # 100-slot 滑动平均
        window = min(100, len(cell_mbps))
        if window > 0:
            smoothed = np.convolve(cell_mbps, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, 'b-', linewidth=0.5)
        ax.set_xlabel('Slot')
        ax.set_ylabel('Cell Throughput (Mbps)')
        ax.set_title('Cell Throughput (100-slot MA)')
        ax.grid(True, alpha=0.3)

    def _plot_bler_ts(self, ax, s):
        """BLER 时间序列 (滑动平均)"""
        # 所有 UE 平均的 TB 成功率
        success = self.collector.ue_tb_success[s].astype(float)
        scheduled = self.collector.ue_num_prbs[s] > 0
        # 每 slot 的平均 BLER (只算被调度的)
        slot_bler = np.zeros(success.shape[0])
        for t in range(success.shape[0]):
            sched_ue = scheduled[t]
            if np.any(sched_ue):
                slot_bler[t] = 1.0 - np.mean(success[t, sched_ue])

        window = min(200, len(slot_bler))
        if window > 0:
            smoothed = np.convolve(slot_bler, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, 'r-', linewidth=0.5)
        ax.axhline(y=0.1, color='k', linestyle='--', linewidth=1, label='Target 10%')
        ax.set_xlabel('Slot')
        ax.set_ylabel('BLER')
        ax.set_title('BLER (200-slot MA)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_mcs_distribution(self, ax, s):
        """MCS 分布直方图"""
        scheduled = self.collector.ue_num_prbs[s] > 0
        mcs_data = self.collector.ue_mcs[s][scheduled]
        if len(mcs_data) > 0:
            ax.hist(mcs_data, bins=range(30), alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('MCS Index')
        ax.set_ylabel('Count')
        ax.set_title('MCS Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_sinr_cdf(self, ax, s):
        """有效 SINR CDF"""
        sinr = self.collector.ue_sinr_eff_db[s].ravel()
        sinr = sinr[sinr > -29]  # 排除未调度
        if len(sinr) > 0:
            sorted_sinr = np.sort(sinr)
            cdf = np.arange(1, len(sorted_sinr) + 1) / len(sorted_sinr)
            ax.plot(sorted_sinr, cdf, 'b-', linewidth=0.5)
        ax.set_xlabel('Effective SINR (dB)')
        ax.set_ylabel('CDF')
        ax.set_title('Effective SINR CDF')
        ax.grid(True, alpha=0.3)

    def _plot_prb_utilization(self, ax, s):
        """PRB 利用率"""
        total_prbs = self.collector.ue_num_prbs[s]
        prb_per_slot = np.sum(total_prbs, axis=1)
        max_prb = self.carrier_config.num_prb
        utilization = prb_per_slot / max_prb * 100.0
        window = min(100, len(utilization))
        if window > 0:
            smoothed = np.convolve(utilization, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, 'g-', linewidth=0.5)
        ax.set_xlabel('Slot')
        ax.set_ylabel('PRB Utilization (%)')
        ax.set_title('PRB Utilization (100-slot MA)')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)

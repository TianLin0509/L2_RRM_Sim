"""L2 RRM 系统级仿真平台 — Streamlit Web 界面

启动: .venv312/Scripts/streamlit.exe run app.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig,
    ChannelConfig, CSIConfig,
)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="L2 RRM 仿真平台",
    page_icon="📡",
    layout="wide",
)

st.title("📡 L2 RRM 无线系统级仿真平台")
st.caption("3GPP NR | Sionna 信道 | PF 调度 | HARQ | CSI 反馈 | 体验速率")

# 带宽 → PRB 映射 (SCS=30kHz)
BW_PRB_MAP = {
    "20 MHz": {15: 106, 30: 51, 60: 24},
    "50 MHz": {15: 270, 30: 133, 60: 65},
    "100 MHz": {15: 0, 30: 273, 60: 135},
}

# ============================================================
# 侧边栏参数
# ============================================================
with st.sidebar:
    st.header("仿真参数")

    with st.expander("仿真控制", expanded=True):
        num_slots = st.number_input("Slots 数", 100, 50000, 2000, step=500)
        random_seed = st.number_input("随机种子", 0, 99999, 42)
        warmup_slots = st.number_input("Warmup Slots", 0, 5000, 200, step=100)

    with st.expander("载波配置", expanded=True):
        scs = st.selectbox("子载波间隔 (kHz)", [15, 30, 60], index=1)
        bw_label = st.selectbox("带宽", ["20 MHz", "50 MHz", "100 MHz"], index=2)
        num_prb = BW_PRB_MAP[bw_label].get(scs, 273)
        if num_prb == 0:
            st.warning(f"{bw_label} @ {scs}kHz 不支持")
            num_prb = 273
        st.info(f"PRB 数: **{num_prb}**")
        carrier_freq = st.number_input("载频 (GHz)", 0.5, 100.0, 3.5, step=0.5)

    with st.expander("小区配置"):
        scenario = st.selectbox("3GPP 场景", ["uma", "umi", "rma"], index=0)
        num_tx_ant = st.selectbox("BS 天线数", [8, 16, 32, 64], index=2)
        num_tx_ports = st.selectbox("CSI-RS 端口", [2, 4, 8], index=1)
        max_layers = st.selectbox("最大层数", [1, 2, 4], index=1)
        total_power = st.slider("总发射功率 (dBm)", 30.0, 56.0, 46.0, step=1.0)
        cell_radius = st.slider("小区半径 (m)", 100, 2000, 500, step=50)
        bs_height = st.number_input("基站高度 (m)", 10.0, 50.0, 25.0)

    with st.expander("UE 配置"):
        num_ue = st.slider("UE 数量", 1, 50, 10)
        num_rx_ant = st.selectbox("UE 天线数", [1, 2, 4], index=1)
        ue_speed = st.slider("UE 速度 (km/h)", 0.0, 120.0, 3.0, step=1.0)
        min_dist = st.number_input("最小距离 (m)", 10.0, 200.0, 35.0)
        max_dist = st.number_input("最大距离 (m)", 100.0, 2000.0, float(cell_radius))

    with st.expander("调度与链路自适应"):
        sched_type = st.selectbox("调度算法", ["PF (Proportional Fair)"])
        pf_beta = st.slider("PF 遗忘因子 β", 0.9, 0.999, 0.98, step=0.005, format="%.3f")
        bler_target = st.slider("目标 BLER", 0.01, 0.3, 0.1, step=0.01)
        olla_delta = st.slider("OLLA Δ_up (dB)", 0.1, 2.0, 0.5, step=0.1)

    with st.expander("流量模型"):
        traffic_type = st.selectbox("流量类型", ["Full Buffer", "FTP Model 3", "Poisson"])
        ftp_file_size = 512000
        ftp_rate = 2.0
        poisson_pps = 200
        if traffic_type == "FTP Model 3":
            ftp_file_size = st.number_input("文件大小 (KB)", 50, 5000, 500) * 1000
            ftp_rate = st.number_input("到达率 (files/s/UE)", 0.1, 50.0, 2.0, step=0.5)
        elif traffic_type == "Poisson":
            poisson_pps = st.number_input("包到达率 (pkt/s/UE)", 10, 5000, 200)

    with st.expander("信道模型"):
        channel_type = st.selectbox("信道类型", ["statistical", "sionna"], index=0)
        if channel_type == "sionna":
            st.info("Sionna 3GPP 信道 (GPU 加速，较慢)")

    with st.expander("CSI 反馈"):
        csi_enabled = st.checkbox("启用 CSI 反馈", value=False)
        csi_period = st.number_input("CSI 周期 (slots)", 5, 100, 10)
        csi_delay = st.number_input("反馈延迟 (slots)", 0, 20, 4)


# ============================================================
# 构建配置
# ============================================================
def build_config():
    traffic_type_map = {
        "Full Buffer": "full_buffer",
        "FTP Model 3": "ftp_model3",
        "Poisson": "bursty",
    }
    return {
        'sim': SimConfig(num_slots=num_slots, random_seed=random_seed,
                         warmup_slots=warmup_slots),
        'carrier': CarrierConfig(subcarrier_spacing=scs, num_prb=num_prb,
                                  bandwidth_mhz=float(bw_label.split()[0]),
                                  carrier_freq_ghz=carrier_freq),
        'cell': CellConfig(num_tx_ant=num_tx_ant, num_tx_ports=num_tx_ports,
                            max_layers=max_layers, total_power_dbm=total_power,
                            cell_radius_m=float(cell_radius), height_m=bs_height,
                            scenario=scenario),
        'ue': UEConfig(num_ue=num_ue, num_rx_ant=num_rx_ant, speed_kmh=ue_speed,
                        min_distance_m=min_dist, max_distance_m=max_dist),
        'scheduler': SchedulerConfig(type='pf', beta=pf_beta),
        'link_adaptation': LinkAdaptationConfig(bler_target=bler_target,
                                                  olla_delta_up=olla_delta),
        'traffic': TrafficConfig(type=traffic_type_map.get(traffic_type, "full_buffer"),
                                  ftp_file_size_bytes=ftp_file_size, ftp_lambda=ftp_rate),
        'channel': ChannelConfig(type=channel_type, scenario=scenario),
        'csi': CSIConfig(enabled=csi_enabled, csi_period_slots=csi_period,
                          feedback_delay_slots=csi_delay),
    }


# ============================================================
# 仿真执行
# ============================================================
def run_simulation(config):
    """运行仿真，带进度条"""
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.kpi.kpi_reporter import KPIReporter

    engine = SimulationEngine(config)

    # 替换流量模型
    if traffic_type == "FTP Model 3":
        from l2_rrm_sim.traffic.ftp_model import FTPModel3
        engine.traffic = FTPModel3(
            file_size_bytes=ftp_file_size, arrival_rate=ftp_rate,
            slot_duration_s=engine.carrier_config.slot_duration_s,
            num_ue=engine.num_ue, rng=engine.rng.traffic,
        )
    elif traffic_type == "Poisson":
        from l2_rrm_sim.traffic.bursty_traffic import PoissonTraffic
        engine.traffic = PoissonTraffic(
            packet_size_bytes=1500, arrival_rate_pps=poisson_pps,
            slot_duration_s=engine.carrier_config.slot_duration_s,
            num_ue=engine.num_ue, rng=engine.rng.traffic,
        )

    # 运行带进度
    total = config['sim'].num_slots
    progress_bar = st.progress(0, text="仿真中...")
    status_text = st.empty()
    t0 = time.time()

    for slot_idx in range(total):
        slot_result = engine.run_slot(slot_idx)
        buf = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(slot_idx, slot_result, buf)

        if (slot_idx + 1) % max(total // 100, 1) == 0 or slot_idx == total - 1:
            pct = (slot_idx + 1) / total
            elapsed = time.time() - t0
            speed = (slot_idx + 1) / max(elapsed, 0.01)
            progress_bar.progress(pct, text=f"Slot {slot_idx+1}/{total} ({speed:.0f} slots/s)")

    elapsed = time.time() - t0
    progress_bar.progress(1.0, text=f"完成! ({elapsed:.1f}s)")

    # 生成报告
    reporter = KPIReporter(engine.kpi, engine.carrier_config)
    report = reporter.report()

    # HARQ 统计
    harq_stats = engine.harq_mgr.get_all_stats()

    return report, engine.kpi, engine.carrier_config, harq_stats


# ============================================================
# 图表生成
# ============================================================
def plot_throughput_cdf(kpi, carrier):
    s = kpi.get_valid_range()
    sd = carrier.slot_duration_s
    ue_avg = np.mean(kpi.ue_throughput_bits[s], axis=0) / sd / 1e6
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_tp = np.sort(ue_avg)
    cdf = np.arange(1, len(sorted_tp) + 1) / len(sorted_tp)
    ax.plot(sorted_tp, cdf, 'b-o', markersize=5, linewidth=2)
    ax.set_xlabel('UE Avg Throughput (Mbps)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('UE Throughput CDF', fontsize=13)
    ax.grid(True, alpha=0.3)
    return fig


def plot_bler_ts(kpi):
    s = kpi.get_valid_range()
    success = kpi.ue_tb_success[s].astype(float)
    sched = kpi.ue_num_prbs[s] > 0
    slot_bler = np.zeros(success.shape[0])
    for t in range(success.shape[0]):
        m = sched[t]
        if np.any(m):
            slot_bler[t] = 1.0 - np.mean(success[t, m])
    fig, ax = plt.subplots(figsize=(6, 4))
    w = min(200, len(slot_bler))
    if w > 0:
        smoothed = np.convolve(slot_bler, np.ones(w)/w, mode='valid')
        ax.plot(smoothed, 'r-', linewidth=1.5)
    ax.axhline(y=0.1, color='k', linestyle='--', linewidth=1, label='Target 10%')
    ax.set_xlabel('Slot', fontsize=11)
    ax.set_ylabel('BLER', fontsize=11)
    ax.set_title('BLER (200-slot MA)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_mcs_dist(kpi):
    s = kpi.get_valid_range()
    sched = kpi.ue_num_prbs[s] > 0
    mcs = kpi.ue_mcs[s][sched]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(mcs) > 0:
        ax.hist(mcs, bins=range(30), alpha=0.7, color='#2ecc71', edgecolor='white')
    ax.set_xlabel('MCS Index', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('MCS Distribution', fontsize=13)
    ax.grid(True, alpha=0.3)
    return fig


def plot_sinr_cdf(kpi):
    s = kpi.get_valid_range()
    sinr = kpi.ue_sinr_eff_db[s].ravel()
    sinr = sinr[sinr > -29]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(sinr) > 0:
        sorted_s = np.sort(sinr)
        cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
        ax.plot(sorted_s, cdf, 'b-', linewidth=1.5)
    ax.set_xlabel('Effective SINR (dB)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('Effective SINR CDF', fontsize=13)
    ax.grid(True, alpha=0.3)
    return fig


def plot_cell_tp_ts(kpi, carrier):
    s = kpi.get_valid_range()
    cell_mbps = kpi.cell_throughput_bits[s] / carrier.slot_duration_s / 1e6
    fig, ax = plt.subplots(figsize=(6, 4))
    w = min(100, len(cell_mbps))
    if w > 0:
        smoothed = np.convolve(cell_mbps, np.ones(w)/w, mode='valid')
        ax.plot(smoothed, color='#3498db', linewidth=1.5)
    ax.set_xlabel('Slot', fontsize=11)
    ax.set_ylabel('Cell Throughput (Mbps)', fontsize=11)
    ax.set_title('Cell Throughput (100-slot MA)', fontsize=13)
    ax.grid(True, alpha=0.3)
    return fig


# ============================================================
# 主界面
# ============================================================
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_btn = st.button("▶ 开始仿真", type="primary", use_container_width=True)
with col_info:
    st.markdown(f"**配置**: {num_ue} UE, {bw_label}@{scs}kHz, {scenario.upper()}, "
                f"{channel_type}, {traffic_type}, {num_slots} slots")

if run_btn:
    config = build_config()
    report, kpi_data, carrier_cfg, harq_stats = run_simulation(config)
    st.session_state['report'] = report
    st.session_state['kpi'] = kpi_data
    st.session_state['carrier'] = carrier_cfg
    st.session_state['harq'] = harq_stats

# ============================================================
# 展示结果
# ============================================================
if 'report' in st.session_state:
    report = st.session_state['report']
    kpi_data = st.session_state['kpi']
    carrier_cfg = st.session_state['carrier']
    harq = st.session_state['harq']

    st.divider()

    # KPI 概览
    st.subheader("KPI 概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("小区吞吐量", f"{report['cell_avg_throughput_mbps']:.1f} Mbps")
    c2.metric("频谱效率", f"{report['spectral_efficiency_bps_hz']:.2f} bps/Hz")
    c3.metric("平均 BLER", f"{report['avg_bler']:.3f}",
              delta=f"目标 {bler_target}")
    c4.metric("Jain 公平指数", f"{report['jain_fairness']:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("平均 MCS", f"{report['avg_mcs']:.1f}")
    c6.metric("平均 SINR", f"{report['avg_sinr_db']:.1f} dB")
    c7.metric("调度占比", f"{report['avg_scheduling_ratio']*100:.1f}%")
    c8.metric("PRB 利用率", f"{report['prb_utilization']*100:.1f}%")

    # 体验速率
    st.subheader("体验速率")
    exp = report.get('experienced_rate_detail', {})
    n_sess = exp.get('total_valid_sessions', 0)
    e1, e2, e3 = st.columns(3)
    e1.metric("小区体验速率", f"{report['cell_experienced_rate_mbps']:.2f} Mbps")
    e2.metric("边缘体验速率", f"{report.get('cell_edge_experienced_rate_mbps', 0):.2f} Mbps")
    e3.metric("有效 Sessions", f"{n_sess}")

    # HARQ
    st.subheader("HARQ 统计")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("总传输", f"{harq['total_transmissions']}")
    h2.metric("重传次数", f"{harq['total_retransmissions']}")
    h3.metric("重传率", f"{harq['retx_rate']:.1%}")
    h4.metric("有效 BLER", f"{harq['effective_bler']:.4f}")

    st.divider()

    # 图表
    st.subheader("分析图表")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["吞吐量 CDF", "BLER 趋势", "MCS 分布", "SINR CDF", "小区吞吐趋势"]
    )
    with tab1:
        st.pyplot(plot_throughput_cdf(kpi_data, carrier_cfg))
    with tab2:
        st.pyplot(plot_bler_ts(kpi_data))
    with tab3:
        st.pyplot(plot_mcs_dist(kpi_data))
    with tab4:
        st.pyplot(plot_sinr_cdf(kpi_data))
    with tab5:
        st.pyplot(plot_cell_tp_ts(kpi_data, carrier_cfg))

    st.divider()

    # UE 详情表
    st.subheader("UE 详情")
    ue_tp = report['ue_avg_throughput_mbps']
    ue_bler = report['bler_per_ue']
    ue_sched = report['ue_scheduling_ratio']
    ue_exp = report.get('ue_experienced_rate_mbps', np.zeros(len(ue_tp)))

    import pandas as pd
    df = pd.DataFrame({
        'UE': range(len(ue_tp)),
        'Avg TP (Mbps)': np.round(ue_tp, 2),
        'Exp Rate (Mbps)': np.round(ue_exp, 2),
        'BLER': np.round(ue_bler, 4),
        'Sched %': np.round(ue_sched * 100, 1),
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

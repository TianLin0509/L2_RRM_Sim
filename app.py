"""Streamlit UI for the L2 RRM single-cell simulator.

Run with:
    .venv312/Scripts/streamlit.exe run app.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

from l2_rrm_sim.config.sim_config import (  # noqa: E402
    CSIConfig,
    CarrierConfig,
    CellConfig,
    ChannelConfig,
    LinkAdaptationConfig,
    SchedulerConfig,
    SimConfig,
    TDDConfig,
    TrafficConfig,
    UEConfig,
)


BW_PRB_MAP = {
    "20 MHz": {15: 106, 30: 51, 60: 24},
    "50 MHz": {15: 270, 30: 133, 60: 65},
    "100 MHz": {15: 0, 30: 273, 60: 135},
}

TDD_PATTERNS = ["DDDSU", "DDSUU", "DDDSUDDSUU", "DDDDDDDSUU"]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe5;
            --panel: rgba(255, 250, 242, 0.9);
            --panel-strong: rgba(255, 248, 236, 0.98);
            --ink: #1d2b34;
            --muted: #5f6f78;
            --accent: #0f766e;
            --accent-2: #c2410c;
            --line: rgba(29, 43, 52, 0.12);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15,118,110,0.14), transparent 32%),
                radial-gradient(circle at top right, rgba(194,65,12,0.16), transparent 28%),
                linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
            color: var(--ink);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #183643 0%, #10262f 100%);
        }
        [data-testid="stSidebar"] * {
            color: #eef6f4;
        }
        .hero-card, .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: 0 16px 40px rgba(18, 34, 40, 0.08);
        }
        .hero-card {
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(135deg, rgba(15,118,110,0.12), rgba(255,250,242,0.95)),
                var(--panel-strong);
        }
        .section-card {
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
        }
        .pill-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.7rem;
        }
        .pill {
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            background: rgba(15,118,110,0.12);
            border: 1px solid rgba(15,118,110,0.18);
            color: var(--ink);
            font-size: 0.92rem;
        }
        .label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
        }
        .value {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
        }
        div[data-testid="metric-container"] {
            background: rgba(255, 250, 242, 0.88);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 12px 28px rgba(18, 34, 40, 0.06);
        }
        div[data-testid="metric-container"] label {
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "preset_name": "单小区 TDD 基线",
        "num_slots": 2000,
        "random_seed": 42,
        "warmup_slots": 200,
        "scs": 30,
        "bw_label": "100 MHz",
        "carrier_freq": 3.5,
        "duplex_mode": "TDD",
        "tdd_pattern": "DDDSU",
        "special_dl_symbols": 10,
        "special_gp_symbols": 2,
        "special_ul_symbols": 2,
        "harq_k1": 4,
        "scenario": "uma",
        "num_tx_ant": 32,
        "num_tx_ports": 4,
        "max_layers": 1,
        "total_power": 46.0,
        "cell_radius": 500,
        "bs_height": 25.0,
        "num_ue": 1,
        "num_rx_ant": 2,
        "ue_speed": 3.0,
        "min_dist": 35.0,
        "max_dist": 35.0,
        "sched_type": "PF (Proportional Fair)",
        "pf_beta": 0.98,
        "bler_target": 0.1,
        "olla_delta": 0.5,
        "traffic_type": "Full Buffer",
        "ftp_file_size": 512000,
        "ftp_rate": 2.0,
        "poisson_pps": 200,
        "channel_type": "statistical",
        "csi_enabled": False,
        "csi_period": 10,
        "csi_delay": 4,
        "subband_size_prb": 4,
        "trace_rows": 40,
        "trace_focus_ue": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(name: str) -> None:
    presets = {
        "单小区 TDD 基线": {
            "preset_name": name,
            "duplex_mode": "TDD",
            "tdd_pattern": "DDDSU",
            "special_dl_symbols": 10,
            "special_gp_symbols": 2,
            "special_ul_symbols": 2,
            "harq_k1": 4,
            "num_ue": 1,
            "max_layers": 1,
            "num_rx_ant": 2,
            "traffic_type": "Full Buffer",
            "channel_type": "statistical",
            "csi_enabled": False,
            "num_slots": 2000,
            "warmup_slots": 200,
            "min_dist": 35.0,
            "max_dist": 35.0,
        },
        "单小区 FDD Rank-1": {
            "preset_name": name,
            "duplex_mode": "FDD",
            "num_ue": 1,
            "max_layers": 1,
            "num_rx_ant": 2,
            "traffic_type": "Full Buffer",
            "channel_type": "statistical",
            "csi_enabled": False,
            "num_slots": 2000,
            "warmup_slots": 200,
            "min_dist": 35.0,
            "max_dist": 35.0,
        },
        "多用户 PF 观察": {
            "preset_name": name,
            "duplex_mode": "TDD",
            "tdd_pattern": "DDDSU",
            "num_ue": 10,
            "max_layers": 2,
            "num_rx_ant": 2,
            "traffic_type": "Full Buffer",
            "channel_type": "statistical",
            "csi_enabled": True,
            "num_slots": 3000,
            "warmup_slots": 300,
            "min_dist": 35.0,
            "max_dist": 500.0,
        },
    }
    for key, value in presets[name].items():
        st.session_state[key] = value


def build_config() -> dict:
    traffic_type_map = {
        "Full Buffer": "full_buffer",
        "FTP Model 3": "ftp_model3",
        "Poisson": "bursty",
    }
    num_prb = BW_PRB_MAP[st.session_state.bw_label].get(st.session_state.scs, 273)
    if num_prb == 0:
        num_prb = 273
    return {
        "sim": SimConfig(
            num_slots=st.session_state.num_slots,
            random_seed=st.session_state.random_seed,
            warmup_slots=st.session_state.warmup_slots,
        ),
        "carrier": CarrierConfig(
            subcarrier_spacing=st.session_state.scs,
            num_prb=num_prb,
            bandwidth_mhz=float(st.session_state.bw_label.split()[0]),
            carrier_freq_ghz=st.session_state.carrier_freq,
        ),
        "cell": CellConfig(
            num_tx_ant=st.session_state.num_tx_ant,
            num_tx_ports=st.session_state.num_tx_ports,
            max_layers=st.session_state.max_layers,
            total_power_dbm=st.session_state.total_power,
            cell_radius_m=float(st.session_state.cell_radius),
            height_m=st.session_state.bs_height,
            scenario=st.session_state.scenario,
        ),
        "ue": UEConfig(
            num_ue=st.session_state.num_ue,
            num_rx_ant=st.session_state.num_rx_ant,
            speed_kmh=st.session_state.ue_speed,
            min_distance_m=st.session_state.min_dist,
            max_distance_m=st.session_state.max_dist,
        ),
        "scheduler": SchedulerConfig(type="pf", beta=st.session_state.pf_beta),
        "link_adaptation": LinkAdaptationConfig(
            bler_target=st.session_state.bler_target,
            olla_delta_up=st.session_state.olla_delta,
        ),
        "traffic": TrafficConfig(
            type=traffic_type_map.get(st.session_state.traffic_type, "full_buffer"),
            ftp_file_size_bytes=int(st.session_state.ftp_file_size),
            ftp_lambda=st.session_state.ftp_rate,
        ),
        "channel": ChannelConfig(
            type=st.session_state.channel_type,
            scenario=st.session_state.scenario,
        ),
        "csi": CSIConfig(
            enabled=st.session_state.csi_enabled,
            csi_period_slots=st.session_state.csi_period,
            feedback_delay_slots=st.session_state.csi_delay,
            subband_size_prb=st.session_state.subband_size_prb,
        ),
        "tdd": TDDConfig(
            duplex_mode=st.session_state.duplex_mode,
            pattern=st.session_state.tdd_pattern,
            special_dl_symbols=st.session_state.special_dl_symbols,
            special_gp_symbols=st.session_state.special_gp_symbols,
            special_ul_symbols=st.session_state.special_ul_symbols,
            harq_k1=st.session_state.harq_k1,
        ),
    }


def run_simulation(config: dict):
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.kpi.kpi_reporter import KPIReporter

    engine = SimulationEngine(config)

    if st.session_state.traffic_type == "FTP Model 3":
        from l2_rrm_sim.traffic.ftp_model import FTPModel3

        engine.traffic = FTPModel3(
            file_size_bytes=st.session_state.ftp_file_size,
            arrival_rate=st.session_state.ftp_rate,
            slot_duration_s=engine.carrier_config.slot_duration_s,
            num_ue=engine.num_ue,
            rng=engine.rng.traffic,
        )
    elif st.session_state.traffic_type == "Poisson":
        from l2_rrm_sim.traffic.bursty_traffic import PoissonTraffic

        engine.traffic = PoissonTraffic(
            packet_size_bytes=1500,
            arrival_rate_pps=st.session_state.poisson_pps,
            slot_duration_s=engine.carrier_config.slot_duration_s,
            num_ue=engine.num_ue,
            rng=engine.rng.traffic,
        )

    ftp_traffic = None
    if st.session_state.traffic_type == "FTP Model 3" and hasattr(engine.traffic, "dequeue_bytes"):
        ftp_traffic = engine.traffic

    total = config["sim"].num_slots
    progress_bar = st.progress(0.0, text="Running simulation...")
    started = time.time()

    for slot_idx in range(total):
        slot_result = engine.run_slot(slot_idx)
        buf_before = engine._buf_after_traffic.copy()

        if ftp_traffic is not None:
            for ue_idx in range(engine.num_ue):
                decoded_bytes = int(slot_result.ue_decoded_bits[ue_idx]) // 8
                if decoded_bytes > 0:
                    ftp_traffic.dequeue_bytes(ue_idx, decoded_bytes, slot_idx)

        buf_after = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(slot_idx, slot_result, buf_before, buf_after)

        if (slot_idx + 1) % max(total // 100, 1) == 0 or slot_idx == total - 1:
            pct = (slot_idx + 1) / total
            elapsed = time.time() - started
            speed = (slot_idx + 1) / max(elapsed, 0.01)
            progress_bar.progress(pct, text=f"Slot {slot_idx + 1}/{total} | {speed:.0f} slots/s")

    reporter = KPIReporter(engine.kpi, engine.carrier_config)
    report = reporter.report()
    harq_stats = engine.harq_mgr.get_all_stats()
    progress_bar.progress(1.0, text=f"Completed in {time.time() - started:.1f}s")
    return report, engine.kpi, engine.carrier_config, harq_stats


def plot_throughput_cdf(kpi, carrier):
    s = kpi.get_valid_range()
    ue_avg = np.mean(kpi.ue_throughput_bits[s], axis=0) / carrier.slot_duration_s / 1e6
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_tp = np.sort(ue_avg)
    cdf = np.arange(1, len(sorted_tp) + 1) / max(len(sorted_tp), 1)
    ax.plot(sorted_tp, cdf, color="#0f766e", linewidth=2)
    ax.set_xlabel("UE Avg Throughput (Mbps)")
    ax.set_ylabel("CDF")
    ax.set_title("UE Throughput CDF")
    ax.grid(True, alpha=0.25)
    return fig


def plot_bler_ts(kpi):
    s = kpi.get_valid_range()
    success = kpi.ue_tb_success[s].astype(float)
    sched = kpi.ue_num_prbs[s] > 0
    slot_bler = np.zeros(success.shape[0])
    for idx in range(success.shape[0]):
        mask = sched[idx]
        if np.any(mask):
            slot_bler[idx] = 1.0 - np.mean(success[idx, mask])
    fig, ax = plt.subplots(figsize=(6, 4))
    window = min(200, len(slot_bler))
    if window > 0:
        smoothed = np.convolve(slot_bler, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, color="#c2410c", linewidth=1.7)
    ax.axhline(y=0.1, color="#1d2b34", linestyle="--", linewidth=1, label="Target 10%")
    ax.set_xlabel("Slot")
    ax.set_ylabel("BLER")
    ax.set_title("BLER Trend")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig


def plot_mcs_dist(kpi):
    s = kpi.get_valid_range()
    sched = kpi.ue_num_prbs[s] > 0
    mcs = kpi.ue_mcs[s][sched]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(mcs) > 0:
        ax.hist(mcs, bins=range(30), color="#0f766e", alpha=0.85, edgecolor="#f8f4ec")
    ax.set_xlabel("MCS Index")
    ax.set_ylabel("Count")
    ax.set_title("MCS Distribution")
    ax.grid(True, alpha=0.25)
    return fig


def plot_sinr_cdf(kpi):
    s = kpi.get_valid_range()
    sinr = kpi.ue_sinr_eff_db[s].ravel()
    sinr = sinr[sinr > -29]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(sinr) > 0:
        sorted_s = np.sort(sinr)
        cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
        ax.plot(sorted_s, cdf, color="#c2410c", linewidth=1.7)
    ax.set_xlabel("Effective SINR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("Effective SINR CDF")
    ax.grid(True, alpha=0.25)
    return fig


def plot_cell_tp_ts(kpi, carrier):
    s = kpi.get_valid_range()
    delivered = kpi.cell_throughput_bits[s] / carrier.slot_duration_s / 1e6
    scheduled = kpi.cell_scheduled_bits[s] / carrier.slot_duration_s / 1e6
    
    # 使用 Pandas 构造数据，方便 Streamlit 原生绘图
    df_tp = pd.DataFrame({
        "Scheduled (Mbps)": scheduled,
        "Delivered (Mbps)": delivered
    })
    # 平滑处理
    window = min(50, len(df_tp))
    if window > 1:
        df_tp_smooth = df_tp.rolling(window=window).mean().dropna()
        return df_tp_smooth
    return df_tp


def render_summary(config: dict, report: dict, harq: dict) -> None:
    carrier = config["carrier"]
    cell = config["cell"]
    ue = config["ue"]
    tdd = config["tdd"]

    if tdd.duplex_mode == "FDD":
        duplex_text = "FDD"
    else:
        duplex_text = f"TDD {tdd.pattern} ({tdd.special_dl_symbols}+{tdd.special_gp_symbols}+{tdd.special_ul_symbols})"

    # 计算平均 Rank
    avg_rank = np.mean(report.get("ue_rank", [1.0]))

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="label">Single-Cell Validation Console</div>
            <div class="value">L2 RRM Simulator</div>
            <div class="small-note">
                围绕单小区场景进行深度协议审计。支持 TDD/FDD 帧结构、HARQ 闭环及 CSI 链路自适应。
            </div>
            <div class="pill-row">
                <div class="pill">{duplex_text}</div>
                <div class="pill">{ue.num_ue} UE</div>
                <div class="pill">{carrier.bandwidth_mhz:.0f} MHz @ {carrier.subcarrier_spacing} kHz</div>
                <div class="pill">Avg Rank: {avg_rank:.2f}</div>
                <div class="pill">Delivery {report['delivery_ratio']*100:.1f}%</div>
                <div class="pill">HARQ retx {harq['retx_rate']*100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def slot_trace_frame(report: dict, focus_ue: int, rows: int) -> pd.DataFrame:
    trace = report["slot_trace"]
    num_ue = trace["ue_num_prbs"].shape[1] if trace["ue_num_prbs"].ndim == 2 else 0
    focus_ue = min(max(focus_ue, 0), max(num_ue - 1, 0))
    frame = pd.DataFrame(
        {
            "slot": trace["slot_idx"],
            "dir": trace["slot_direction"],
            "dl_symbols": trace["num_dl_symbols"],
            "cell_sched_bits": trace["cell_scheduled_bits"],
            "cell_deliv_bits": trace["cell_delivered_bits"],
            "ue_prbs": trace["ue_num_prbs"][:, focus_ue] if num_ue else [],
            "ue_re": trace["ue_num_re"][:, focus_ue] if num_ue else [],
            "ue_sched_bits": trace["ue_scheduled_bits"][:, focus_ue] if num_ue else [],
            "ue_deliv_bits": trace["ue_delivered_bits"][:, focus_ue] if num_ue else [],
            "ue_mcs": trace["ue_mcs"][:, focus_ue] if num_ue else [],
            "ue_rank": trace["ue_rank"][:, focus_ue] if num_ue else [],
            "ue_sinr_db": np.round(trace["ue_sinr_eff_db"][:, focus_ue], 2) if num_ue else [],
            "tb_success": trace["ue_tb_success"][:, focus_ue] if num_ue else [],
        }
    )
    return frame.tail(rows).reset_index(drop=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 场景控制")
        st.caption("当前网页入口运行的是单小区仿真主引擎。")

        with st.expander("仿真规模", expanded=True):
            st.number_input("Slots", min_value=100, max_value=50000, step=100, key="num_slots")
            st.number_input("Random Seed", min_value=0, max_value=99999, key="random_seed")
            st.number_input("Warmup Slots", min_value=0, max_value=5000, step=50, key="warmup_slots")

        with st.expander("链路与帧结构", expanded=True):
            st.selectbox("Duplex", ["TDD", "FDD"], key="duplex_mode")
            if st.session_state.duplex_mode == "TDD":
                st.selectbox("TDD Pattern", TDD_PATTERNS, key="tdd_pattern")
                st.slider("Special DL Symbols", min_value=1, max_value=12, key="special_dl_symbols")
                st.slider("Special GP Symbols", min_value=0, max_value=6, key="special_gp_symbols")
                st.slider("Special UL Symbols", min_value=0, max_value=6, key="special_ul_symbols")
                st.slider("HARQ K1", min_value=1, max_value=10, key="harq_k1")
            st.selectbox("SCS (kHz)", [15, 30, 60], key="scs")
            st.selectbox("Bandwidth", ["20 MHz", "50 MHz", "100 MHz"], key="bw_label")
            st.number_input("Carrier Frequency (GHz)", min_value=0.5, max_value=100.0, step=0.1, key="carrier_freq")
            num_prb = BW_PRB_MAP[st.session_state.bw_label].get(st.session_state.scs, 273)
            if num_prb == 0:
                st.warning(f"{st.session_state.bw_label} @ {st.session_state.scs} kHz is not in the preset map. Use 273 PRB.")
                num_prb = 273
            st.info(f"Derived PRB: {num_prb}")

        with st.expander("小区与天线", expanded=True):
            st.selectbox("Scenario", ["uma", "umi", "rma"], key="scenario")
            st.selectbox("BS Tx Antennas", [8, 16, 32, 64], key="num_tx_ant")
            st.selectbox("CSI-RS Ports", [2, 4, 8], key="num_tx_ports")
            st.selectbox("Max Layers", [1, 2, 4], key="max_layers")
            st.slider("Total Power (dBm)", min_value=30.0, max_value=56.0, step=1.0, key="total_power")
            st.slider("Cell Radius (m)", min_value=100, max_value=2000, step=50, key="cell_radius")
            st.number_input("BS Height (m)", min_value=10.0, max_value=50.0, step=1.0, key="bs_height")

        with st.expander("UE 场景", expanded=True):
            st.slider("UE Count", min_value=1, max_value=50, key="num_ue")
            st.selectbox("UE Rx Antennas", [1, 2, 4], key="num_rx_ant")
            st.slider("UE Speed (km/h)", min_value=0.0, max_value=120.0, step=1.0, key="ue_speed")
            st.number_input("Min Distance (m)", min_value=10.0, max_value=2000.0, step=5.0, key="min_dist")
            st.number_input("Max Distance (m)", min_value=10.0, max_value=2000.0, step=5.0, key="max_dist")

        with st.expander("调度与自适应", expanded=True):
            st.selectbox("Scheduler", ["PF (Proportional Fair)"], key="sched_type")
            st.slider("PF Beta", min_value=0.9, max_value=0.999, step=0.001, format="%.3f", key="pf_beta")
            st.slider("BLER Target", min_value=0.01, max_value=0.30, step=0.01, key="bler_target")
            st.slider("OLLA Delta Up (dB)", min_value=0.1, max_value=2.0, step=0.1, key="olla_delta")

        with st.expander("业务与信道", expanded=True):
            st.selectbox("Traffic", ["Full Buffer", "FTP Model 3", "Poisson"], key="traffic_type")
            if st.session_state.traffic_type == "FTP Model 3":
                st.number_input("FTP File Size (bytes)", min_value=10000, max_value=5000000, step=10000, key="ftp_file_size")
                st.number_input("FTP Arrival Rate (files/s/UE)", min_value=0.1, max_value=50.0, step=0.1, key="ftp_rate")
            elif st.session_state.traffic_type == "Poisson":
                st.number_input("Poisson Packets/s/UE", min_value=10, max_value=5000, step=10, key="poisson_pps")
            st.selectbox("Channel", ["statistical", "sionna"], key="channel_type")
            st.checkbox("Enable CSI", key="csi_enabled")
            if st.session_state.csi_enabled:
                st.number_input("CSI Period (slots)", min_value=5, max_value=100, key="csi_period")
                st.number_input("Feedback Delay (slots)", min_value=0, max_value=20, key="csi_delay")
                st.number_input("Subband Size (PRB)", min_value=1, max_value=32, key="subband_size_prb")

        with st.expander("审计视图", expanded=False):
            st.slider("Slot Trace Rows", min_value=10, max_value=200, step=10, key="trace_rows")
            st.number_input("Focus UE", min_value=0, max_value=max(st.session_state.num_ue - 1, 0), step=1, key="trace_focus_ue")


def main() -> None:
    st.set_page_config(page_title="L2 RRM Simulator", page_icon="📶", layout="wide")
    inject_css()
    init_state()

    st.title("L2 RRM 单小区仿真控制台")
    st.caption("围绕单小区场景做配置、运行、审计和结果解释。")

    preset_cols = st.columns([1.1, 1.1, 1.1, 2.7])
    preset_labels = ["单小区 TDD 基线", "单小区 FDD Rank-1", "多用户 PF 观察"]
    for idx, label in enumerate(preset_labels):
        if preset_cols[idx].button(label, use_container_width=True):
            apply_preset(label)
            st.rerun()
    preset_cols[3].markdown(
        f"""
        <div class="section-card">
            <div class="label">Active Preset</div>
            <div class="value" style="font-size:1.2rem;">{st.session_state.preset_name}</div>
            <div class="small-note">预设会改写当前参数，适合快速切到单小区验证基线。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    config = build_config()
    if st.session_state.min_dist > st.session_state.max_dist:
        st.error("Min Distance 不能大于 Max Distance。")
        return
    if st.session_state.duplex_mode == "TDD":
        if (
            st.session_state.special_dl_symbols
            + st.session_state.special_gp_symbols
            + st.session_state.special_ul_symbols
            > 14
        ):
            st.error("Special slot 的 DL/GP/UL symbols 之和不能超过 14。")
            return

    summary_cols = st.columns([1.2, 2.6, 1.2])
    with summary_cols[0]:
        st.markdown(
            """
            <div class="section-card">
                <div class="label">Scenario Scope</div>
                <div class="value" style="font-size:1.15rem;">Single-Cell</div>
                <div class="small-note">当前网页入口走的是 SimulationEngine，不是 multicell engine。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_cols[1]:
        tdd_text = (
            f"{st.session_state.duplex_mode} · {st.session_state.tdd_pattern}"
            if st.session_state.duplex_mode == "TDD"
            else "FDD"
        )
        st.markdown(
            f"""
            <div class="section-card">
                <div class="label">Run Summary</div>
                <div class="value" style="font-size:1.25rem;">{st.session_state.num_ue} UE · {st.session_state.bw_label} @ {st.session_state.scs} kHz</div>
                <div class="small-note">
                    {tdd_text} | {st.session_state.channel_type} channel | {st.session_state.traffic_type} traffic | {st.session_state.num_slots} slots
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_cols[2]:
        run_btn = st.button("运行仿真", type="primary", use_container_width=True)

    if run_btn:
        report, kpi_data, carrier_cfg, harq_stats = run_simulation(config)
        st.session_state["report"] = report
        st.session_state["kpi"] = kpi_data
        st.session_state["carrier"] = carrier_cfg
        st.session_state["harq"] = harq_stats
        st.session_state["last_config"] = config

    if "report" not in st.session_state:
        st.info("选择预设或调整参数后运行仿真。页面会展示单小区 KPI、HARQ 状态和 slot trace 审计结果。")
        return

    report = st.session_state["report"]
    kpi_data = st.session_state["kpi"]
    carrier_cfg = st.session_state["carrier"]
    harq = st.session_state["harq"]
    last_config = st.session_state.get("last_config", config)

    render_summary(last_config, report, harq)

    st.subheader("核心 KPI")
    kpi_row_1 = st.columns(5)
    kpi_row_1[0].metric("Delivered Throughput", f"{report['cell_avg_throughput_mbps']:.2f} Mbps")
    kpi_row_1[1].metric("Scheduled Throughput", f"{report['cell_avg_scheduled_throughput_mbps']:.2f} Mbps")
    kpi_row_1[2].metric("Delivery Ratio", f"{report['delivery_ratio']*100:.1f}%")
    kpi_row_1[3].metric("Avg BLER", f"{report['avg_bler']:.3f}", delta=f"target {st.session_state.bler_target:.2f}")
    kpi_row_1[4].metric("Avg SINR", f"{report['avg_sinr_db']:.1f} dB")

    kpi_row_2 = st.columns(5)
    kpi_row_2[0].metric("Avg MCS", f"{report['avg_mcs']:.1f}")
    kpi_row_2[1].metric("PRB Utilization", f"{report['prb_utilization']*100:.1f}%")
    kpi_row_2[2].metric("DL-slot PRB Util.", f"{report['dl_schedulable_prb_utilization']*100:.1f}%")
    kpi_row_2[3].metric("Spectral Efficiency", f"{report['spectral_efficiency_bps_hz']:.2f} bps/Hz")
    kpi_row_2[4].metric("Jain Fairness", f"{report['jain_fairness']:.3f}")

    st.subheader("HARQ 与时隙结构")
    hcols = st.columns(5)
    hcols[0].metric("Total TX", f"{harq['total_transmissions']}")
    hcols[1].metric("Total ReTX", f"{harq['total_retransmissions']}")
    hcols[2].metric("ReTX Rate", f"{harq['retx_rate']*100:.1f}%")
    hcols[3].metric("Effective BLER", f"{harq['effective_bler']:.4f}")
    if last_config["tdd"].duplex_mode == "TDD":
        hcols[4].metric("TDD Pattern", last_config["tdd"].pattern)
    else:
        hcols[4].metric("Duplex", "FDD")

    st.subheader("图表")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["吞吐 CDF", "BLER", "MCS 分布", "SINR CDF", "小区吞吐趋势"]
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
        st.line_chart(plot_cell_tp_ts(kpi_data, carrier_cfg))

    st.subheader("UE 视图")
    ue_df = pd.DataFrame(
        {
            "UE": range(len(report["ue_avg_throughput_mbps"])),
            "Avg TP (Mbps)": np.round(report["ue_avg_throughput_mbps"], 2),
            "Scheduled TP (Mbps)": np.round(report["ue_avg_scheduled_throughput_mbps"], 2),
            "Exp Rate (Mbps)": np.round(report.get("ue_experienced_rate_mbps", np.zeros(0)), 2),
            "BLER": np.round(report["bler_per_ue"], 4),
            "Sched %": np.round(report["ue_scheduling_ratio"] * 100, 1),
        }
    )
    st.dataframe(ue_df, use_container_width=True, hide_index=True)

    st.subheader("Slot Trace 审计")
    st.caption("把当前 slot 发包量和到账量拆开看，便于分析 TDD/K1/HARQ 时序。")
    trace_df = slot_trace_frame(
        report,
        focus_ue=st.session_state.trace_focus_ue,
        rows=st.session_state.trace_rows,
    )
    st.dataframe(trace_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

"""L2 RRM 5G NR System-Level Simulator — Streamlit UI

Run:  .venv312/Scripts/streamlit.exe run app.py
"""
from __future__ import annotations

import os, sys, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

from l2_rrm_sim.config.sim_config import (
    CSIConfig, CarrierConfig, CellConfig, ChannelConfig,
    LinkAdaptationConfig, SchedulerConfig, SimConfig,
    TDDConfig, TrafficConfig, UEConfig,
)

# ── Constants ────────────────────────────────────────────────────────
BW_PRB = {
    "20 MHz":  {15: 106, 30: 51,  60: 24},
    "50 MHz":  {15: 270, 30: 133, 60: 65},
    "100 MHz": {15: 0,   30: 273, 60: 135},
}
TDD_PATTERNS = ["DDDSU", "DDSUU", "DDDSUDDSUU", "DDDDDDDSUU"]
COLORS = dict(teal="#0d9488", orange="#ea580c", slate="#334155",
              bg="#f8fafc", card="#ffffff", border="#e2e8f0")

# ── CSS ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""<style>
    .stApp {{ background: {COLORS['bg']}; }}
    .main .block-container {{ max-width: 1400px; padding-top: 1.5rem; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }}
    [data-testid="stSidebar"] * {{ color: #e2e8f0; }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {{ color: #94a3b8 !important; }}
    div[data-testid="metric-container"] {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .kpi-hero {{
        background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
        color: white; border-radius: 16px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .kpi-hero .title {{ font-size: 1.6rem; font-weight: 700; }}
    .kpi-hero .subtitle {{ font-size: 0.95rem; opacity: 0.85; margin-top: 0.3rem; }}
    .kpi-hero .pills {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem; }}
    .kpi-hero .pill {{
        background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3);
        border-radius: 999px; padding: 0.25rem 0.7rem; font-size: 0.85rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0.5rem; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0; padding: 0.5rem 1.2rem;
    }}
    </style>""", unsafe_allow_html=True)


# ── State / Presets ──────────────────────────────────────────────────
def init_state():
    defaults = dict(
        preset="TDD Baseline", num_slots=2000, random_seed=42, warmup_slots=200,
        scs=30, bw="100 MHz", freq=3.5,
        duplex="TDD", tdd_pat="DDDSU", sp_dl=10, sp_gp=2, sp_ul=2, k1=4,
        scenario="uma", tx_ant=32, tx_ports=4, max_layers=1,
        power=46.0, radius=500, bs_h=25.0,
        n_ue=1, rx_ant=2, speed=3.0, d_min=35.0, d_max=35.0,
        beta=0.98, bler_t=0.1, olla_d=0.5,
        traffic="Full Buffer", ch_type="statistical",
        csi_on=False, csi_period=10, csi_delay=4, sb_prb=4,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

PRESETS = {
    "TDD Baseline": dict(
        duplex="TDD", tdd_pat="DDDSU", n_ue=1, max_layers=1,
        traffic="Full Buffer", d_min=35.0, d_max=35.0,
        num_slots=2000, warmup_slots=200,
    ),
    "FDD Rank-1": dict(
        duplex="FDD", n_ue=1, max_layers=1,
        traffic="Full Buffer", d_min=35.0, d_max=35.0,
        num_slots=2000, warmup_slots=200,
    ),
    "Multi-UE PF": dict(
        duplex="TDD", tdd_pat="DDDSU", n_ue=10, max_layers=2,
        traffic="Full Buffer", d_min=35.0, d_max=500.0,
        num_slots=3000, warmup_slots=300, csi_on=True,
    ),
}


# ── Config Builder ───────────────────────────────────────────────────
def build_config():
    S = st.session_state
    traffic_map = {"Full Buffer": "full_buffer", "FTP Model 3": "ftp_model3", "Poisson": "bursty"}
    num_prb = BW_PRB[S.bw].get(S.scs, 273) or 273
    return {
        "sim": SimConfig(num_slots=S.num_slots, random_seed=S.random_seed, warmup_slots=S.warmup_slots),
        "carrier": CarrierConfig(subcarrier_spacing=S.scs, num_prb=num_prb,
                                 bandwidth_mhz=float(S.bw.split()[0]), carrier_freq_ghz=S.freq),
        "cell": CellConfig(num_tx_ant=S.tx_ant, num_tx_ports=S.tx_ports, max_layers=S.max_layers,
                           total_power_dbm=S.power, cell_radius_m=float(S.radius),
                           height_m=S.bs_h, scenario=S.scenario),
        "ue": UEConfig(num_ue=S.n_ue, num_rx_ant=S.rx_ant, speed_kmh=S.speed,
                       min_distance_m=S.d_min, max_distance_m=S.d_max),
        "scheduler": SchedulerConfig(type="pf", beta=S.beta),
        "link_adaptation": LinkAdaptationConfig(bler_target=S.bler_t, olla_delta_up=S.olla_d),
        "traffic": TrafficConfig(type=traffic_map.get(S.traffic, "full_buffer")),
        "channel": ChannelConfig(type=S.ch_type, scenario=S.scenario),
        "csi": CSIConfig(enabled=S.csi_on, csi_period_slots=S.csi_period,
                         feedback_delay_slots=S.csi_delay, subband_size_prb=S.sb_prb),
        "tdd": TDDConfig(duplex_mode=S.duplex, pattern=S.tdd_pat,
                         special_dl_symbols=S.sp_dl, special_gp_symbols=S.sp_gp,
                         special_ul_symbols=S.sp_ul, harq_k1=S.k1),
    }


# ── Simulation Runner ────────────────────────────────────────────────
def run_simulation(config):
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.kpi.kpi_reporter import KPIReporter

    engine = SimulationEngine(config)
    total = config["sim"].num_slots
    bar = st.progress(0.0, text="Initializing...")
    t0 = time.time()

    for s in range(total):
        result = engine.run_slot(s)
        buf_before = engine._buf_after_traffic.copy()
        buf_after = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(s, result, buf_before, buf_after)
        if (s + 1) % max(total // 50, 1) == 0 or s == total - 1:
            elapsed = time.time() - t0
            bar.progress((s + 1) / total,
                         text=f"Slot {s+1}/{total}  |  {(s+1)/elapsed:.0f} slots/s")

    bar.progress(1.0, text=f"Done in {time.time()-t0:.1f}s")
    report = KPIReporter(engine.kpi, engine.carrier_config).report()
    harq = engine.harq_mgr.get_all_stats()

    # UE topology data
    positions = np.array([ue.position[:2] for ue in engine.ue_states])
    vr = engine.kpi.get_valid_range()
    avg_sinr = np.mean(engine.kpi.ue_sinr_eff_db[vr], axis=0)
    avg_tp = np.mean(engine.kpi.ue_throughput_bits[vr], axis=0) / engine.carrier_config.slot_duration_s / 1e6

    topo = pd.DataFrame({
        "x (m)": positions[:, 0], "y (m)": positions[:, 1],
        "UE": [f"UE {i}" for i in range(len(positions))],
        "SINR (dB)": np.round(avg_sinr, 1),
        "Throughput (Mbps)": np.round(avg_tp, 1),
        "BLER": np.round(report["bler_per_ue"], 3),
    })

    return report, engine.kpi, engine.carrier_config, harq, topo


# ── Topology Plot ────────────────────────────────────────────────────
def render_topology(topo: pd.DataFrame, cell_radius: float, color_by: str = "SINR (dB)"):
    fig = px.scatter(
        topo, x="x (m)", y="y (m)", color=color_by,
        hover_data=["UE", "SINR (dB)", "Throughput (Mbps)", "BLER"],
        color_continuous_scale="Viridis", size_max=14,
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))

    # BS marker at origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=18, color=COLORS["orange"], symbol="star",
                    line=dict(width=2, color="white")),
        text=["BS"], textposition="top center",
        textfont=dict(size=12, color=COLORS["slate"]),
        showlegend=False, hoverinfo="text", hovertext="Base Station (0, 0)",
    ))

    # Cell boundary circle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=cell_radius * np.cos(theta), y=cell_radius * np.sin(theta),
        mode="lines", line=dict(color=COLORS["border"], width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        height=480,
        xaxis=dict(title="East (m)", scaleanchor="y", range=[-cell_radius*1.15, cell_radius*1.15]),
        yaxis=dict(title="North (m)", range=[-cell_radius*1.15, cell_radius*1.15]),
        margin=dict(l=40, r=40, t=30, b=40),
        plot_bgcolor="white",
        coloraxis_colorbar=dict(title=color_by, thickness=15),
    )
    return fig


# ── Chart Helpers ────────────────────────────────────────────────────
def chart_throughput_cdf(kpi, carrier):
    s = kpi.get_valid_range()
    tp = np.sort(np.mean(kpi.ue_throughput_bits[s], axis=0) / carrier.slot_duration_s / 1e6)
    cdf = np.arange(1, len(tp) + 1) / max(len(tp), 1)
    fig = px.line(x=tp, y=cdf, labels={"x": "UE Avg Throughput (Mbps)", "y": "CDF"})
    fig.update_traces(line=dict(color=COLORS["teal"], width=2.5))
    fig.update_layout(height=350, margin=dict(t=20, b=40))
    return fig

def chart_bler_trend(kpi):
    s = kpi.get_valid_range()
    success = kpi.ue_tb_success[s].astype(float)
    sched = kpi.ue_num_prbs[s] > 0
    bler = np.array([1.0 - np.mean(success[i, sched[i]]) if np.any(sched[i]) else 0
                     for i in range(success.shape[0])])
    w = min(100, max(len(bler) // 5, 1))
    smoothed = np.convolve(bler, np.ones(w)/w, mode="valid")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=smoothed, mode="lines", name="BLER",
                             line=dict(color=COLORS["orange"], width=2)))
    fig.add_hline(y=0.1, line_dash="dash", line_color=COLORS["slate"],
                  annotation_text="Target 10%", annotation_position="top left")
    fig.update_layout(height=350, margin=dict(t=20, b=40),
                      xaxis_title="Slot", yaxis_title="BLER")
    return fig

def chart_mcs_dist(kpi):
    s = kpi.get_valid_range()
    mcs = kpi.ue_mcs[s][kpi.ue_num_prbs[s] > 0]
    if len(mcs) == 0:
        return None
    counts = np.bincount(mcs, minlength=29)[:29]
    df = pd.DataFrame({"MCS": range(29), "Count": counts})
    fig = px.bar(df[df.Count > 0], x="MCS", y="Count", color_discrete_sequence=[COLORS["teal"]])
    fig.update_layout(height=350, margin=dict(t=20, b=40))
    return fig

def chart_sinr_cdf(kpi):
    s = kpi.get_valid_range()
    sinr = kpi.ue_sinr_eff_db[s].ravel()
    sinr = np.sort(sinr[sinr > -29])
    if len(sinr) == 0:
        return None
    cdf = np.arange(1, len(sinr)+1) / len(sinr)
    fig = px.line(x=sinr, y=cdf, labels={"x": "SINR (dB)", "y": "CDF"})
    fig.update_traces(line=dict(color=COLORS["orange"], width=2.5))
    fig.update_layout(height=350, margin=dict(t=20, b=40))
    return fig

def chart_cell_tp(kpi, carrier):
    s = kpi.get_valid_range()
    delivered = kpi.cell_throughput_bits[s] / carrier.slot_duration_s / 1e6
    scheduled = kpi.cell_scheduled_bits[s] / carrier.slot_duration_s / 1e6
    df = pd.DataFrame({"Scheduled": scheduled, "Delivered": delivered})
    w = min(50, max(len(df) // 10, 1))
    df = df.rolling(window=w).mean().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["Scheduled"], mode="lines", name="Scheduled",
                             line=dict(color=COLORS["slate"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(y=df["Delivered"], mode="lines", name="Delivered",
                             line=dict(color=COLORS["teal"], width=2.5)))
    fig.update_layout(height=350, margin=dict(t=20, b=40),
                      xaxis_title="Slot", yaxis_title="Mbps")
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────
def render_sidebar():
    S = st.session_state
    with st.sidebar:
        st.markdown("### Configuration")

        with st.expander("Simulation", expanded=True):
            st.number_input("Slots", 100, 50000, step=100, key="num_slots")
            st.number_input("Seed", 0, 99999, key="random_seed")
            st.number_input("Warmup", 0, 5000, step=50, key="warmup_slots")

        with st.expander("Frame Structure", expanded=True):
            st.selectbox("Duplex", ["TDD", "FDD"], key="duplex")
            if S.duplex == "TDD":
                st.selectbox("Pattern", TDD_PATTERNS, key="tdd_pat")
                c1, c2, c3 = st.columns(3)
                c1.number_input("DL", 1, 12, key="sp_dl")
                c2.number_input("GP", 0, 6, key="sp_gp")
                c3.number_input("UL", 0, 6, key="sp_ul")
                st.slider("HARQ K1", 1, 10, key="k1")
            st.selectbox("SCS (kHz)", [15, 30, 60], key="scs")
            st.selectbox("Bandwidth", list(BW_PRB.keys()), key="bw")

        with st.expander("Cell & Antenna", expanded=False):
            st.selectbox("Scenario", ["uma", "umi", "rma"], key="scenario")
            st.selectbox("TX Antennas", [8, 16, 32, 64], key="tx_ant")
            st.selectbox("CSI-RS Ports", [2, 4, 8], key="tx_ports")
            st.selectbox("Max Layers", [1, 2, 4], key="max_layers")
            st.slider("Power (dBm)", 30.0, 56.0, step=1.0, key="power")
            st.slider("Cell Radius (m)", 100, 2000, step=50, key="radius")

        with st.expander("UE", expanded=False):
            st.slider("UE Count", 1, 50, key="n_ue")
            st.selectbox("RX Antennas", [1, 2, 4], key="rx_ant")
            st.slider("Speed (km/h)", 0.0, 120.0, step=1.0, key="speed")
            c1, c2 = st.columns(2)
            c1.number_input("Min Dist", 10.0, 2000.0, step=5.0, key="d_min")
            c2.number_input("Max Dist", 10.0, 2000.0, step=5.0, key="d_max")

        with st.expander("Scheduler & LA", expanded=False):
            st.slider("PF Beta", 0.9, 0.999, step=0.001, format="%.3f", key="beta")
            st.slider("BLER Target", 0.01, 0.30, step=0.01, key="bler_t")

        with st.expander("Traffic & Channel", expanded=False):
            st.selectbox("Traffic", ["Full Buffer", "FTP Model 3", "Poisson"], key="traffic")
            st.selectbox("Channel", ["statistical", "sionna"], key="ch_type")
            st.checkbox("CSI Feedback", key="csi_on")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="L2 RRM Simulator", page_icon="📡", layout="wide")
    inject_css()
    init_state()
    render_sidebar()

    # ── Header + Presets ──
    st.markdown("# 📡 L2 RRM Simulator")
    cols = st.columns(len(PRESETS) + 1)
    for i, (name, vals) in enumerate(PRESETS.items()):
        if cols[i].button(name, use_container_width=True):
            for k, v in vals.items():
                st.session_state[k] = v
            st.session_state.preset = name
            st.rerun()

    S = st.session_state
    duplex_label = f"TDD {S.tdd_pat}" if S.duplex == "TDD" else "FDD"
    cols[-1].markdown(
        f"<div style='text-align:center;padding:0.5rem;color:{COLORS['slate']};font-size:0.9rem'>"
        f"<b>{S.preset}</b><br>{S.n_ue} UE · {S.bw} · {duplex_label}</div>",
        unsafe_allow_html=True)

    # ── Validation ──
    if S.d_min > S.d_max:
        st.error("Min Distance > Max Distance"); return
    if S.duplex == "TDD" and S.sp_dl + S.sp_gp + S.sp_ul > 14:
        st.error("Special slot symbols exceed 14"); return

    # ── Run Button ──
    if st.button("Run Simulation", type="primary", use_container_width=True):
        config = build_config()
        report, kpi, carrier, harq, topo = run_simulation(config)
        st.session_state.update(dict(
            report=report, kpi=kpi, carrier=carrier, harq=harq, topo=topo, cfg=config))

    if "report" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")
        return

    # ── Unpack results ──
    report = st.session_state.report
    kpi = st.session_state.kpi
    carrier = st.session_state.carrier
    harq = st.session_state.harq
    topo = st.session_state.topo
    cfg = st.session_state.cfg

    # ── Hero Card ──
    avg_rank = report.get("avg_rank", 1.0)
    st.markdown(f"""<div class="kpi-hero">
        <div class="title">{report['cell_avg_throughput_mbps']:.1f} Mbps</div>
        <div class="subtitle">Cell Average Throughput (Delivered)</div>
        <div class="pills">
            <span class="pill">BLER {report['avg_bler']:.3f}</span>
            <span class="pill">MCS {report['avg_mcs']:.1f}</span>
            <span class="pill">Rank {avg_rank:.2f}</span>
            <span class="pill">PRB {report['prb_utilization']*100:.0f}%</span>
            <span class="pill">Delivery {report['delivery_ratio']*100:.1f}%</span>
            <span class="pill">HARQ retx {harq['retx_rate']*100:.1f}%</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI Metrics ──
    r1 = st.columns(6)
    r1[0].metric("Delivered TP", f"{report['cell_avg_throughput_mbps']:.1f} Mbps")
    r1[1].metric("Scheduled TP", f"{report['cell_avg_scheduled_throughput_mbps']:.1f} Mbps")
    r1[2].metric("Avg BLER", f"{report['avg_bler']:.3f}")
    r1[3].metric("Avg SINR", f"{report['avg_sinr_db']:.1f} dB")
    r1[4].metric("Spectral Eff.", f"{report['spectral_efficiency_bps_hz']:.2f} bps/Hz")
    r1[5].metric("Jain Fairness", f"{report['jain_fairness']:.3f}")

    # ── Topology + Charts ──
    tab_topo, tab_charts, tab_ue, tab_trace = st.tabs(
        ["Cell Topology", "Performance Charts", "UE Details", "Slot Trace"])

    with tab_topo:
        col_map, col_ctrl = st.columns([4, 1])
        with col_ctrl:
            color_by = st.radio("Color by", ["SINR (dB)", "Throughput (Mbps)", "BLER"],
                                label_visibility="collapsed")
        with col_map:
            fig = render_topology(topo, float(cfg["cell"].cell_radius_m), color_by)
            st.plotly_chart(fig, use_container_width=True)

        # UE summary table below map
        st.dataframe(topo.style.format({
            "x (m)": "{:.0f}", "y (m)": "{:.0f}",
            "SINR (dB)": "{:.1f}", "Throughput (Mbps)": "{:.1f}", "BLER": "{:.3f}",
        }), use_container_width=True, hide_index=True)

    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Throughput CDF**")
            st.plotly_chart(chart_throughput_cdf(kpi, carrier), use_container_width=True)
            st.markdown("**MCS Distribution**")
            fig = chart_mcs_dist(kpi)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**BLER Trend**")
            st.plotly_chart(chart_bler_trend(kpi), use_container_width=True)
            st.markdown("**SINR CDF**")
            fig = chart_sinr_cdf(kpi)
            if fig: st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cell Throughput Trend**")
        st.plotly_chart(chart_cell_tp(kpi, carrier), use_container_width=True)

    with tab_ue:
        ue_df = pd.DataFrame({
            "UE": range(len(report["ue_avg_throughput_mbps"])),
            "Avg TP (Mbps)": np.round(report["ue_avg_throughput_mbps"], 2),
            "Sched TP (Mbps)": np.round(report["ue_avg_scheduled_throughput_mbps"], 2),
            "BLER": np.round(report["bler_per_ue"], 4),
            "Sched %": np.round(report["ue_scheduling_ratio"] * 100, 1),
        })
        st.dataframe(ue_df, use_container_width=True, hide_index=True)

    with tab_trace:
        st.caption("Slot-level audit: scheduled vs delivered bits, per-UE breakdown.")
        trace = report["slot_trace"]
        n_ue = trace["ue_num_prbs"].shape[1] if trace["ue_num_prbs"].ndim == 2 else 0
        focus = st.number_input("Focus UE", 0, max(n_ue - 1, 0), 0)
        rows = st.slider("Rows", 20, 200, 50)
        frame = pd.DataFrame({
            "slot": trace["slot_idx"], "dir": trace["slot_direction"],
            "dl_sym": trace["num_dl_symbols"],
            "cell_sched": trace["cell_scheduled_bits"],
            "cell_deliv": trace["cell_delivered_bits"],
            "ue_prbs": trace["ue_num_prbs"][:, focus] if n_ue else 0,
            "ue_mcs": trace["ue_mcs"][:, focus] if n_ue else 0,
            "ue_rank": trace["ue_rank"][:, focus] if n_ue else 0,
            "ue_sinr": np.round(trace["ue_sinr_eff_db"][:, focus], 1) if n_ue else 0,
            "tb_ok": trace["ue_tb_success"][:, focus] if n_ue else False,
        })
        st.dataframe(frame.tail(rows).reset_index(drop=True),
                     use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

"""L2 RRM 5G NR System-Level Simulator

Run:  .venv312/Scripts/streamlit.exe run app.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from l2_rrm_sim.config.sim_config import (
    CSIConfig, CarrierConfig, CellConfig, ChannelConfig,
    LinkAdaptationConfig, SchedulerConfig, SimConfig,
    TDDConfig, TrafficConfig, UEConfig,
)

BW_PRB = {
    "20 MHz":  {15: 106, 30: 51,  60: 24},
    "50 MHz":  {15: 270, 30: 133, 60: 65},
    "100 MHz": {15: 0,   30: 273, 60: 135},
}
TDD_PATTERNS = ["DDDSU", "DDSUU", "DDDSUDDSUU", "DDDDDDDSUU"]

# ── CSS ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
    .stApp { font-family: 'Noto Sans SC', -apple-system, sans-serif; }
    .main .block-container { max-width: 1400px; padding-top: 1rem; }
    [data-testid="stSidebar"] { background: #f8f9fa; border-right: 1px solid #e5e7eb; }
    div[data-testid="metric-container"] {
        background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 0.6rem 0.8rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .sec-label {
        font-size: 0.7rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 0.3rem; font-weight: 600;
    }
    .tdd-frame { display: flex; gap: 2px; margin: 0.3rem 0; }
    .tdd-slot {
        flex: 1; height: 24px; border-radius: 3px; display: flex;
        align-items: center; justify-content: center;
        font-size: 0.65rem; font-weight: 700;
    }
    .tdd-slot.D { background: #dbeafe; color: #2563eb; border: 1px solid #bfdbfe; }
    .tdd-slot.S { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
    .tdd-slot.U { background: #f3f4f6; color: #9ca3af; border: 1px solid #e5e7eb; }
    </style>""", unsafe_allow_html=True)


# ── State / Presets ──────────────────────────────────────────────────
def init_state():
    defaults = dict(
        preset="TDD Baseline", num_slots=2000, random_seed=42, warmup_slots=200,
        scs=30, bw="100 MHz", freq=3.5,
        duplex="TDD", tdd_pat="DDDSU", sp_dl=10, sp_gp=2, sp_ul=2, k1=4,
        scenario="uma", tx_ant=32, tx_ports=4, max_layers=1,
        power=46.0, radius=500, bs_h=25.0,
        n_ue=1, rx_ant=2, speed=3.0, d_min=35.0, d_max=500.0,
        sched_type="pf", beta=0.98, bler_t=0.1, olla_d=0.5,
        traffic="Full Buffer", ch_type="statistical",
        csi_on=False, csi_period=10, csi_delay=4, sb_prb=4,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

PRESETS = {
    "TDD Baseline": dict(duplex="TDD", tdd_pat="DDDSU", n_ue=1, max_layers=1,
                          d_min=35.0, d_max=35.0, num_slots=2000, warmup_slots=200),
    "FDD Rank-1":   dict(duplex="FDD", n_ue=1, max_layers=1,
                          d_min=35.0, d_max=35.0, num_slots=2000, warmup_slots=200),
    "Multi-UE PF":  dict(duplex="TDD", tdd_pat="DDDSU", n_ue=10, max_layers=2,
                          d_min=35.0, d_max=500.0, num_slots=3000, warmup_slots=300, csi_on=True),
}

# ── Config ───────────────────────────────────────────────────────────
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
        "scheduler": SchedulerConfig(type=S.sched_type, beta=S.beta),
        "link_adaptation": LinkAdaptationConfig(bler_target=S.bler_t, olla_delta_up=S.olla_d),
        "traffic": TrafficConfig(type=traffic_map.get(S.traffic, "full_buffer")),
        "channel": ChannelConfig(type=S.ch_type, scenario=S.scenario),
        "csi": CSIConfig(enabled=S.csi_on, csi_period_slots=S.csi_period,
                         feedback_delay_slots=S.csi_delay, subband_size_prb=S.sb_prb),
        "tdd": TDDConfig(duplex_mode=S.duplex, pattern=S.tdd_pat,
                         special_dl_symbols=S.sp_dl, special_gp_symbols=S.sp_gp,
                         special_ul_symbols=S.sp_ul, harq_k1=S.k1),
    }

# ── Simulation ───────────────────────────────────────────────────────
def run_simulation(config):
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.kpi.kpi_reporter import KPIReporter
    engine = SimulationEngine(config)
    total = config["sim"].num_slots
    bar = st.progress(0.0, text=f"Running 0/{total}...")
    t0 = time.time()
    for s in range(total):
        result = engine.run_slot(s)
        buf_before = engine._buf_after_traffic.copy()
        buf_after = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(s, result, buf_before, buf_after)
        if (s + 1) % max(total // 50, 1) == 0 or s == total - 1:
            elapsed = time.time() - t0
            bar.progress((s+1)/total, text=f"Slot {s+1}/{total} | {(s+1)/elapsed:.0f} slots/s")
    bar.progress(1.0, text=f"Done in {time.time()-t0:.1f}s")
    report = KPIReporter(engine.kpi, engine.carrier_config).report()
    harq = engine.harq_mgr.get_all_stats()
    positions = np.array([ue.position[:2] for ue in engine.ue_states])
    vr = engine.kpi.get_valid_range()
    avg_sinr = np.mean(engine.kpi.ue_sinr_eff_db[vr], axis=0)
    avg_tp = np.mean(engine.kpi.ue_throughput_bits[vr], axis=0) / engine.carrier_config.slot_duration_s / 1e6
    topo = pd.DataFrame({
        "x": positions[:, 0], "y": positions[:, 1],
        "UE": [f"UE-{i}" for i in range(len(positions))],
        "Dist (m)": np.round(np.sqrt(positions[:,0]**2 + positions[:,1]**2), 0),
        "SINR (dB)": np.round(avg_sinr, 1),
        "TP (Mbps)": np.round(avg_tp, 1),
        "BLER": np.round(report["bler_per_ue"], 3),
    })
    return report, engine.kpi, engine.carrier_config, harq, topo

# ── Plotly helpers ───────────────────────────────────────────────────
def _theme(fig, h=360):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafbfc",
        font=dict(family="Noto Sans SC, sans-serif", color="#374151", size=11),
        margin=dict(l=45, r=15, t=25, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=h,
    )
    fig.update_xaxes(gridcolor="#f0f0f0", zerolinecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#f0f0f0", zerolinecolor="#e5e7eb")
    return fig

def _make_cell_canvas(radius, n_ue, d_min, d_max, seed=42):
    """Generate UE positions for the preview canvas (before simulation)."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(d_min**2, d_max**2, n_ue))
    theta = rng.uniform(0, 2*np.pi, n_ue)
    return r * np.cos(theta), r * np.sin(theta)

def render_cell_canvas(radius, n_ue, d_min, d_max, seed, topo_df=None):
    """Draw the cell topology canvas — works both before and after simulation."""
    fig = go.Figure()
    # Cell boundary rings
    for frac in [0.25, 0.5, 0.75, 1.0]:
        r = radius * frac
        t = np.linspace(0, 2*np.pi, 80)
        fig.add_trace(go.Scatter(x=r*np.cos(t), y=r*np.sin(t), mode="lines",
            line=dict(color="#e5e7eb", width=0.8, dash="dot"),
            showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=r*0.71, y=r*0.71, text=f"{r:.0f}m", showarrow=False,
                           font=dict(size=8, color="#9ca3af"), bgcolor="white", borderpad=1)

    if topo_df is not None and len(topo_df) > 0:
        # Post-simulation: color by SINR
        fig.add_trace(go.Scatter(
            x=topo_df["x"], y=topo_df["y"], mode="markers+text",
            marker=dict(size=12, color=topo_df["SINR (dB)"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="SINR(dB)", thickness=10, len=0.5),
                        line=dict(width=1, color="white")),
            text=topo_df["UE"], textposition="top center",
            textfont=dict(size=8, color="#6b7280"),
            customdata=np.stack([topo_df["UE"], topo_df["Dist (m)"],
                                 topo_df["SINR (dB)"], topo_df["TP (Mbps)"], topo_df["BLER"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}m<br>"
                          "SINR: %{customdata[2]} dB<br>TP: %{customdata[3]} Mbps<br>"
                          "BLER: %{customdata[4]}<extra></extra>",
            showlegend=False,
        ))
    else:
        # Pre-simulation: preview positions
        ux, uy = _make_cell_canvas(radius, n_ue, d_min, d_max, seed)
        fig.add_trace(go.Scatter(
            x=ux, y=uy, mode="markers+text",
            marker=dict(size=10, color="#3b82f6", line=dict(width=1, color="white")),
            text=[f"UE-{i}" for i in range(n_ue)],
            textposition="top center", textfont=dict(size=8, color="#9ca3af"),
            showlegend=False, hoverinfo="text",
            hovertext=[f"UE-{i}" for i in range(n_ue)],
        ))

    # BS
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
        marker=dict(size=16, color="#ef4444", symbol="diamond", line=dict(width=2, color="white")),
        text=["gNB"], textposition="top center", textfont=dict(size=10, color="#ef4444"),
        showlegend=False, hoverinfo="text", hovertext="Base Station"))

    lim = radius * 1.15
    _theme(fig, h=450)
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title="East (m)", scaleanchor="y", range=[-lim, lim])
    fig.update_yaxes(title="North (m)", range=[-lim, lim])
    return fig

def chart_tp_cdf(kpi, carrier):
    s = kpi.get_valid_range()
    tp = np.sort(np.mean(kpi.ue_throughput_bits[s], axis=0) / carrier.slot_duration_s / 1e6)
    cdf = np.arange(1, len(tp)+1) / max(len(tp), 1)
    fig = go.Figure(go.Scatter(x=tp, y=cdf, mode="lines", line=dict(color="#2563eb", width=2)))
    fig.update_layout(xaxis_title="Throughput (Mbps)", yaxis_title="CDF")
    return _theme(fig)

def chart_bler(kpi):
    s = kpi.get_valid_range()
    success = kpi.ue_tb_success[s].astype(float)
    sched = kpi.ue_num_prbs[s] > 0
    bler = np.array([1.0 - np.mean(success[i, sched[i]]) if np.any(sched[i]) else 0
                     for i in range(success.shape[0])])
    w = min(100, max(len(bler)//5, 1))
    sm = np.convolve(bler, np.ones(w)/w, mode="valid")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sm, mode="lines", name="BLER", line=dict(color="#f59e0b", width=2)))
    fig.add_hline(y=0.1, line_dash="dash", line_color="#ef4444", line_width=1,
                  annotation_text="Target", annotation_font_size=9)
    fig.update_layout(xaxis_title="Slot", yaxis_title="BLER")
    return _theme(fig)

def chart_mcs(kpi):
    s = kpi.get_valid_range()
    mcs = kpi.ue_mcs[s][kpi.ue_num_prbs[s] > 0]
    if len(mcs) == 0: return None
    counts = np.bincount(mcs, minlength=29)[:29]
    fig = go.Figure(go.Bar(x=list(range(29)), y=counts, marker_color="#2563eb", marker_line_width=0))
    fig.update_layout(xaxis_title="MCS", yaxis_title="Count", bargap=0.15)
    return _theme(fig)

def chart_sinr(kpi):
    s = kpi.get_valid_range()
    sinr = np.sort(kpi.ue_sinr_eff_db[s].ravel())
    sinr = sinr[sinr > -29]
    if len(sinr) == 0: return None
    cdf = np.arange(1, len(sinr)+1) / len(sinr)
    fig = go.Figure(go.Scatter(x=sinr, y=cdf, mode="lines", line=dict(color="#f59e0b", width=2)))
    fig.update_layout(xaxis_title="SINR (dB)", yaxis_title="CDF")
    return _theme(fig)

def chart_cell_tp(kpi, carrier):
    s = kpi.get_valid_range()
    delivered = kpi.cell_throughput_bits[s] / carrier.slot_duration_s / 1e6
    scheduled = kpi.cell_scheduled_bits[s] / carrier.slot_duration_s / 1e6
    df = pd.DataFrame({"Scheduled": scheduled, "Delivered": delivered})
    w = min(50, max(len(df)//10, 1))
    df = df.rolling(window=w).mean().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["Scheduled"], mode="lines", name="Scheduled",
                             line=dict(color="#9ca3af", width=1, dash="dot")))
    fig.add_trace(go.Scatter(y=df["Delivered"], mode="lines", name="Delivered",
                             line=dict(color="#2563eb", width=2),
                             fill="tozeroy", fillcolor="rgba(37,99,235,0.05)"))
    fig.update_layout(xaxis_title="Slot", yaxis_title="Mbps")
    return _theme(fig)

def render_tdd_frame(pattern, sp_dl, sp_gp, sp_ul):
    slots = ""
    for ch in pattern:
        if ch == 'D': slots += '<div class="tdd-slot D">D</div>'
        elif ch == 'S': slots += f'<div class="tdd-slot S">S</div>'
        elif ch == 'U': slots += '<div class="tdd-slot U">U</div>'
    return f'<div class="tdd-frame">{slots}</div>'


# ── Sidebar ──────────────────────────────────────────────────────────
def render_sidebar():
    S = st.session_state
    with st.sidebar:
        st.markdown("### L2 RRM Simulator")
        st.caption("5G NR System-Level Simulation")
        st.markdown("---")

        # Presets
        st.markdown('<div class="sec-label">Quick Presets</div>', unsafe_allow_html=True)
        for name, vals in PRESETS.items():
            if st.button(name, key=f"p_{name}", use_container_width=True):
                for k, v in vals.items():
                    st.session_state[k] = v
                st.session_state.preset = name
                st.rerun()
        st.markdown("---")

        # Simulation
        st.markdown('<div class="sec-label">Simulation</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.number_input("Slots", 100, 50000, step=100, key="num_slots")
        c2.number_input("Warmup", 0, 5000, step=50, key="warmup_slots")
        st.number_input("Seed", 0, 99999, key="random_seed")
        st.markdown("---")

        # Radio
        st.markdown('<div class="sec-label">Radio Frame</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.selectbox("Duplex", ["TDD", "FDD"], key="duplex")
        c2.selectbox("SCS", [15, 30, 60], key="scs", format_func=lambda x: f"{x} kHz")
        st.selectbox("Bandwidth", list(BW_PRB.keys()), key="bw")
        if S.duplex == "TDD":
            st.selectbox("TDD Pattern", TDD_PATTERNS, key="tdd_pat")
            st.markdown(render_tdd_frame(S.tdd_pat, S.sp_dl, S.sp_gp, S.sp_ul), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.number_input("DL", 1, 12, key="sp_dl")
            c2.number_input("GP", 0, 6, key="sp_gp")
            c3.number_input("UL", 0, 6, key="sp_ul")
        st.markdown("---")

        # Cell
        st.markdown('<div class="sec-label">Cell</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.selectbox("Scenario", ["uma", "umi", "rma"], key="scenario")
        c2.selectbox("TX Ant", [8, 16, 32, 64], key="tx_ant")
        c1, c2 = st.columns(2)
        c1.selectbox("Ports", [2, 4, 8], key="tx_ports")
        c2.selectbox("Layers", [1, 2, 4], key="max_layers")
        st.slider("Power (dBm)", 30.0, 56.0, step=1.0, key="power")
        st.slider("Cell Radius (m)", 100, 2000, step=50, key="radius")
        st.markdown("---")

        # UE
        st.markdown('<div class="sec-label">UE</div>', unsafe_allow_html=True)
        st.slider("UE Count", 1, 50, key="n_ue")
        c1, c2 = st.columns(2)
        c1.selectbox("RX Ant", [1, 2, 4], key="rx_ant")
        c2.slider("Speed", 0.0, 120.0, step=1.0, key="speed", format="%.0f km/h")
        c1, c2 = st.columns(2)
        c1.number_input("Min Dist (m)", 10.0, 2000.0, step=5.0, key="d_min")
        c2.number_input("Max Dist (m)", 10.0, 2000.0, step=5.0, key="d_max")

        # Advanced
        with st.expander("Advanced", expanded=False):
            st.selectbox("Scheduler", ["pf", "mu_mimo"], key="sched_type",
                         format_func=lambda x: {"pf": "PF (SU-MIMO)", "mu_mimo": "PF (MU-MIMO)"}[x])
            st.slider("PF Beta", 0.9, 0.999, step=0.001, format="%.3f", key="beta")
            st.slider("BLER Target", 0.01, 0.30, step=0.01, key="bler_t")
            st.selectbox("Traffic", ["Full Buffer", "FTP Model 3", "Poisson"], key="traffic")
            st.selectbox("Channel", ["statistical", "sionna"], key="ch_type")
            st.checkbox("CSI Feedback", key="csi_on")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="L2 RRM Simulator", page_icon="📡", layout="wide")
    inject_css()
    init_state()
    render_sidebar()
    S = st.session_state

    # Validation
    if S.d_min > S.d_max:
        st.error("Min Distance > Max Distance"); return
    if S.duplex == "TDD" and S.sp_dl + S.sp_gp + S.sp_ul > 14:
        st.error("Special slot symbols exceed 14"); return

    # ── Title ──
    st.markdown("## L2 RRM Simulator")
    duplex_label = f"TDD {S.tdd_pat}" if S.duplex == "TDD" else "FDD"
    st.caption(f"{S.preset}  |  {S.n_ue} UE  |  {S.bw} @ {S.scs} kHz  |  "
               f"{S.tx_ant}T{S.rx_ant}R  |  {duplex_label}  |  {S.scenario.upper()}")

    # ── Scene Canvas (always visible) ──
    st.markdown("### Scene Preview")
    topo_df = st.session_state.get("topo", None)
    fig_canvas = render_cell_canvas(S.radius, S.n_ue, S.d_min, S.d_max, S.random_seed, topo_df)
    st.plotly_chart(fig_canvas, use_container_width=True)

    # ── Run Button ──
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Loading simulation engine..."):
            config = build_config()
        report, kpi, carrier, harq, topo = run_simulation(config)
        st.session_state.update(dict(report=report, kpi=kpi, carrier=carrier,
                                     harq=harq, topo=topo, cfg=config))
        st.rerun()

    if "report" not in st.session_state:
        st.info("Adjust parameters in sidebar, then click **Run Simulation**. "
                "The canvas above shows UE placement based on current settings.")
        return

    # ── Results ──
    report = st.session_state.report
    kpi = st.session_state.kpi
    carrier = st.session_state.carrier
    harq = st.session_state.harq

    st.markdown("### Results")
    avg_rank = report.get("avg_rank", 1.0)
    m = st.columns(6)
    m[0].metric("Cell Throughput", f"{report['cell_avg_throughput_mbps']:.1f} Mbps")
    m[1].metric("Spectral Eff.", f"{report['spectral_efficiency_bps_hz']:.2f} bps/Hz")
    m[2].metric("Avg BLER", f"{report['avg_bler']:.4f}")
    m[3].metric("Avg SINR", f"{report['avg_sinr_db']:.1f} dB")
    m[4].metric("Avg MCS", f"{report['avg_mcs']:.1f}")
    m[5].metric("PRB Util.", f"{report['prb_utilization']*100:.0f}%")

    m2 = st.columns(6)
    m2[0].metric("Scheduled TP", f"{report['cell_avg_scheduled_throughput_mbps']:.1f} Mbps")
    m2[1].metric("Delivery Ratio", f"{report['delivery_ratio']*100:.1f}%")
    m2[2].metric("HARQ ReTX", f"{harq['retx_rate']*100:.1f}%")
    m2[3].metric("Avg Rank", f"{avg_rank:.2f}")
    m2[4].metric("Fairness", f"{report['jain_fairness']:.3f}")
    m2[5].metric("Valid Slots", f"{report['num_valid_slots']}")

    # ── Charts ──
    st.markdown("### Charts")
    t1, t2, t3, t4, t5 = st.tabs(["Throughput", "BLER", "MCS", "SINR", "Cell TP Trend"])
    with t1: st.plotly_chart(chart_tp_cdf(kpi, carrier), use_container_width=True)
    with t2: st.plotly_chart(chart_bler(kpi), use_container_width=True)
    with t3:
        fig = chart_mcs(kpi)
        if fig: st.plotly_chart(fig, use_container_width=True)
    with t4:
        fig = chart_sinr(kpi)
        if fig: st.plotly_chart(fig, use_container_width=True)
    with t5: st.plotly_chart(chart_cell_tp(kpi, carrier), use_container_width=True)

    # ── UE Table ──
    st.markdown("### UE Details")
    ue_df = pd.DataFrame({
        "UE": range(len(report["ue_avg_throughput_mbps"])),
        "TP (Mbps)": np.round(report["ue_avg_throughput_mbps"], 2),
        "Sched TP": np.round(report["ue_avg_scheduled_throughput_mbps"], 2),
        "BLER": np.round(report["bler_per_ue"], 4),
        "Sched %": np.round(report["ue_scheduling_ratio"] * 100, 1),
    })
    st.dataframe(ue_df, use_container_width=True, hide_index=True)

    # ── Slot Trace ──
    with st.expander("Slot Trace (audit)", expanded=False):
        trace = report["slot_trace"]
        n_ue = trace["ue_num_prbs"].shape[1] if trace["ue_num_prbs"].ndim == 2 else 0
        c1, c2 = st.columns([1, 3])
        focus = c1.number_input("Focus UE", 0, max(n_ue-1, 0), 0)
        rows = c2.slider("Rows", 20, 200, 50)
        frame = pd.DataFrame({
            "slot": trace["slot_idx"], "dir": trace["slot_direction"],
            "dl_sym": trace["num_dl_symbols"],
            "cell_sched": trace["cell_scheduled_bits"],
            "cell_deliv": trace["cell_delivered_bits"],
            "prbs": trace["ue_num_prbs"][:, focus] if n_ue else 0,
            "mcs": trace["ue_mcs"][:, focus] if n_ue else 0,
            "sinr_db": np.round(trace["ue_sinr_eff_db"][:, focus], 1) if n_ue else 0,
            "tb_ok": trace["ue_tb_success"][:, focus] if n_ue else False,
        })
        st.dataframe(frame.tail(rows).reset_index(drop=True), use_container_width=True, hide_index=True)

    # Footer
    st.caption("L2 RRM Simulator v2.0 | Powered by Sionna 2.0 + PyTorch | "
               "github.com/TianLin0509/L2_RRM_Sim")

if __name__ == "__main__":
    main()

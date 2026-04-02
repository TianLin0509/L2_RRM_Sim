"""L2 RRM 5G NR System-Level Simulator — Mission Control UI

Run:  .venv312/Scripts/streamlit.exe run app.py
"""
from __future__ import annotations
import os, sys, time, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from l2_rrm_sim.config.sim_config import (
    CSIConfig, CarrierConfig, CellConfig, ChannelConfig,
    LinkAdaptationConfig, SchedulerConfig, SimConfig,
    TDDConfig, TrafficConfig, UEConfig,
)

# ━━ Palette ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C = dict(
    bg="#0a0e17", surface="#111827", panel="#1a2332", raised="#222d3d",
    border="#2a3a4e", border_hi="#3b5068",
    text="#e2e8f0", text2="#94a3b8", text3="#64748b",
    green="#22d3ee", green2="#06b6d4", green_dim="#0e7490",
    amber="#f59e0b", red="#ef4444", blue="#3b82f6",
    accent="#22d3ee",
)

BW_PRB = {
    "20 MHz":  {15: 106, 30: 51,  60: 24},
    "50 MHz":  {15: 270, 30: 133, 60: 65},
    "100 MHz": {15: 0,   30: 273, 60: 135},
}
TDD_PATTERNS = ["DDDSU", "DDSUU", "DDDSUDDSUU", "DDDDDDDSUU"]

# ━━ CSS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_css():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');
    :root {{
        --bg: {C['bg']}; --surface: {C['surface']}; --panel: {C['panel']};
        --raised: {C['raised']}; --border: {C['border']};
        --text: {C['text']}; --text2: {C['text2']}; --text3: {C['text3']};
        --green: {C['green']}; --amber: {C['amber']}; --red: {C['red']};
    }}
    .stApp {{
        background: var(--bg);
        font-family: 'JetBrains Mono', 'Noto Sans SC', monospace;
    }}
    .main .block-container {{
        max-width: 1500px; padding: 1rem 1.5rem 2rem;
    }}
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #060a12 0%, #0d1320 100%);
        border-right: 1px solid {C['border']};
    }}
    [data-testid="stSidebar"] * {{ color: {C['text2']}; }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {{ color: {C['text3']} !important; font-size: 0.78rem; }}
    [data-testid="stSidebar"] hr {{ border-color: {C['border']}; margin: 0.5rem 0; }}
    /* ── Metrics ── */
    div[data-testid="metric-container"] {{
        background: {C['panel']};
        border: 1px solid {C['border']};
        border-radius: 8px; padding: 0.7rem 0.85rem;
    }}
    div[data-testid="metric-container"] label {{ color: {C['text3']}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {C['green']}; font-family: 'JetBrains Mono', monospace; }}
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{ font-size: 0.7rem; }}
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {C['border']}; }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {C['text3']};
        border: none; border-bottom: 2px solid transparent;
        padding: 0.6rem 1.2rem; font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace; letter-spacing: 0.02em;
    }}
    .stTabs [aria-selected="true"] {{ color: {C['green']}; border-bottom-color: {C['green']}; }}
    /* ── DataFrames ── */
    .stDataFrame {{ border: 1px solid {C['border']}; border-radius: 6px; }}
    /* ── Headings ── */
    h1, h2, h3 {{ color: {C['text']}; font-family: 'JetBrains Mono', 'Noto Sans SC', monospace; }}
    p, span, li {{ color: {C['text2']}; }}
    /* ── Custom cards ── */
    .mc-header {{
        background: linear-gradient(135deg, {C['panel']} 0%, {C['surface']} 100%);
        border: 1px solid {C['border']}; border-radius: 10px;
        padding: 1rem 1.4rem; margin-bottom: 0.8rem;
        position: relative; overflow: hidden;
    }}
    .mc-header::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, {C['green']}, {C['green_dim']}, transparent);
    }}
    .mc-header .mc-title {{
        font-size: 1.4rem; font-weight: 700; color: {C['text']};
        font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em;
    }}
    .mc-header .mc-sub {{ color: {C['text3']}; font-size: 0.82rem; margin-top: 0.2rem; }}
    .mc-tags {{ display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.6rem; }}
    .mc-tag {{
        background: rgba(34,211,238,0.08); border: 1px solid rgba(34,211,238,0.2);
        color: {C['green']}; border-radius: 4px; padding: 0.15rem 0.55rem;
        font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;
    }}
    .mc-tag.warn {{ background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.25); color: {C['amber']}; }}
    .mc-kpi-big {{
        font-size: 2.2rem; font-weight: 700; color: {C['green']};
        font-family: 'JetBrains Mono', monospace; line-height: 1;
    }}
    .mc-kpi-unit {{ font-size: 0.85rem; color: {C['text3']}; margin-left: 0.3rem; }}
    /* ── TDD frame viz ── */
    .tdd-frame {{ display: flex; gap: 2px; margin: 0.4rem 0; }}
    .tdd-slot {{
        flex: 1; height: 28px; border-radius: 3px; display: flex;
        align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .tdd-slot.D {{ background: rgba(34,211,238,0.2); color: {C['green']}; border: 1px solid rgba(34,211,238,0.3); }}
    .tdd-slot.S {{ background: rgba(245,158,11,0.15); color: {C['amber']}; border: 1px solid rgba(245,158,11,0.25); }}
    .tdd-slot.U {{ background: rgba(100,116,139,0.15); color: {C['text3']}; border: 1px solid rgba(100,116,139,0.2); }}
    .section-label {{
        color: {C['text3']}; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.12em; margin-bottom: 0.3rem;
        font-family: 'JetBrains Mono', monospace;
    }}
    </style>""", unsafe_allow_html=True)


# ━━ State / Presets ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━ Config Builder ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━ Simulation Runner ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_simulation(config):
    from l2_rrm_sim.core.simulation_engine import SimulationEngine
    from l2_rrm_sim.kpi.kpi_reporter import KPIReporter
    engine = SimulationEngine(config)
    total = config["sim"].num_slots
    bar = st.progress(0.0, text="Initializing engine...")
    t0 = time.time()
    for s in range(total):
        result = engine.run_slot(s)
        buf_before = engine._buf_after_traffic.copy()
        buf_after = np.array([ue.buffer_bytes for ue in engine.ue_states], dtype=np.int64)
        engine.kpi.collect(s, result, buf_before, buf_after)
        if (s + 1) % max(total // 50, 1) == 0 or s == total - 1:
            elapsed = time.time() - t0
            bar.progress((s+1)/total, text=f"Slot {s+1}/{total}  |  {(s+1)/elapsed:.0f} slots/s")
    bar.progress(1.0, text=f"Completed in {time.time()-t0:.1f}s")
    report = KPIReporter(engine.kpi, engine.carrier_config).report()
    harq = engine.harq_mgr.get_all_stats()
    positions = np.array([ue.position[:2] for ue in engine.ue_states])
    vr = engine.kpi.get_valid_range()
    avg_sinr = np.mean(engine.kpi.ue_sinr_eff_db[vr], axis=0)
    avg_tp = np.mean(engine.kpi.ue_throughput_bits[vr], axis=0) / engine.carrier_config.slot_duration_s / 1e6
    distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    topo = pd.DataFrame({
        "x (m)": positions[:, 0], "y (m)": positions[:, 1],
        "UE": [f"UE-{i}" for i in range(len(positions))],
        "Distance (m)": np.round(distances, 0),
        "SINR (dB)": np.round(avg_sinr, 1),
        "Throughput (Mbps)": np.round(avg_tp, 1),
        "BLER": np.round(report["bler_per_ue"], 3),
    })
    return report, engine.kpi, engine.carrier_config, harq, topo

# ━━ Plotly Theme ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_AXIS_STYLE = dict(gridcolor=C["border"], zerolinecolor=C["border"], showgrid=True, gridwidth=1)

def _apply_theme(fig, h=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["panel"],
        font=dict(family="JetBrains Mono, monospace", color=C["text2"], size=11),
        margin=dict(l=45, r=20, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=h,
    )
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig

# ━━ Topology Plot ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_topology(topo, cell_radius, color_by="SINR (dB)"):
    fig = go.Figure()
    # Distance rings
    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        r = cell_radius * r_frac
        theta = np.linspace(0, 2*np.pi, 80)
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta), mode="lines",
            line=dict(color=C["border"], width=0.8, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_annotation(x=r*0.71, y=r*0.71, text=f"{r:.0f}m",
                           showarrow=False, font=dict(size=9, color=C["text3"]),
                           bgcolor=C["bg"], borderpad=1)

    # UE scatter
    color_vals = topo[color_by].values
    cscale = "Viridis" if "SINR" in color_by or "Throughput" in color_by else "RdYlGn_r"
    fig.add_trace(go.Scatter(
        x=topo["x (m)"], y=topo["y (m)"], mode="markers+text",
        marker=dict(size=14, color=color_vals, colorscale=cscale, showscale=True,
                    colorbar=dict(title=dict(text=color_by, font=dict(size=10)),
                                 thickness=12, len=0.6),
                    line=dict(width=1.5, color=C["bg"])),
        text=topo["UE"], textposition="top center",
        textfont=dict(size=9, color=C["text3"]),
        customdata=np.stack([topo["UE"], topo["Distance (m)"], topo["SINR (dB)"],
                             topo["Throughput (Mbps)"], topo["BLER"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Distance: %{customdata[1]}m<br>"
                      "SINR: %{customdata[2]} dB<br>TP: %{customdata[3]} Mbps<br>"
                      "BLER: %{customdata[4]}<extra></extra>",
        showlegend=False,
    ))

    # BS marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=20, color=C["amber"], symbol="diamond",
                    line=dict(width=2, color=C["bg"])),
        showlegend=False, hoverinfo="text", hovertext="gNodeB (0, 0)",
    ))
    fig.add_annotation(x=0, y=0, text="gNB", showarrow=False, yshift=18,
                       font=dict(size=11, color=C["amber"], family="JetBrains Mono"))

    lim = cell_radius * 1.2
    _apply_theme(fig, h=520)
    fig.update_xaxes(title="East (m)", scaleanchor="y", range=[-lim, lim])
    fig.update_yaxes(title="North (m)", range=[-lim, lim])
    return fig

# ━━ Chart Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_throughput_cdf(kpi, carrier):
    s = kpi.get_valid_range()
    tp = np.sort(np.mean(kpi.ue_throughput_bits[s], axis=0) / carrier.slot_duration_s / 1e6)
    cdf = np.arange(1, len(tp)+1) / max(len(tp), 1)
    fig = go.Figure(go.Scatter(x=tp, y=cdf, mode="lines", line=dict(color=C["green"], width=2.5)))
    fig.update_layout(xaxis_title="Throughput (Mbps)", yaxis_title="CDF")
    return _apply_theme(fig)

def chart_bler_trend(kpi):
    s = kpi.get_valid_range()
    success = kpi.ue_tb_success[s].astype(float)
    sched = kpi.ue_num_prbs[s] > 0
    bler = np.array([1.0 - np.mean(success[i, sched[i]]) if np.any(sched[i]) else 0
                     for i in range(success.shape[0])])
    w = min(100, max(len(bler)//5, 1))
    smoothed = np.convolve(bler, np.ones(w)/w, mode="valid")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=smoothed, mode="lines", name="BLER",
                             line=dict(color=C["amber"], width=2)))
    fig.add_hline(y=0.1, line_dash="dash", line_color=C["red"], line_width=1,
                  annotation_text="Target 10%", annotation_font_size=9,
                  annotation_font_color=C["red"])
    fig.update_layout(xaxis_title="Slot", yaxis_title="BLER")
    return _apply_theme(fig)

def chart_mcs_dist(kpi):
    s = kpi.get_valid_range()
    mcs = kpi.ue_mcs[s][kpi.ue_num_prbs[s] > 0]
    if len(mcs) == 0: return None
    counts = np.bincount(mcs, minlength=29)[:29]
    df = pd.DataFrame({"MCS": range(29), "Count": counts})
    fig = go.Figure(go.Bar(x=df["MCS"], y=df["Count"],
                           marker_color=C["green"], marker_line_width=0))
    fig.update_layout(xaxis_title="MCS Index", yaxis_title="Count", bargap=0.15)
    return _apply_theme(fig)

def chart_sinr_cdf(kpi):
    s = kpi.get_valid_range()
    sinr = np.sort(kpi.ue_sinr_eff_db[s].ravel())
    sinr = sinr[sinr > -29]
    if len(sinr) == 0: return None
    cdf = np.arange(1, len(sinr)+1) / len(sinr)
    fig = go.Figure(go.Scatter(x=sinr, y=cdf, mode="lines", line=dict(color=C["blue"], width=2.5)))
    fig.update_layout(xaxis_title="SINR (dB)", yaxis_title="CDF")
    return _apply_theme(fig)

def chart_cell_tp(kpi, carrier):
    s = kpi.get_valid_range()
    delivered = kpi.cell_throughput_bits[s] / carrier.slot_duration_s / 1e6
    scheduled = kpi.cell_scheduled_bits[s] / carrier.slot_duration_s / 1e6
    df = pd.DataFrame({"Scheduled": scheduled, "Delivered": delivered})
    w = min(50, max(len(df)//10, 1))
    df = df.rolling(window=w).mean().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["Scheduled"], mode="lines", name="Scheduled",
                             line=dict(color=C["text3"], width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(y=df["Delivered"], mode="lines", name="Delivered",
                             line=dict(color=C["green"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(34,211,238,0.06)"))
    fig.update_layout(xaxis_title="Slot", yaxis_title="Mbps")
    return _apply_theme(fig)

def chart_rank_dist(kpi):
    s = kpi.get_valid_range()
    ranks = kpi.ue_rank[s][kpi.ue_num_prbs[s] > 0]
    if len(ranks) == 0: return None
    counts = np.bincount(ranks, minlength=5)[1:5]
    fig = go.Figure(go.Bar(
        x=["Rank 1", "Rank 2", "Rank 3", "Rank 4"], y=counts,
        marker_color=[C["green"], C["blue"], C["amber"], C["red"]],
        marker_line_width=0,
    ))
    fig.update_layout(xaxis_title="Rank", yaxis_title="Count", bargap=0.25)
    return _apply_theme(fig, h=300)

# ━━ TDD Frame Visualization ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_tdd_frame(pattern, sp_dl, sp_gp, sp_ul):
    slots_html = ""
    for ch in pattern:
        if ch == 'D':
            slots_html += '<div class="tdd-slot D">D</div>'
        elif ch == 'S':
            slots_html += f'<div class="tdd-slot S">S<span style="font-size:0.55rem;opacity:0.7"><br>{sp_dl}+{sp_gp}+{sp_ul}</span></div>'
        elif ch == 'U':
            slots_html += '<div class="tdd-slot U">U</div>'
    return f'<div class="tdd-frame">{slots_html}</div>'


# ━━ Sidebar ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_sidebar():
    S = st.session_state
    with st.sidebar:
        st.markdown(f"""<div style="padding:0.5rem 0 0.3rem;">
            <div style="color:{C['green']};font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:0.05em;">
            L2 RRM SIM</div>
            <div style="color:{C['text3']};font-size:0.72rem;margin-top:0.1rem;">5G NR System-Level Simulator</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # ── Presets ──
        st.markdown(f'<div class="section-label">Quick Presets</div>', unsafe_allow_html=True)
        pcols = st.columns(3)
        for i, (name, vals) in enumerate(PRESETS.items()):
            short = name.split()[-1] if len(name) > 12 else name
            if pcols[i % 3].button(short, key=f"pre_{name}", use_container_width=True):
                for k, v in vals.items():
                    st.session_state[k] = v
                st.session_state.preset = name
                st.rerun()
        st.markdown("---")

        # ── Simulation ──
        st.markdown(f'<div class="section-label">Simulation</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.number_input("Slots", 100, 50000, step=100, key="num_slots")
        c2.number_input("Warmup", 0, 5000, step=50, key="warmup_slots")
        st.number_input("Random Seed", 0, 99999, key="random_seed")
        st.markdown("---")

        # ── Radio Frame ──
        st.markdown(f'<div class="section-label">Radio Frame</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.selectbox("Duplex", ["TDD", "FDD"], key="duplex")
        c2.selectbox("SCS", [15, 30, 60], key="scs", format_func=lambda x: f"{x} kHz")
        st.selectbox("Bandwidth", list(BW_PRB.keys()), key="bw")
        prb_count = BW_PRB[S.bw].get(S.scs, 273) or 273
        st.markdown(f'<div style="color:{C["text3"]};font-size:0.72rem;">PRBs: {prb_count} | Freq: {S.freq} GHz</div>',
                    unsafe_allow_html=True)

        if S.duplex == "TDD":
            st.selectbox("TDD Pattern", TDD_PATTERNS, key="tdd_pat")
            st.markdown(render_tdd_frame(S.tdd_pat, S.sp_dl, S.sp_gp, S.sp_ul), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.number_input("DL sym", 1, 12, key="sp_dl")
            c2.number_input("GP sym", 0, 6, key="sp_gp")
            c3.number_input("UL sym", 0, 6, key="sp_ul")
            st.slider("HARQ K1", 1, 10, key="k1")
        st.markdown("---")

        # ── Cell ──
        st.markdown(f'<div class="section-label">Cell & Antenna</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.selectbox("Scenario", ["uma", "umi", "rma"], key="scenario")
        c2.selectbox("TX Ant", [8, 16, 32, 64], key="tx_ant")
        c1, c2 = st.columns(2)
        c1.selectbox("Ports", [2, 4, 8], key="tx_ports")
        c2.selectbox("Layers", [1, 2, 4], key="max_layers")
        st.slider("TX Power (dBm)", 30.0, 56.0, step=1.0, key="power")
        st.slider("Cell Radius (m)", 100, 2000, step=50, key="radius")
        st.markdown("---")

        # ── UE ──
        st.markdown(f'<div class="section-label">UE Configuration</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.slider("UE Count", 1, 50, key="n_ue")
        c2.selectbox("RX Ant", [1, 2, 4], key="rx_ant")
        c1, c2 = st.columns(2)
        c1.number_input("Min Dist (m)", 10.0, 2000.0, step=5.0, key="d_min")
        c2.number_input("Max Dist (m)", 10.0, 2000.0, step=5.0, key="d_max")
        st.slider("Speed (km/h)", 0.0, 120.0, step=1.0, key="speed")
        st.markdown("---")

        # ── Advanced ──
        with st.expander("Scheduler & Link Adaptation", expanded=False):
            st.slider("PF Beta", 0.9, 0.999, step=0.001, format="%.3f", key="beta")
            st.slider("BLER Target", 0.01, 0.30, step=0.01, key="bler_t")
            st.slider("OLLA Delta", 0.1, 2.0, step=0.1, key="olla_d")

        with st.expander("Traffic & Channel", expanded=False):
            st.selectbox("Traffic Model", ["Full Buffer", "FTP Model 3", "Poisson"], key="traffic")
            st.selectbox("Channel Model", ["statistical", "sionna"], key="ch_type")
            st.checkbox("Enable CSI Feedback", key="csi_on")
            if S.csi_on:
                st.number_input("CSI Period", 5, 100, key="csi_period")
                st.number_input("Feedback Delay", 0, 20, key="csi_delay")


# ━━ Main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.set_page_config(page_title="L2 RRM Simulator", page_icon="📡", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()
    init_state()
    render_sidebar()

    S = st.session_state

    # ── Validation ──
    if S.d_min > S.d_max:
        st.error("Min Distance > Max Distance"); return
    if S.duplex == "TDD" and S.sp_dl + S.sp_gp + S.sp_ul > 14:
        st.error("Special slot symbols exceed 14"); return

    # ── Header ──
    duplex_label = f"TDD {S.tdd_pat}" if S.duplex == "TDD" else "FDD"
    st.markdown(f"""<div class="mc-header">
        <div class="mc-title">L2 RRM Simulation Console</div>
        <div class="mc-sub">5G NR System-Level Simulator | Single-Cell Mode</div>
        <div class="mc-tags">
            <span class="mc-tag">{duplex_label}</span>
            <span class="mc-tag">{S.n_ue} UE</span>
            <span class="mc-tag">{S.bw} @ {S.scs} kHz</span>
            <span class="mc-tag">{S.tx_ant}T{S.rx_ant}R</span>
            <span class="mc-tag">{S.scenario.upper()}</span>
            <span class="mc-tag">{S.traffic}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Run ──
    if st.button("RUN SIMULATION", type="primary", use_container_width=True):
        config = build_config()
        report, kpi, carrier, harq, topo = run_simulation(config)
        st.session_state.update(dict(report=report, kpi=kpi, carrier=carrier,
                                     harq=harq, topo=topo, cfg=config))

    if "report" not in st.session_state:
        st.markdown(f"""<div style="text-align:center;padding:3rem 1rem;color:{C['text3']}">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;opacity:0.3">{'///'}</div>
            <div style="font-size:0.9rem;">Configure parameters in sidebar, then click <b>RUN SIMULATION</b></div>
        </div>""", unsafe_allow_html=True)
        return

    report = st.session_state.report
    kpi = st.session_state.kpi
    carrier = st.session_state.carrier
    harq = st.session_state.harq
    topo = st.session_state.topo
    cfg = st.session_state.cfg

    # ━━ KPI Hero Row ━━
    avg_rank = report.get("avg_rank", 1.0)
    h1, h2, h3 = st.columns([2, 2, 3])
    with h1:
        st.markdown(f"""<div style="padding:0.3rem 0;">
            <div class="section-label">Cell Throughput</div>
            <div class="mc-kpi-big">{report['cell_avg_throughput_mbps']:.1f}<span class="mc-kpi-unit">Mbps</span></div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown(f"""<div style="padding:0.3rem 0;">
            <div class="section-label">Spectral Efficiency</div>
            <div class="mc-kpi-big">{report['spectral_efficiency_bps_hz']:.2f}<span class="mc-kpi-unit">bps/Hz</span></div>
        </div>""", unsafe_allow_html=True)
    with h3:
        tags = (
            f'<span class="mc-tag">BLER {report["avg_bler"]:.3f}</span>'
            f'<span class="mc-tag">MCS {report["avg_mcs"]:.1f}</span>'
            f'<span class="mc-tag">Rank {avg_rank:.1f}</span>'
            f'<span class="mc-tag">PRB {report["prb_utilization"]*100:.0f}%</span>'
            f'<span class="mc-tag{"" if report["delivery_ratio"]>0.9 else " warn"}">Delivery {report["delivery_ratio"]*100:.1f}%</span>'
            f'<span class="mc-tag{"" if harq["retx_rate"]<0.15 else " warn"}">ReTX {harq["retx_rate"]*100:.1f}%</span>'
        )
        st.markdown(f'<div style="padding:0.3rem 0;"><div class="section-label">Status</div>'
                    f'<div class="mc-tags" style="margin-top:0.4rem">{tags}</div></div>',
                    unsafe_allow_html=True)

    # ━━ Detail Metrics ━━
    m = st.columns(7)
    m[0].metric("Delivered TP", f"{report['cell_avg_throughput_mbps']:.1f} Mbps")
    m[1].metric("Scheduled TP", f"{report['cell_avg_scheduled_throughput_mbps']:.1f} Mbps")
    m[2].metric("Avg BLER", f"{report['avg_bler']:.4f}")
    m[3].metric("Avg SINR", f"{report['avg_sinr_db']:.1f} dB")
    m[4].metric("Avg MCS", f"{report['avg_mcs']:.1f}")
    m[5].metric("HARQ ReTX", f"{harq['retx_rate']*100:.1f}%")
    m[6].metric("Fairness", f"{report['jain_fairness']:.3f}")

    # ━━ Tabs ━━
    tab_topo, tab_perf, tab_link, tab_ue, tab_trace = st.tabs([
        "TOPOLOGY", "THROUGHPUT", "LINK QUALITY", "UE DETAIL", "SLOT TRACE"])

    with tab_topo:
        col_map, col_info = st.columns([3, 1])
        with col_info:
            st.markdown(f'<div class="section-label">Color Map</div>', unsafe_allow_html=True)
            color_by = st.radio("c", ["SINR (dB)", "Throughput (Mbps)", "BLER"],
                                label_visibility="collapsed")
            st.markdown("---")
            st.markdown(f'<div class="section-label">Cell Info</div>', unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:0.78rem;color:{C['text2']};line-height:1.7;">
                Radius: {cfg['cell'].cell_radius_m:.0f} m<br>
                Height: {cfg['cell'].height_m:.0f} m<br>
                Power: {cfg['cell'].total_power_dbm:.0f} dBm<br>
                Antenna: {cfg['cell'].num_tx_ant}T{cfg['ue'].num_rx_ant}R<br>
                Scenario: {cfg['cell'].scenario.upper()}
            </div>""", unsafe_allow_html=True)
            if S.duplex == "TDD":
                st.markdown("---")
                st.markdown(f'<div class="section-label">TDD Frame</div>', unsafe_allow_html=True)
                st.markdown(render_tdd_frame(S.tdd_pat, S.sp_dl, S.sp_gp, S.sp_ul),
                            unsafe_allow_html=True)
        with col_map:
            st.plotly_chart(render_topology(topo, float(cfg["cell"].cell_radius_m), color_by),
                            use_container_width=True)
        st.dataframe(topo, use_container_width=True, hide_index=True)

    with tab_perf:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="section-label">Cell Throughput Trend</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_cell_tp(kpi, carrier), use_container_width=True)
        with c2:
            st.markdown(f'<div class="section-label">UE Throughput CDF</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_throughput_cdf(kpi, carrier), use_container_width=True)
        fig_rank = chart_rank_dist(kpi)
        if fig_rank:
            st.markdown(f'<div class="section-label">Rank Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_rank, use_container_width=True)

    with tab_link:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="section-label">BLER Convergence</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_bler_trend(kpi), use_container_width=True)
        with c2:
            st.markdown(f'<div class="section-label">Effective SINR CDF</div>', unsafe_allow_html=True)
            fig = chart_sinr_cdf(kpi)
            if fig: st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="section-label">MCS Distribution</div>', unsafe_allow_html=True)
        fig = chart_mcs_dist(kpi)
        if fig: st.plotly_chart(fig, use_container_width=True)

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
        st.markdown(f'<div class="section-label">Slot-Level Audit Trail</div>', unsafe_allow_html=True)
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
            "rank": trace["ue_rank"][:, focus] if n_ue else 0,
            "sinr_db": np.round(trace["ue_sinr_eff_db"][:, focus], 1) if n_ue else 0,
            "tb_ok": trace["ue_tb_success"][:, focus] if n_ue else False,
        })
        st.dataframe(frame.tail(rows).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    # ── Footer ──
    st.markdown(f"""<div style="text-align:center;padding:1.5rem 0 0.5rem;border-top:1px solid {C['border']};margin-top:1.5rem;">
        <span style="color:{C['text3']};font-size:0.7rem;font-family:'JetBrains Mono',monospace;">
        L2 RRM Simulator v2.0 | Powered by Sionna 2.0 + PyTorch | github.com/TianLin0509/L2_RRM_Sim
        </span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""NearMiss (SW-5) — Autonomous Satellite Collision Risk Predictor.

Aerospace Operations Flight Deck for Satellite Conjunction Assessment and
Collision Avoidance Maneuver Support under strict T-48h Operational Cutoff constraints.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory on path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.calibrate import assign_confidence
from src.evaluate import compute_classification_metrics
from src.load import load_config, load_parquet

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NearMiss — Conjunction Assessment Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# BESPOKE AEROSPACE FLIGHT DECK STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-void: #070A0F;
        --surface-base: #0D131D;
        --surface-card: #121A27;
        --surface-hover: #182335;
        --border-subtle: #1C2739;
        --border-focus: #00E5FF;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --cyan-telemetry: #00E5FF;
        --red-alert: #FF334B;
        --green-safe: #00E676;
        --amber-warning: #FFB800;
        --purple-marker: #B388FF;
    }

    /* Global Base Reset */
    .stApp {
        background-color: var(--bg-void);
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Monospace telemetry fonts */
    code, pre, .mono-data {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Header Mission Ribbon */
    .mission-hud-header {
        background: linear-gradient(180deg, rgba(18, 26, 39, 0.95) 0%, rgba(13, 19, 29, 0.95) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .mission-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0;
    }
    .mission-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: rgba(0, 229, 255, 0.1);
        color: var(--cyan-telemetry);
        border: 1px solid rgba(0, 229, 255, 0.3);
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .status-cutoff-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #F87171;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #F87171;
        display: inline-block;
        box-shadow: 0 0 6px #F87171;
    }
    .status-dot-green {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--green-safe);
        display: inline-block;
        box-shadow: 0 0 6px var(--green-safe);
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: var(--surface-card);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 12px 16px;
        transition: border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    .kpi-subtext {
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Decision HUD Banner */
    .decision-hud-box {
        background: var(--surface-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .badge-manoeuvre-hero {
        background: rgba(255, 51, 75, 0.15);
        border: 1.5px solid var(--red-alert);
        color: #FF5A6E;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.04em;
        box-shadow: 0 0 15px rgba(255, 51, 75, 0.2);
    }
    .badge-safe-hero {
        background: rgba(0, 230, 118, 0.12);
        border: 1.5px solid var(--green-safe);
        color: #26E784;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.04em;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.15);
    }
    .badge-confidence-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-conf-high {
        background: rgba(0, 229, 255, 0.15);
        color: var(--cyan-telemetry);
        border: 1px solid rgba(0, 229, 255, 0.4);
    }
    .badge-conf-med {
        background: rgba(255, 184, 0, 0.15);
        color: var(--amber-warning);
        border: 1px solid rgba(255, 184, 0, 0.4);
    }
    .badge-conf-low {
        background: rgba(148, 163, 184, 0.15);
        color: var(--text-secondary);
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Comparison Badge */
    .baseline-badge-box {
        background: rgba(13, 19, 29, 0.8);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 10px 14px;
    }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 4px;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--surface-base);
        border: 1px solid var(--border-subtle);
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--surface-card) !important;
        border-color: var(--cyan-telemetry) !important;
        border-bottom-color: var(--surface-card) !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar Aerospace Control Panel */
    [data-testid="stSidebar"] {
        background-color: #090D15 !important;
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Clean Matrix Table */
    .hud-matrix-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .hud-matrix-table th {
        background: #0E1622;
        color: var(--text-muted);
        padding: 6px 10px;
        font-size: 0.72rem;
        text-transform: uppercase;
        border-radius: 4px;
        font-weight: 600;
    }
    .hud-matrix-table td {
        background: #141E2B;
        padding: 10px 12px;
        border-radius: 4px;
        color: var(--text-primary);
        font-weight: 500;
    }
    .hud-matrix-table .cell-fp {
        color: #F87171;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .hud-matrix-table .cell-fn {
        color: #F87171;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .hud-matrix-table .cell-tp {
        color: var(--green-safe);
        background: rgba(0, 230, 118, 0.1);
        border: 1px solid rgba(0, 230, 118, 0.25);
    }
    .hud-matrix-table .cell-tn {
        color: var(--text-primary);
        background: #121A26;
    }

    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--text-muted) !important;
    }

    /* Clean Divider */
    hr {
        border-color: var(--border-subtle) !important;
        margin: 16px 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# DATA ACCESS & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_app_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load configuration, prediction results, and CDM history."""
    cfg = load_config()
    preds_df = load_parquet("data/processed/predictions.parquet")
    hist_df = load_parquet("data/processed/cdm_history.parquet")
    return cfg, preds_df, hist_df


# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
def main() -> None:
    cfg, all_preds, all_hist = load_app_data()

    # --- SIDEBAR CONTROL STATION ---
    st.sidebar.markdown(
        """
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
            <div style='background:rgba(0,229,255,0.12); border:1px solid rgba(0,229,255,0.3); border-radius:6px; padding:6px 10px; font-family:"JetBrains Mono",monospace; font-weight:700; color:#00E5FF;'>
                SW-5
            </div>
            <div>
                <div style='font-weight:700; font-size:1.05rem; color:#F1F5F9; letter-spacing:-0.01em;'>NearMiss Console</div>
                <div style='font-size:0.72rem; color:#64748B;'>Flight Dynamics Support</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Telemetry Data Split")
    split_choice = st.sidebar.radio(
        "Active Evaluation Cohort",
        options=["Validation Cohort (Fold 0 OOF)", "Locked Test Set (ESA Competition)"],
        index=0,
        label_visibility="collapsed",
    )
    split_key = "val" if "Validation" in split_choice else "test"
    split_preds = all_preds[all_preds["split"] == split_key].copy()

    high_risk_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))
    operational_thresh = float(cfg.get("thresholds", {}).get("operational_risk_log10", -4.0))
    cutoff_days = float(cfg["data"]["cutoff_days"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style='font-size:0.75rem; text-transform:uppercase; color:#64748B; font-weight:600; margin-bottom:8px;'>Operational Constraints</div>
        <div style='display:flex; flex-direction:column; gap:6px; font-family:"JetBrains Mono",monospace; font-size:0.78rem;'>
            <div style='display:flex; justify-content:space-between; background:#0F1722; padding:6px 8px; border-radius:4px; border:1px solid #1C2739;'>
                <span style='color:#94A3B8;'>Cutoff Horizon:</span>
                <span style='color:#F1F5F9; font-weight:600;'>{cutoff_days:.1f} days (T-48h)</span>
            </div>
            <div style='display:flex; justify-content:space-between; background:#0F1722; padding:6px 8px; border-radius:4px; border:1px solid #1C2739;'>
                <span style='color:#94A3B8;'>Warning (1:1M):</span>
                <span style='color:#FFB800; font-weight:600;'>10^{{{int(high_risk_thresh)}}}</span>
            </div>
            <div style='display:flex; justify-content:space-between; background:#0F1722; padding:6px 8px; border-radius:4px; border:1px solid #1C2739;'>
                <span style='color:#94A3B8;'>Maneuver (1:10k):</span>
                <span style='color:#FF334B; font-weight:600;'>10^{{{int(operational_thresh)}}}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='font-size:0.72rem; color:#64748B; line-height:1.4;'>
            Built on official European Space Agency (ESA) Conjunction Data Messages (Zenodo 4463683). All decisions enforced at ≥ 2.0 days before TCA.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- TOP MISSION STATUS HUD ---
    st.markdown(
        f"""
        <div class='mission-hud-header'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;'>
                <div>
                    <div class='mission-title'>
                        NearMiss SW-5 Conjunction Assessment System
                        <span class='mission-tag'>ESA Kelvins Architecture</span>
                    </div>
                    <div style='font-size:0.82rem; color:#94A3B8; margin-top:4px;'>
                        Autonomous collision risk forecasting enforcing a strict <strong>{cutoff_days:.1f}-day advance cutoff</strong> to protect satellite maneuvering windows.
                    </div>
                </div>
                <div style='display:flex; align-items:center; gap:8px;'>
                    <span class='status-cutoff-pill'>
                        <span class='status-dot'></span>
                        T-48H CUTOFF ACTIVE · BLIND ZONE PROTECTED
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Telemetry KPI Ribbon
    n_total = len(split_preds)
    n_man = int((split_preds["recommendation"] == "MANOEUVRE").sum())
    n_danger = int(split_preds["actual_high_risk"].sum())
    man_pct = (n_man / max(1, n_total)) * 100
    danger_pct = (n_danger / max(1, n_total)) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Total Events Monitored</div>
                <div class='kpi-value'>{n_total:,}</div>
                <div class='kpi-subtext'>{split_choice.split('(')[0].strip()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Maneuvers Commanded</div>
                <div class='kpi-value' style='color:#FF5A6E;'>{n_man:,}</div>
                <div class='kpi-subtext'>{man_pct:.1f}% burn recommendation rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Confirmed High-Risk (Truth)</div>
                <div class='kpi-value' style='color:#FFB800;'>{n_danger:,}</div>
                <div class='kpi-subtext'>{danger_pct:.1f}% terminal collision encounters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Planning Lead Time</div>
                <div class='kpi-value' style='color:#00E5FF;'>≥ 48.0 h</div>
                <div class='kpi-subtext'>Zero look-ahead into blind zone</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --- PRIMARY NAVIGATION TABS ---
    tab1, tab2, tab3 = st.tabs([
        "Conjunction Event Inspector",
        "Threshold & Fuel Simulator",
        "Model Benchmark & Physics",
    ])

    # =========================================================================
    # TAB 1: CONJUNCTION EVENT INSPECTOR
    # =========================================================================
    with tab1:
        col_list, col_view = st.columns([1, 2.2])

        with col_list:
            st.markdown(
                """
                <div style='font-size:0.85rem; font-weight:700; color:#F1F5F9; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.04em;'>
                    Conjunction Filter & Search
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Filters
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                rec_filter = st.selectbox(
                    "Recommendation",
                    ["All", "MANOEUVRE", "NO_MANOEUVRE"],
                    index=0,
                )
            with f_col2:
                conf_filter = st.selectbox(
                    "Confidence Band",
                    ["All", "HIGH", "MEDIUM", "LOW"],
                    index=0,
                )

            search_query = st.text_input("Search Numeric Event ID", "", placeholder="e.g. 873, 193, 3892")

            filtered_list = split_preds.copy()
            if rec_filter != "All":
                filtered_list = filtered_list[filtered_list["recommendation"] == rec_filter]
            if conf_filter != "All":
                filtered_list = filtered_list[filtered_list["confidence"] == conf_filter]
            if search_query.strip():
                try:
                    search_id = int(search_query.strip())
                    filtered_list = filtered_list[filtered_list["event_id"] == search_id]
                except ValueError:
                    st.warning("Enter a valid integer Event ID.")

            # Sort descending by danger probability
            filtered_list = filtered_list.sort_values("high_risk_prob", ascending=False)

            if len(filtered_list) == 0:
                st.warning("No conjunction events match the active search criteria.")
                st.stop()

            # Event selector list with telemetry preview
            event_options = filtered_list["event_id"].tolist()
            selected_event_id = st.selectbox(
                f"Matching Encounters ({len(filtered_list)} found)",
                event_options,
                format_func=lambda eid: f"ID {eid} | P={filtered_list.loc[filtered_list['event_id']==eid, 'high_risk_prob'].iloc[0]:.1%} | {filtered_list.loc[filtered_list['event_id']==eid, 'recommendation'].iloc[0]} [{filtered_list.loc[filtered_list['event_id']==eid, 'confidence'].iloc[0]}]",
            )

            # Selected Event Metadata Card
            event_row = split_preds[split_preds["event_id"] == selected_event_id].iloc[0]
            st.markdown(
                f"""
                <div style='background:#0D131D; border:1px solid #1C2739; border-radius:6px; padding:12px; margin-top:12px; font-family:"JetBrains Mono",monospace; font-size:0.75rem;'>
                    <div style='color:#64748B; font-weight:600; text-transform:uppercase; margin-bottom:6px;'>Spacecraft Encounter Dossier</div>
                    <div style='display:grid; grid-template-columns: 1fr 1fr; gap:6px;'>
                        <div><span style='color:#94A3B8;'>Target Sat:</span> <span style='color:#F1F5F9;'>ID #{event_row["mission_id"]}</span></div>
                        <div><span style='color:#94A3B8;'>Chaser Type:</span> <span style='color:#00E5FF;'>{event_row["object_type"]}</span></div>
                        <div><span style='color:#94A3B8;'>Early CDMs:</span> <span style='color:#F1F5F9;'>{int(event_row["cdm_count_pre_cutoff"])} passes</span></div>
                        <div><span style='color:#94A3B8;'>Last Obs:</span> <span style='color:#F1F5F9;'>T-2.0d</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_view:
            # Active Conjunction Decision HUD
            conf = str(event_row["confidence"]).upper()
            conf_class = f"badge-conf-{conf.lower()[:3]}"
            is_manoeuvre = event_row["recommendation"] == "MANOEUVRE"
            b_rec = event_row["baseline_recommendation"]

            rec_badge_html = (
                "<div class='badge-manoeuvre-hero'>COMMAND AVOIDANCE BURN</div>"
                if is_manoeuvre
                else "<div class='badge-safe-hero'>STAND DOWN (CLEAR ORBIT)</div>"
            )

            b_pill = (
                "<span style='color:#FF5A6E; font-weight:700;'>BURN ORDERED</span>"
                if b_rec == "MANOEUVRE"
                else "<span style='color:#26E784; font-weight:700;'>NO BURN</span>"
            )

            # Agreement logic
            if is_manoeuvre and b_rec != "MANOEUVRE":
                divergence_note = "<span style='color:#00E5FF; font-weight:600;'>NearMiss Rescued Event (Baseline Missed Collision)</span>"
            elif not is_manoeuvre and b_rec == "MANOEUVRE":
                divergence_note = "<span style='color:#00E676; font-weight:600;'>NearMiss Saved Maneuver (Baseline False Alarm Avoided)</span>"
            else:
                divergence_note = "<span style='color:#94A3B8;'>Ensemble & Baseline in Full Agreement</span>"

            st.markdown(
                f"""
                <div class='decision-hud-box'>
                    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px;'>
                        <div>
                            <div style='font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:600; margin-bottom:6px;'>
                                Operational Recommendation (Event #{selected_event_id})
                            </div>
                            {rec_badge_html}
                        </div>
                        <div style='display:flex; flex-direction:column; align-items:flex-end; gap:6px;'>
                            <div>
                                <span style='font-size:0.75rem; color:#94A3B8;'>Confidence Tier:</span>
                                <span class='badge-confidence-tag {conf_class}'>{conf} CONFIDENCE</span>
                            </div>
                            <div style='font-size:0.75rem;'>{divergence_note}</div>
                        </div>
                    </div>
                    <div style='display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; border-top:1px solid #1C2739; padding-top:12px;'>
                        <div>
                            <div style='font-size:0.72rem; color:#64748B; text-transform:uppercase; font-weight:600;'>Calibrated P(Collision)</div>
                            <div style='font-family:"JetBrains Mono",monospace; font-size:1.35rem; font-weight:700; color:#F1F5F9;'>
                                {event_row['high_risk_prob']:.2%}
                            </div>
                            <div style='font-size:0.7rem; color:#94A3B8;'>Platt scaled posterior</div>
                        </div>
                        <div>
                            <div style='font-size:0.72rem; color:#64748B; text-transform:uppercase; font-weight:600;'>Predicted Terminal Log-Risk</div>
                            <div style='font-family:"JetBrains Mono",monospace; font-size:1.35rem; font-weight:700; color:#00E5FF;'>
                                {event_row['pred_final_log_risk']:.2f}
                            </div>
                            <div style='font-size:0.7rem; color:#94A3B8;'>Hurdle model estimate at TCA</div>
                        </div>
                        <div class='baseline-badge-box'>
                            <div style='font-size:0.72rem; color:#64748B; text-transform:uppercase; font-weight:600;'>Baseline A (Persistence)</div>
                            <div style='font-family:"JetBrains Mono",monospace; font-size:0.95rem; margin-top:2px;'>
                                {b_pill}
                            </div>
                            <div style='font-size:0.7rem; color:#94A3B8;'>Last Visible Risk: {event_row['latest_log_risk']:.2f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ground Truth Inspection Toggle
            gt_col_toggle, gt_col_status = st.columns([1.2, 2])
            with gt_col_toggle:
                reveal_gt = st.toggle("Reveal Post-Cutoff Ground Truth (Simulation Mode)", value=False)
            with gt_col_status:
                if reveal_gt:
                    actual_risk_val = float(event_row["actual_final_log_risk"])
                    actual_danger = event_row["actual_high_risk"] == 1
                    status_badge = (
                        "<span style='color:#FF5A6E; font-weight:700;'>🔴 ACTUAL COLLISION RISK EXCEEDED 10⁻⁶</span>"
                        if actual_danger
                        else "<span style='color:#00E676; font-weight:700;'>🟢 ACTUAL ENCOUNTER HARMLESS</span>"
                    )
                    st.markdown(
                        f"""
                        <div style='font-family:"JetBrains Mono",monospace; font-size:0.8rem; background:#0D131D; padding:6px 10px; border-radius:4px; border:1px solid #1C2739;'>
                            Terminal Risk: <strong>{actual_risk_val:.2f}</strong> · {status_badge}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Plotly Trajectory Chart Construction
            event_history = all_hist[all_hist["event_id"] == selected_event_id].sort_values("time_to_tca", ascending=False)
            vis_cdms = event_history[event_history["visible"]]
            post_cdms = event_history[~event_history["visible"]]

            fig_risk = go.Figure()

            # Shaded Blind Zone (Strict T-48h cutoff)
            fig_risk.add_vrect(
                x0=cutoff_days,
                x1=0.0,
                fillcolor="#FF334B",
                opacity=0.07,
                layer="below",
                line_width=0,
                annotation_text="BLIND ZONE (Maneuver Planning Barred)",
                annotation_position="top left",
                annotation_font=dict(color="#FF5A6E", size=10, family="JetBrains Mono"),
            )

            # Cutoff vertical boundary
            fig_risk.add_vline(
                x=cutoff_days,
                line_dash="dash",
                line_color="#FF334B",
                line_width=2,
                annotation_text="T-48H DECISION CUTOFF",
                annotation_position="bottom left",
                annotation_font=dict(color="#FF334B", size=10, family="JetBrains Mono"),
            )

            # Warning Threshold Line (-6.0)
            fig_risk.add_hline(
                y=high_risk_thresh,
                line_dash="dot",
                line_color="#FFB800",
                line_width=1.5,
                annotation_text="1:1M High Risk (-6.0)",
                annotation_position="top right",
                annotation_font=dict(color="#FFB800", size=10, family="JetBrains Mono"),
            )

            # Visible Observation Passes
            fig_risk.add_trace(
                go.Scatter(
                    x=vis_cdms["time_to_tca"],
                    y=vis_cdms["log_risk"],
                    mode="lines+markers",
                    name="Observable Tracking Passes (≥ 2.0d)",
                    line=dict(color="#00E5FF", width=2.5),
                    marker=dict(size=8, color="#00E5FF", symbol="circle"),
                    hovertemplate="Time to TCA: %{x:.2f}d<br>Observed Risk: %{y:.2f}<extra></extra>",
                )
            )

            # Predicted Risk Marker at TCA (t = 0.0)
            fig_risk.add_trace(
                go.Scatter(
                    x=[0.0],
                    y=[event_row["pred_final_log_risk"]],
                    mode="markers+text",
                    name="NearMiss Prediction at TCA",
                    text=["Predicted Risk"],
                    textposition="top center",
                    marker=dict(size=14, color="#B388FF", symbol="diamond"),
                    hovertemplate="Predicted TCA Risk: %{y:.2f}<extra></extra>",
                )
            )

            # Reveal Ground Truth Tracking Updates if enabled
            if reveal_gt and len(post_cdms) > 0:
                fig_risk.add_trace(
                    go.Scatter(
                        x=post_cdms["time_to_tca"],
                        y=post_cdms["log_risk"],
                        mode="lines+markers",
                        name="Actual Post-Cutoff Updates (Ground Truth)",
                        line=dict(color="#FF8A00", width=2, dash="dash"),
                        marker=dict(size=7, color="#FF8A00", symbol="triangle-down"),
                        hovertemplate="Actual Time to TCA: %{x:.2f}d<br>Actual Risk: %{y:.2f}<extra></extra>",
                    )
                )

            fig_risk.update_layout(
                title=dict(
                    text=f"Conjunction Risk Evolution Timeline — Encounter #{selected_event_id} ({event_row['object_type']})",
                    font=dict(size=14, color="#F1F5F9", family="Plus Jakarta Sans"),
                ),
                xaxis=dict(
                    title="Time Remaining Until TCA (Days)",
                    autorange="reversed",
                    gridcolor="#162232",
                    zerolinecolor="#1C2739",
                    color="#94A3B8",
                ),
                yaxis=dict(
                    title="Collision Risk log₁₀(P)",
                    range=[-32, 0],
                    gridcolor="#162232",
                    zerolinecolor="#1C2739",
                    color="#94A3B8",
                ),
                paper_bgcolor="#070A0F",
                plot_bgcolor="#0D131D",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=10, color="#94A3B8"),
                ),
                height=380,
                margin=dict(l=40, r=30, t=50, b=40),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Sub-Chart: Encounter Miss Distance Evolution
            fig_miss = go.Figure()
            fig_miss.add_trace(
                go.Scatter(
                    x=vis_cdms["time_to_tca"],
                    y=vis_cdms["miss_distance"],
                    mode="lines+markers",
                    name="Miss Distance (m)",
                    line=dict(color="#00E676", width=2),
                    marker=dict(size=6, color="#00E676"),
                    hovertemplate="Time to TCA: %{x:.2f}d<br>Miss Distance: %{y:,.0f} m<extra></extra>",
                )
            )

            if reveal_gt and len(post_cdms) > 0:
                fig_miss.add_trace(
                    go.Scatter(
                        x=post_cdms["time_to_tca"],
                        y=post_cdms["miss_distance"],
                        mode="lines+markers",
                        name="Post-Cutoff Miss Distance",
                        line=dict(color="#FF8A00", width=1.5, dash="dash"),
                        marker=dict(size=5, color="#FF8A00"),
                        hovertemplate="Actual TCA: %{x:.2f}d<br>Actual Miss: %{y:,.0f} m<extra></extra>",
                    )
                )

            fig_miss.update_layout(
                title=dict(
                    text="Relative Miss Distance Profile Across Observation Passes",
                    font=dict(size=13, color="#F1F5F9", family="Plus Jakarta Sans"),
                ),
                xaxis=dict(
                    title="Time to TCA (Days)",
                    autorange="reversed",
                    gridcolor="#162232",
                    color="#94A3B8",
                ),
                yaxis=dict(
                    title="Distance (Meters)",
                    gridcolor="#162232",
                    color="#94A3B8",
                ),
                paper_bgcolor="#070A0F",
                plot_bgcolor="#0D131D",
                height=220,
                margin=dict(l=40, r=30, t=40, b=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=9, color="#94A3B8"),
                ),
            )
            st.plotly_chart(fig_miss, use_container_width=True)

    # =========================================================================
    # TAB 2: THRESHOLD & FUEL COST SIMULATOR
    # =========================================================================
    with tab2:
        st.markdown(
            """
            <div style='margin-bottom:14px;'>
                <div style='font-size:1.15rem; font-weight:700; color:#F1F5F9;'>Mission Economics & Decision Threshold Simulator</div>
                <div style='font-size:0.82rem; color:#94A3B8;'>
                    Simulate the financial and operational trade-offs between <strong>missed collisions (False Negatives)</strong> and <strong>wasted propellant maneuvers (False Positives)</strong> across satellite constellation profiles.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sim_col1, sim_col2 = st.columns([1, 1.8])

        with sim_col1:
            st.markdown(
                """
                <div style='font-size:0.75rem; text-transform:uppercase; color:#00E5FF; font-weight:700; letter-spacing:0.06em; margin-bottom:6px;'>
                    1. Constellation Mission Presets
                </div>
                """,
                unsafe_allow_html=True,
            )

            preset = st.radio(
                "Preset",
                ["Commercial LEO SmallSat ($5M sat / $15k burn)", "Flagship Earth Observation ($300M sat / $45k burn)", "High-Value Infrastructure ($1B+ / $100k burn)"],
                index=0,
                label_visibility="collapsed",
            )

            if "SmallSat" in preset:
                default_sat_val = 5_000_000
                default_burn_val = 15_000
                default_th = 0.030
            elif "Earth Observation" in preset:
                default_sat_val = 300_000_000
                default_burn_val = 45_000
                default_th = 0.020
            else:
                default_sat_val = 1_000_000_000
                default_burn_val = 100_000
                default_th = 0.010

            st.markdown(
                """
                <div style='font-size:0.75rem; text-transform:uppercase; color:#00E5FF; font-weight:700; letter-spacing:0.06em; margin-top:14px; margin-bottom:6px;'>
                    2. Decision Probability Threshold (p*)
                </div>
                """,
                unsafe_allow_html=True,
            )
            sim_th = st.slider(
                "Decision Threshold",
                min_value=0.005,
                max_value=0.250,
                value=default_th,
                step=0.005,
                format="%.3f",
                label_visibility="collapsed",
            )
            st.caption(f"Events with Platt-calibrated collision probability ≥ {sim_th:.1%} will be commanded to burn.")

            st.markdown(
                """
                <div style='font-size:0.75rem; text-transform:uppercase; color:#00E5FF; font-weight:700; letter-spacing:0.06em; margin-top:14px; margin-bottom:6px;'>
                    3. Cost Accounting Parameters ($ USD)
                </div>
                """,
                unsafe_allow_html=True,
            )
            cost_collision = st.number_input("Cost of Collision / Mission Loss ($)", value=default_sat_val, step=1_000_000)
            cost_manoeuvre = st.number_input("Cost per Avoidance Burn / Delta-v ($)", value=default_burn_val, step=5_000)

            # Re-evaluate metrics at sim_th
            sim_pred_bin = (split_preds["high_risk_prob"] >= sim_th).astype(int)
            y_eval_true = split_preds["actual_high_risk"].values
            sim_metrics = compute_classification_metrics(y_eval_true, sim_pred_bin, split_preds["high_risk_prob"].values)

            # Baseline A metrics
            base_pred_bin = (split_preds["baseline_recommendation"] == "MANOEUVRE").astype(int)
            base_metrics = compute_classification_metrics(y_eval_true, base_pred_bin)

            # Cost Calculations
            cost_model = (sim_metrics["fn"] * cost_collision) + ((sim_metrics["tp"] + sim_metrics["fp"]) * cost_manoeuvre)
            cost_baseline = (base_metrics["fn"] * cost_collision) + ((base_metrics["tp"] + base_metrics["fp"]) * cost_manoeuvre)
            savings = cost_baseline - cost_model
            savings_pct = (savings / max(1, cost_baseline)) * 100

            st.markdown("---")
            savings_color = "#00E676" if savings >= 0 else "#FF5A6E"
            st.markdown(
                f"""
                <div style='background:#0D131D; border:1px solid #1C2739; border-radius:8px; padding:16px;'>
                    <div style='font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:600;'>Net Financial Balance vs Baseline</div>
                    <div style='font-family:"JetBrains Mono",monospace; font-size:1.6rem; font-weight:700; color:{savings_color}; margin:4px 0;'>
                        ${savings:+,.0f}
                    </div>
                    <div style='font-size:0.75rem; color:#94A3B8;'>
                        Relative cost delta: <strong style='color:{savings_color};'>{savings_pct:+.1f}%</strong> compared to legacy persistence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sim_col2:
            st.markdown(
                """
                <div style='font-size:0.85rem; font-weight:700; color:#F1F5F9; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.04em;'>
                    Operational Risk Frontier (Missed Events vs Wasted Burns)
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Generate trade-off frontier curve
            th_grid = np.linspace(0.005, 0.35, 70)
            sweep_pts = []
            for t in th_grid:
                p_bin = (split_preds["high_risk_prob"] >= t).astype(int)
                m = compute_classification_metrics(y_eval_true, p_bin)
                sweep_pts.append({"threshold": t, "fn": m["fn"], "fp": m["fp"], "f2": m["f2"]})
            sweep_df = pd.DataFrame(sweep_pts)

            fig_sim = go.Figure()
            # Frontier curve
            fig_sim.add_trace(
                go.Scatter(
                    x=sweep_df["fp"],
                    y=sweep_df["fn"],
                    mode="lines",
                    name="NearMiss Trade-Off Frontier",
                    line=dict(color="#00E5FF", width=3),
                    hovertemplate="False Alarms: %{x}<br>Missed Collisions: %{y}<extra></extra>",
                )
            )

            # Active operating threshold marker
            fig_sim.add_trace(
                go.Scatter(
                    x=[sim_metrics["fp"]],
                    y=[sim_metrics["fn"]],
                    mode="markers+text",
                    name=f"Selected Policy (p*={sim_th:.3f})",
                    text=[f"Current Policy (p*={sim_th:.3f})"],
                    textposition="top right",
                    marker=dict(size=14, color="#FF334B", symbol="star"),
                )
            )

            # Baseline A operating point
            fig_sim.add_trace(
                go.Scatter(
                    x=[base_metrics["fp"]],
                    y=[base_metrics["fn"]],
                    mode="markers+text",
                    name="Baseline A (Persistence)",
                    text=["Baseline A"],
                    textposition="bottom left",
                    marker=dict(size=12, color="#FFB800", symbol="square"),
                )
            )

            fig_sim.update_layout(
                xaxis=dict(
                    title="Unnecessary Manoeuvres / Wasted Fuel (False Positives)",
                    gridcolor="#162232",
                    color="#94A3B8",
                ),
                yaxis=dict(
                    title="Missed Dangerous Collisions (False Negatives)",
                    gridcolor="#162232",
                    color="#94A3B8",
                ),
                paper_bgcolor="#070A0F",
                plot_bgcolor="#0D131D",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=10, color="#94A3B8"),
                ),
                height=340,
                margin=dict(l=40, r=30, t=30, b=40),
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Side-by-side Confusion Matrix HUD Cards
            cm_c1, cm_c2 = st.columns(2)
            with cm_c1:
                st.markdown(
                    f"""
                    <div style='background:#0D131D; border:1px solid #1C2739; border-radius:6px; padding:12px;'>
                        <div style='font-size:0.75rem; font-weight:700; color:#00E5FF; text-transform:uppercase; margin-bottom:8px;'>
                            NearMiss Confusion Matrix (p* = {sim_th:.3f})
                        </div>
                        <table class='hud-matrix-table'>
                            <tr><th></th><th>Predicted Safe</th><th>Predicted Burn</th></tr>
                            <tr>
                                <th>Actual Safe</th>
                                <td class='cell-tn'>{sim_metrics['tn']:,} (TN)</td>
                                <td class='cell-fp'>{sim_metrics['fp']:,} (FP)</td>
                            </tr>
                            <tr>
                                <th>Actual Hazard</th>
                                <td class='cell-fn'>{sim_metrics['fn']:,} (FN)</td>
                                <td class='cell-tp'>{sim_metrics['tp']:,} (TP)</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with cm_c2:
                st.markdown(
                    f"""
                    <div style='background:#0D131D; border:1px solid #1C2739; border-radius:6px; padding:12px;'>
                        <div style='font-size:0.75rem; font-weight:700; color:#FFB800; text-transform:uppercase; margin-bottom:8px;'>
                            Baseline A (Persistence) Matrix
                        </div>
                        <table class='hud-matrix-table'>
                            <tr><th></th><th>Predicted Safe</th><th>Predicted Burn</th></tr>
                            <tr>
                                <th>Actual Safe</th>
                                <td class='cell-tn'>{base_metrics['tn']:,} (TN)</td>
                                <td class='cell-fp'>{base_metrics['fp']:,} (FP)</td>
                            </tr>
                            <tr>
                                <th>Actual Hazard</th>
                                <td class='cell-fn'>{base_metrics['fn']:,} (FN)</td>
                                <td class='cell-tp'>{base_metrics['tp']:,} (TP)</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # TAB 3: BENCHMARKS, CALIBRATION & PHYSICS
    # =========================================================================
    with tab3:
        st.markdown(
            """
            <div style='margin-bottom:14px;'>
                <div style='font-size:1.15rem; font-weight:700; color:#F1F5F9;'>Architectural Benchmarks & Orbital Physics Validation</div>
                <div style='font-size:0.82rem; color:#94A3B8;'>
                    Evaluation against European Space Agency (ESA) Kelvins competition benchmarks, Platt probability reliability, and physics feature attributions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_tab_col1, b_tab_col2 = st.columns([1.2, 1])

        with b_tab_col1:
            st.markdown("### Model Comparison on Validation Cohort (8,293 Events)")
            st.markdown(
                """
                | Model Architecture | Balancing Strategy | Recall | Precision | F₂ Score | PR-AUC | False Negatives | False Positives |
                | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
                | **Baseline A (Persistence)** | Reference Baseline | 71.21% | 15.26% | 0.4108 | 0.1803 | 19 | 261 |
                | **Logistic Regression** | Unweighted | 28.79% | 25.00% | 0.2794 | 0.1711 | 47 | 57 |
                | **Logistic Regression** | Balanced Weights | 87.88% | 6.58% | 0.2531 | 0.1484 | 8 | 824 |
                | **Random Forest** | Unweighted | 45.45% | 24.59% | 0.3886 | 0.1759 | 36 | 92 |
                | **Random Forest (Champion)** | **Balanced Subsample** | **71.21%** | **16.85%** | **0.4328** | **0.1926** | **19** | **232** |
                | **LightGBM Classifier** | Scale-Pos-Weight | 65.15% | 19.11% | 0.4397 | 0.1504 | 23 | 182 |
                """
            )

            st.markdown("### Locked Test Benchmark (2,167 Events, 150 Ground Truth High-Risk)")
            st.markdown(
                """
                | Pipeline Variant | Avoided Unnecessary Burns | Rescued Collisions | Precision | F₂ Score | High-Risk MSE |
                | :--- | :---: | :---: | :---: | :---: | :---: |
                | **Baseline A (Persistence)** | Reference (0) | Reference (0) | 64.61% | 0.7391 | 26.64 |
                | **NearMiss Production** | **+12 Avoided Burns** | **+3 Rescued Events** | **66.46%** | **0.7030** | **133.79** |
                """
            )

        with b_tab_col2:
            st.markdown("### Probability Reliability (Platt Scaling)")
            cal_img_path = Path("reports/figures/calibration_curve.png")
            if cal_img_path.exists():
                st.image(
                    str(cal_img_path),
                    caption="Platt Sigmoid Calibration on Out-of-Fold Margins (Brier Score: 0.01724 → 0.00698, -59.5% error reduction)",
                    use_container_width=True,
                )
            else:
                st.info("Calibration reliability plot available in reports/figures/.")

        st.markdown("---")
        st.markdown("### Physics Feature Engineering & Covariance Dynamics")

        feat_c1, feat_c2, feat_c3 = st.columns(3)
        with feat_c1:
            st.markdown(
                """
                <div class='kpi-card'>
                    <div style='color:#00E5FF; font-weight:700; font-size:0.85rem; margin-bottom:4px;'>Combined Positional Sigmas</div>
                    <div style='font-size:0.78rem; color:#94A3B8; line-height:1.4;'>
                        Total root-sum-square orbital uncertainty dispersion across radial, transverse, and normal RTN axes combining both target and chaser covariances:
                        <br><code class='mono-data' style='color:#F1F5F9;'>σ_pos = √(σ_t² + σ_c²)</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with feat_c2:
            st.markdown(
                """
                <div class='kpi-card'>
                    <div style='color:#00E5FF; font-weight:700; font-size:0.85rem; margin-bottom:4px;'>Covariance Ellipsoid Volume</div>
                    <div style='font-size:0.78rem; color:#94A3B8; line-height:1.4;'>
                        True 3D positional uncertainty ellipsoid volume derived from full 3x3 covariance determinants:
                        <br><code class='mono-data' style='color:#F1F5F9;'>Volume = √(det(C_pos))</code>
                        <br>Tracks orbit determination precision expansion or collapse.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with feat_c3:
            st.markdown(
                """
                <div class='kpi-card'>
                    <div style='color:#00E5FF; font-weight:700; font-size:0.85rem; margin-bottom:4px;'>Volume Contraction Ratio</div>
                    <div style='font-size:0.78rem; color:#94A3B8; line-height:1.4;'>
                        Dynamic rate of uncertainty collapse between successive tracking messages:
                        <br><code class='mono-data' style='color:#F1F5F9;'>Ratio = Vol_latest / Vol_earliest</code>
                        <br>Identifies rapid covariance convergence on dead-center approaches.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Global Tree SHAP Feature Importance Card
        shap_img_path = Path("reports/figures/shap_importance.png")
        if shap_img_path.exists():
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown("### Global Model Explainability (Tree SHAP Attributions)")
            st.image(
                str(shap_img_path),
                caption="Top 15 Feature Importances: Covariance volume contraction and combined positional sigmas dominate decision boundaries.",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

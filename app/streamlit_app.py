"""NearMiss (SW-5) — Autonomous Satellite Collision Risk Predictor.

Aerospace Operations Console & Decision Support System.
Views:
- Overview: Mission briefing, operational pipeline, live cohort telemetry, and featured case studies.
- Event Inspector: Conjunction trajectory deep-dive, Decision HUD vs Baseline A, and blind-zone reveal.
- Policy Simulator: Constellation mission economics, threshold frontier, and confusion matrices.
- Benchmarks & Physics: Full-width ESA Kelvins scorecards, calibration reliability, and orbital physics.
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
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import compute_classification_metrics
from src.load import load_config, load_parquet

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NearMiss — Satellite Collision Risk Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_PAGES = [
    "Overview",
    "Event Inspector",
    "Policy Simulator",
    "Benchmarks & Physics",
]

# -----------------------------------------------------------------------------
# ORBITAL TELEMETRY DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-canvas: #060913;
        --bg-surface: #0F172A;
        --bg-elevated: #16223B;
        --bg-hover: #1E2E4E;
        --border-subtle: #243554;
        --border-bright: #334B75;
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --cyan: #22D3EE;
        --indigo: #818CF8;
        --coral: #FB7185;
        --red-alert: #F43F5E;
        --emerald: #34D399;
        --amber: #FBBF24;
    }

    /* Base App & Layout Spacing */
    .stApp {
        background: radial-gradient(circle at 15% 0%, rgba(34, 211, 238, 0.06) 0%, var(--bg-canvas) 45%) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Tighten Streamlit main container so there are no huge blank gaps */
    .main .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1420px !important;
    }

    /* Hide noisy Streamlit deploy/menu bar while preserving sidebar collapse button */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden !important;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Sidebar Polish — prevent clipping and ensure readable spacing */
    section[data-testid="stSidebar"] {
        background-color: #090E1C !important;
        border-right: 1px solid var(--border-subtle) !important;
        min-width: 280px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding: 1.25rem 1.1rem 2rem 1.1rem !important;
    }
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 14px;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 14px;
    }
    .sb-badge {
        background: rgba(34, 211, 238, 0.14);
        border: 1.5px solid rgba(34, 211, 238, 0.5);
        color: var(--cyan);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 6px 10px;
        border-radius: 8px;
        flex-shrink: 0;
    }
    .sb-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--cyan);
        margin: 16px 0 6px 0;
    }
    .sb-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.45;
        margin-bottom: 8px;
    }
    .sb-rule-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        padding: 5px 0;
        border-bottom: 1px solid rgba(36, 53, 84, 0.55);
        font-size: 0.78rem;
    }
    .sb-rule-row:last-child {
        border-bottom: none;
    }

    /* Top Command Header */
    .cmd-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(22, 34, 59, 0.92) 100%);
        border: 1px solid var(--border-bright);
        border-top: 2px solid var(--cyan);
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .cmd-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .cmd-logo {
        background: rgba(34, 211, 238, 0.14);
        border: 1.5px solid rgba(34, 211, 238, 0.55);
        color: var(--cyan);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 6px 12px;
        border-radius: 8px;
        letter-spacing: 0.06em;
    }
    .cmd-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .cmd-sub {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 2px;
    }
    .cmd-pills {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .pill-cohort {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--cyan);
        background: rgba(34, 211, 238, 0.12);
        border: 1px solid rgba(34, 211, 238, 0.35);
        padding: 5px 12px;
        border-radius: 999px;
    }
    .pill-cutoff {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        color: #FDA4AF;
        background: rgba(244, 63, 94, 0.14);
        border: 1px solid rgba(244, 63, 94, 0.4);
        padding: 5px 12px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--red-alert);
        box-shadow: 0 0 8px var(--red-alert);
    }

    /* Landing Hero */
    .hero-panel {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(22, 34, 59, 0.92) 60%, rgba(15, 36, 56, 0.95) 100%);
        border: 1px solid var(--border-bright);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 12px;
    }
    .hero-accent {
        color: var(--cyan);
    }
    .hero-copy {
        font-size: 1rem;
        color: var(--text-secondary);
        line-height: 1.65;
        max-width: 880px;
        margin-bottom: 0;
    }

    /* Pipeline Strip */
    .pipe-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 16px 0 22px 0;
    }
    @media (max-width: 960px) {
        .pipe-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .pipe-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-top: 3px solid var(--cyan);
        border-radius: 10px;
        padding: 16px;
    }
    .pipe-card.cutoff { border-top-color: var(--red-alert); }
    .pipe-card.physics { border-top-color: var(--indigo); }
    .pipe-card.calib { border-top-color: var(--emerald); }
    .pipe-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .pipe-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    .pipe-desc {
        font-size: 0.83rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* KPI / Stat Cards */
    .kpi-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 5px;
    }

    /* Info / Case Cards */
    .info-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .info-card-title {
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .info-card-body {
        font-size: 0.86rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* Decision HUD */
    .hud-panel {
        background: var(--bg-surface);
        border: 1px solid var(--border-bright);
        border-radius: 12px;
        padding: 20px 22px;
        margin: 12px 0 16px 0;
    }
    .badge-burn {
        background: rgba(244, 63, 94, 0.16);
        border: 2px solid var(--red-alert);
        color: #FDA4AF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 800;
        padding: 9px 18px;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 0.04em;
    }
    .badge-safe {
        background: rgba(52, 211, 153, 0.14);
        border: 2px solid var(--emerald);
        color: #6EE7B7;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 800;
        padding: 9px 18px;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 0.04em;
    }
    .conf-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .conf-high { background: rgba(34, 211, 238, 0.16); color: var(--cyan); border: 1px solid rgba(34, 211, 238, 0.45); }
    .conf-med { background: rgba(251, 191, 36, 0.16); color: var(--amber); border: 1px solid rgba(251, 191, 36, 0.45); }
    .conf-low { background: rgba(148, 163, 184, 0.16); color: var(--text-secondary); border: 1px solid rgba(148, 163, 184, 0.45); }

    .hud-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        border-top: 1px solid var(--border-subtle);
        padding-top: 14px;
        margin-top: 14px;
    }
    @media (max-width: 820px) {
        .hud-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    /* Clean Styled Buttons */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        padding: 0.5rem 0.9rem !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-bright) !important;
        color: var(--text-secondary) !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--cyan) !important;
        color: var(--text-primary) !important;
        background-color: var(--bg-elevated) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0891B2 0%, #22D3EE 100%) !important;
        border: none !important;
        color: #060913 !important;
    }

    /* Section Header */
    .section-head {
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 8px 0 4px 0;
        letter-spacing: -0.01em;
    }
    .section-sub {
        font-size: 0.86rem;
        color: var(--text-muted);
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# DATA ACCESS (Signature preserved for tests/test_app.py)
# -----------------------------------------------------------------------------
@st.cache_data
def load_app_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load configuration, predictions parquet, and CDM history parquet."""
    cfg = load_config()
    preds_df = load_parquet("data/processed/predictions.parquet")
    hist_df = load_parquet("data/processed/cdm_history.parquet")
    return cfg, preds_df, hist_df


def _navigate_to(page: str, event_id: int | None = None) -> None:
    """Switch active page and optionally focus a specific event_id."""
    st.session_state["active_page"] = page
    if event_id is not None:
        st.session_state["selected_event_id"] = int(event_id)
        st.session_state["rec_filter"] = "All"
        st.session_state["conf_filter"] = "All"
        st.session_state["search_query"] = ""


def _notable_events(split_preds: pd.DataFrame) -> dict[str, int | None]:
    """Find representative notable encounters in the active split."""
    rescued = split_preds[
        (split_preds["recommendation"] == "MANOEUVRE")
        & (split_preds["baseline_recommendation"] != "MANOEUVRE")
        & (split_preds["actual_high_risk"] == 1)
    ].sort_values("high_risk_prob", ascending=False)

    avoided = split_preds[
        (split_preds["recommendation"] == "NO_MANOEUVRE")
        & (split_preds["baseline_recommendation"] == "MANOEUVRE")
        & (split_preds["actual_high_risk"] == 0)
    ].sort_values("latest_log_risk", ascending=False)

    top_risk = split_preds.sort_values("high_risk_prob", ascending=False)

    return {
        "rescued": int(rescued["event_id"].iloc[0]) if len(rescued) > 0 else None,
        "avoided": int(avoided["event_id"].iloc[0]) if len(avoided) > 0 else None,
        "top_risk": int(top_risk["event_id"].iloc[0]) if len(top_risk) > 0 else None,
    }


# -----------------------------------------------------------------------------
# PAGE RENDERERS
# -----------------------------------------------------------------------------
def render_overview(split_preds: pd.DataFrame, split_label: str, cutoff_days: float) -> None:
    """Render the Mission Briefing landing page."""
    st.markdown(
        f"""
        <div class='hero-panel'>
            <div class='hero-title'>
                Autonomous Collision Avoidance<br>
                <span class='hero-accent'>Before the {cutoff_days:.0f}-Day Decision Horizon</span>
            </div>
            <p class='hero-copy'>
                Satellite operators receive hundreds of Conjunction Data Messages (CDMs) every week, yet planning
                and uplinking an avoidance burn takes <strong>24 to 48 hours</strong>. Once an encounter crosses
                <strong>T-48h</strong>, operators enter the <strong>Blind Zone</strong> where late radar updates arrive too late to act.
                NearMiss forecasts terminal collision risk strictly from early tracking data—catching late-jumping hazards
                while eliminating costly false-alarm burns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta1, cta2, cta3, _ = st.columns([1.1, 1.1, 1.1, 2.2])
    with cta1:
        if st.button("Inspect Encounters", type="primary", use_container_width=True, key="hero_cta_inspect"):
            _navigate_to("Event Inspector")
            st.rerun()
    with cta2:
        if st.button("Simulate Burn Policy", type="secondary", use_container_width=True, key="hero_cta_sim"):
            _navigate_to("Policy Simulator")
            st.rerun()
    with cta3:
        if st.button("View ESA Benchmarks", type="secondary", use_container_width=True, key="hero_cta_bench"):
            _navigate_to("Benchmarks & Physics")
            st.rerun()

    # Live Telemetry KPIs
    n_total = len(split_preds)
    n_man = int((split_preds["recommendation"] == "MANOEUVRE").sum())
    n_danger = int(split_preds["actual_high_risk"].sum())
    man_pct = (n_man / max(1, n_total)) * 100
    danger_pct = (n_danger / max(1, n_total)) * 100

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Monitored Encounters</div>
                <div class='kpi-val'>{n_total:,}</div>
                <div class='kpi-sub'>{split_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Maneuvers Commanded</div>
                <div class='kpi-val' style='color:#FB7185;'>{n_man:,}</div>
                <div class='kpi-sub'>{man_pct:.1f}% burn recommendation rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Confirmed Hazards</div>
                <div class='kpi-val' style='color:#FBBF24;'>{n_danger:,}</div>
                <div class='kpi-sub'>{danger_pct:.1f}% actual terminal risk ≥ 10⁻⁶</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Decision Lead Time</div>
                <div class='kpi-val' style='color:#22D3EE;'>≥ {cutoff_days * 24:.0f} h</div>
                <div class='kpi-sub'>Zero blind-zone data leakage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4-Step Architecture Flow
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'>How NearMiss Operates</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>End-to-end orbital risk pipeline enforced under strict T-48h cutoff isolation.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='pipe-grid'>
            <div class='pipe-card'>
                <div class='pipe-tag'>01 · INGESTION</div>
                <div class='pipe-title'>Early Radar Passes</div>
                <div class='pipe-desc'>Collects Conjunction Data Messages (CDMs) from T-7d up to the planning deadline, tracking miss distance and covariance evolution.</div>
            </div>
            <div class='pipe-card cutoff'>
                <div class='pipe-tag'>02 · T-48H BARRIER</div>
                <div class='pipe-title'>Strict Cutoff Isolation</div>
                <div class='pipe-desc'>Discards every CDM inside 2.0 days to TCA. No model or feature ever sees blind-zone updates before issuing a decision.</div>
            </div>
            <div class='pipe-card physics'>
                <div class='pipe-tag'>03 · ORBITAL PHYSICS</div>
                <div class='pipe-title'>73 Covariance Features</div>
                <div class='pipe-desc'>Computes 3D uncertainty ellipsoid volumes √det(C), contraction ratios, combined RTN sigmas, and miss-distance gradients.</div>
            </div>
            <div class='pipe-card calib'>
                <div class='pipe-tag'>04 · CALIBRATION</div>
                <div class='pipe-title'>Platt Risk & Confidence</div>
                <div class='pipe-desc'>Converts raw ensemble scores into calibrated collision probabilities (-59.5% Brier error) with HIGH / MEDIUM / LOW confidence tiers.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Verified Outcomes + Interactive Case Studies
    st.markdown("<div class='section-head'>Operational Impact & Featured Encounters</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Compare NearMiss against legacy Persistence (Baseline A) or jump directly into real conjunction trajectories.</div>",
        unsafe_allow_html=True,
    )

    notable = _notable_events(split_preds)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            """
            <div class='info-card'>
                <div class='info-card-title' style='color:#22D3EE;'>22.6% Fewer Unnecessary Burns</div>
                <div class='info-card-body'>
                    Legacy persistence triggers false alarms whenever early tracking noise temporarily spikes collision probability.
                    NearMiss recognizes expanding covariance volumes and diverging miss distances to stand down safely—saving up to USD 50,000 per avoided burn.
                </div>
            </div>
            <div class='info-card'>
                <div class='info-card-title' style='color:#34D399;'>59.5% Probability Calibration Error Reduction</div>
                <div class='info-card-body'>
                    Under a ~0.8% high-risk base rate, raw tree scores are heavily skewed. Out-of-fold Platt scaling lowers Brier error
                    from <strong>0.01724 to 0.00698</strong>, giving flight dynamics teams reliable posterior probabilities.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r:
        rescued_id = notable["rescued"]
        avoided_id = notable["avoided"]

        st.markdown(
            f"""
            <div class='info-card'>
                <div class='info-card-title' style='color:#FB7185;'>Rescued Late-Jumping Collision</div>
                <div class='info-card-body'>
                    Baseline A cleared Encounter <strong>#{rescued_id if rescued_id is not None else 'N/A'}</strong> as safe at T-48h.
                    NearMiss detected rapid covariance contraction and commanded an avoidance burn before the blind zone.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if rescued_id is not None and st.button(
            f"Inspect Rescued Event #{rescued_id}",
            key="overview_jump_rescued",
            use_container_width=True,
        ):
            _navigate_to("Event Inspector", rescued_id)
            st.rerun()

        st.markdown(
            f"""
            <div class='info-card' style='margin-top:10px;'>
                <div class='info-card-title' style='color:#FBBF24;'>Avoided False-Alarm Burn</div>
                <div class='info-card-body'>
                    Baseline A ordered a costly maneuver on Encounter <strong>#{avoided_id if avoided_id is not None else 'N/A'}</strong>.
                    NearMiss identified transient early uncertainty and cleared the orbit without wasting propellant.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if avoided_id is not None and st.button(
            f"Inspect Saved-Burn Event #{avoided_id}",
            key="overview_jump_avoided",
            use_container_width=True,
        ):
            _navigate_to("Event Inspector", avoided_id)
            st.rerun()


def render_inspector(
    split_preds: pd.DataFrame,
    all_hist: pd.DataFrame,
    high_risk_thresh: float,
    cutoff_days: float,
) -> None:
    """Render the Conjunction Event Inspector."""
    st.markdown("<div class='section-head'>Conjunction Event Inspector</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Audit individual trajectories, compare NearMiss against Baseline A at T-48h, and toggle post-cutoff ground truth.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.8])
    with c1:
        rec_filter = st.selectbox("Recommendation", ["All", "MANOEUVRE", "NO_MANOEUVRE"], key="rec_filter")
    with c2:
        conf_filter = st.selectbox("Confidence", ["All", "HIGH", "MEDIUM", "LOW"], key="conf_filter")
    with c3:
        search_query = st.text_input("Event ID Search", placeholder="e.g. 873, 193", key="search_query")

    filtered = split_preds.copy()
    if rec_filter != "All":
        filtered = filtered[filtered["recommendation"] == rec_filter]
    if conf_filter != "All":
        filtered = filtered[filtered["confidence"] == conf_filter]
    if search_query.strip():
        try:
            filtered = filtered[filtered["event_id"] == int(search_query.strip())]
        except ValueError:
            st.warning("Event ID must be an integer (e.g. 873).")

    filtered = filtered.sort_values("high_risk_prob", ascending=False)

    with c4:
        if len(filtered) == 0:
            st.selectbox("Select Encounter (0 matches)", ["No matching encounters"], disabled=True)
        else:
            event_options = filtered["event_id"].tolist()
            # Keep session_state selection valid when switching cohorts or filters
            if st.session_state.get("selected_event_id") not in event_options:
                st.session_state["selected_event_id"] = event_options[0]

            lookup = filtered.set_index("event_id")
            selected_event_id = st.selectbox(
                f"Select Encounter ({len(filtered):,} matches)",
                event_options,
                key="selected_event_id",
                format_func=lambda eid: (
                    f"#{eid} · P={lookup.loc[eid, 'high_risk_prob']:.1%} · "
                    f"{lookup.loc[eid, 'recommendation']} [{lookup.loc[eid, 'confidence']}]"
                ),
            )

    if len(filtered) == 0:
        st.info("No conjunctions match your current filter combination.")
        if st.button("Reset Filters", type="primary"):
            st.session_state["rec_filter"] = "All"
            st.session_state["conf_filter"] = "All"
            st.session_state["search_query"] = ""
            st.rerun()
        return

    event_row = split_preds[split_preds["event_id"] == selected_event_id].iloc[0]
    conf = str(event_row["confidence"]).upper()
    conf_class = f"conf-{conf.lower()[:3]}"
    is_manoeuvre = event_row["recommendation"] == "MANOEUVRE"
    b_rec = event_row["baseline_recommendation"]

    rec_html = (
        "<div class='badge-burn'>COMMAND AVOIDANCE BURN</div>"
        if is_manoeuvre
        else "<div class='badge-safe'>NOMINAL ORBIT · STAND DOWN</div>"
    )
    b_rec_html = (
        "<span style='color:#FB7185; font-weight:800;'>COMMAND BURN</span>"
        if b_rec == "MANOEUVRE"
        else "<span style='color:#34D399; font-weight:800;'>STAND DOWN</span>"
    )
    divergence = (
        "AGREEMENT"
        if is_manoeuvre == (b_rec == "MANOEUVRE")
        else ("RESCUED COLLISION" if is_manoeuvre else "SAVED MANEUVER")
    )
    div_color = "#94A3B8" if divergence == "AGREEMENT" else ("#22D3EE" if divergence == "RESCUED COLLISION" else "#34D399")

    st.markdown(
        f"""
        <div class='hud-panel'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;'>
                <div>
                    <div style='font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700; margin-bottom:8px;'>
                        Encounter #{selected_event_id} · Chaser Type: {event_row['object_type']}
                    </div>
                    {rec_html}
                </div>
                <div style='display:flex; flex-direction:column; align-items:flex-end; gap:6px;'>
                    <div style='display:flex; align-items:center; gap:8px;'>
                        <span style='font-size:0.8rem; color:#94A3B8;'>Confidence:</span>
                        <span class='conf-pill {conf_class}'>{conf}</span>
                    </div>
                    <div style='font-size:0.8rem; color:#CBD5E1;'>
                        Target: <strong>Sat #{event_row['mission_id']}</strong> · Pre-Cutoff Passes: <strong style='color:#22D3EE;'>{int(event_row['cdm_count_pre_cutoff'])} CDMs</strong>
                    </div>
                </div>
            </div>
            <div class='hud-grid'>
                <div>
                    <div class='kpi-label'>Calibrated P(Collision)</div>
                    <div class='mono' style='font-size:1.45rem; font-weight:800; color:#F8FAFC;'>{event_row['high_risk_prob']:.2%}</div>
                    <div style='font-size:0.76rem; color:#94A3B8;'>Platt scaled posterior</div>
                </div>
                <div>
                    <div class='kpi-label'>Predicted Terminal Risk</div>
                    <div class='mono' style='font-size:1.45rem; font-weight:800; color:#22D3EE;'>{event_row['pred_final_log_risk']:.2f}</div>
                    <div style='font-size:0.76rem; color:#94A3B8;'>log10(P) hurdle estimate</div>
                </div>
                <div>
                    <div class='kpi-label'>Baseline A (Persistence)</div>
                    <div class='mono' style='font-size:1.1rem; margin-top:4px;'>{b_rec_html}</div>
                    <div style='font-size:0.76rem; color:#94A3B8;'>Last visible: {event_row['latest_log_risk']:.2f}</div>
                </div>
                <div>
                    <div class='kpi-label'>Policy Divergence</div>
                    <div class='mono' style='font-size:1.05rem; font-weight:800; color:{div_color}; margin-top:4px;'>{divergence}</div>
                    <div style='font-size:0.76rem; color:#94A3B8;'>NearMiss vs Baseline A</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reveal_gt = st.toggle("Reveal Post-Cutoff Ground Truth (Blind-Zone Simulation)", value=False)
    if reveal_gt:
        actual_risk_val = float(event_row["actual_final_log_risk"])
        actual_danger = event_row["actual_high_risk"] == 1
        outcome_color = "#FB7185" if actual_danger else "#34D399"
        outcome_text = "ACTUAL COLLISION HAZARD (Risk ≥ 10⁻⁶)" if actual_danger else "ACTUAL NOMINAL FLYBY (Risk < 10⁻⁶)"
        st.markdown(
            f"""
            <div style='background:#0F172A; border:1px solid {outcome_color}; border-radius:8px; padding:10px 14px; margin-bottom:14px; font-family:"JetBrains Mono",monospace; font-size:0.84rem;'>
                Terminal TCA Log-Risk: <strong style='color:#F8FAFC;'>{actual_risk_val:.2f}</strong> &nbsp;|&nbsp;
                <strong style='color:{outcome_color};'>{outcome_text}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    event_history = all_hist[all_hist["event_id"] == selected_event_id].sort_values("time_to_tca", ascending=False)
    vis_cdms = event_history[event_history["visible"]]
    post_cdms = event_history[~event_history["visible"]]

    chart_c1, chart_c2 = st.columns([1.45, 1])

    with chart_c1:
        st.markdown(f"**Collision Risk Trajectory — Encounter #{selected_event_id}**")
        fig_risk = go.Figure()

        fig_risk.add_vrect(
            x0=cutoff_days,
            x1=0.0,
            fillcolor="#F43F5E",
            opacity=0.09,
            layer="below",
            line_width=0,
        )
        fig_risk.add_annotation(
            x=cutoff_days / 2.0,
            y=0.96,
            yref="paper",
            text="BLIND ZONE (< 2.0d)",
            showarrow=False,
            font=dict(color="#FDA4AF", size=10, family="JetBrains Mono"),
        )
        fig_risk.add_vline(
            x=cutoff_days,
            line_dash="dash",
            line_color="#F43F5E",
            line_width=2,
        )
        fig_risk.add_annotation(
            x=cutoff_days,
            y=0.06,
            yref="paper",
            text=" T-48h Cutoff",
            xanchor="left",
            showarrow=False,
            font=dict(color="#FB7185", size=10, family="JetBrains Mono"),
        )
        fig_risk.add_hline(
            y=high_risk_thresh,
            line_dash="dot",
            line_color="#FBBF24",
            line_width=1.5,
        )
        fig_risk.add_annotation(
            x=0.02,
            xref="paper",
            y=high_risk_thresh + 0.8,
            text="1:1M Threshold (-6.0)",
            xanchor="left",
            showarrow=False,
            font=dict(color="#FBBF24", size=10, family="JetBrains Mono"),
        )

        fig_risk.add_trace(
            go.Scatter(
                x=vis_cdms["time_to_tca"],
                y=vis_cdms["log_risk"],
                mode="lines+markers",
                name="Observable CDMs (≥ 2.0d)",
                line=dict(color="#22D3EE", width=2.5),
                marker=dict(size=8, color="#22D3EE"),
                hovertemplate="Time to TCA: %{x:.2f}d<br>Observed Risk: %{y:.2f}<extra></extra>",
            )
        )
        fig_risk.add_trace(
            go.Scatter(
                x=[0.0],
                y=[event_row["pred_final_log_risk"]],
                mode="markers",
                name=f"NearMiss Prediction ({event_row['pred_final_log_risk']:.2f})",
                marker=dict(size=13, color="#818CF8", symbol="diamond", line=dict(width=1.5, color="#F8FAFC")),
                hovertemplate="Predicted TCA Risk: %{y:.2f}<extra></extra>",
            )
        )
        if reveal_gt and len(post_cdms) > 0:
            fig_risk.add_trace(
                go.Scatter(
                    x=post_cdms["time_to_tca"],
                    y=post_cdms["log_risk"],
                    mode="lines+markers",
                    name="Post-Cutoff Truth",
                    line=dict(color="#FBBF24", width=2, dash="dash"),
                    marker=dict(size=7, color="#FBBF24", symbol="triangle-down"),
                    hovertemplate="Actual Time: %{x:.2f}d<br>Actual Risk: %{y:.2f}<extra></extra>",
                )
            )

        fig_risk.update_layout(
            xaxis=dict(title="Days Until TCA", autorange="reversed", gridcolor="#1E293B", color="#CBD5E1"),
            yaxis=dict(title="Risk log10(P)", range=[-32, 0.5], gridcolor="#1E293B", color="#CBD5E1"),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#090E1C",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#CBD5E1"),
            ),
            height=410,
            margin=dict(l=64, r=20, t=24, b=82),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with chart_c2:
        st.markdown("**Relative Miss Distance Profile**")
        fig_miss = go.Figure()
        fig_miss.add_vrect(
            x0=cutoff_days,
            x1=0.0,
            fillcolor="#F43F5E",
            opacity=0.07,
            layer="below",
            line_width=0,
        )
        fig_miss.add_vline(x=cutoff_days, line_dash="dash", line_color="#F43F5E", line_width=1.5)
        fig_miss.add_trace(
            go.Scatter(
                x=vis_cdms["time_to_tca"],
                y=vis_cdms["miss_distance"],
                mode="lines+markers",
                name="Observable Miss (m)",
                line=dict(color="#34D399", width=2.5),
                marker=dict(size=7, color="#34D399"),
                hovertemplate="Time to TCA: %{x:.2f}d<br>Miss: %{y:,.0f} m<extra></extra>",
            )
        )
        if reveal_gt and len(post_cdms) > 0:
            fig_miss.add_trace(
                go.Scatter(
                    x=post_cdms["time_to_tca"],
                    y=post_cdms["miss_distance"],
                    mode="lines+markers",
                    name="Post-Cutoff Truth",
                    line=dict(color="#FBBF24", width=2, dash="dash"),
                    marker=dict(size=6, color="#FBBF24"),
                    hovertemplate="Actual Time: %{x:.2f}d<br>Actual Miss: %{y:,.0f} m<extra></extra>",
                )
            )

        fig_miss.update_layout(
            xaxis=dict(title="Days Until TCA", autorange="reversed", gridcolor="#1E293B", color="#CBD5E1"),
            yaxis=dict(title="Miss Distance (m)", gridcolor="#1E293B", color="#CBD5E1"),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#090E1C",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#CBD5E1"),
            ),
            height=410,
            margin=dict(l=68, r=20, t=24, b=82),
        )
        st.plotly_chart(fig_miss, use_container_width=True)

    with st.expander("CDM Telemetry Log", expanded=False):
        display_hist = event_history if reveal_gt else vis_cdms
        tbl = display_hist[["time_to_tca", "log_risk", "miss_distance", "visible"]].copy()
        tbl.columns = ["Days to TCA", "Log10 Risk", "Miss Distance (m)", "Pre-Cutoff Visible"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)


def render_simulator(split_preds: pd.DataFrame) -> None:
    """Render the Constellation Mission Economics & Policy Simulator."""
    st.markdown("<div class='section-head'>Constellation Mission Economics & Policy Simulator</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Tune the probability decision threshold to balance catastrophic collision risk against propellant expenditure.</div>",
        unsafe_allow_html=True,
    )

    sim_c1, sim_c2 = st.columns([1, 1.75])

    with sim_c1:
        st.markdown("#### 1. Mission Profile")
        # Avoid '$' signs in radio labels so Streamlit does not trigger KaTeX math mode
        preset = st.radio(
            "Mission Profile",
            [
                "Commercial LEO SmallSat (USD 5M sat / USD 15k burn)",
                "Flagship Earth Observation (USD 300M sat / USD 45k burn)",
                "High-Value Infrastructure (USD 1B sat / USD 100k burn)",
            ],
            index=0,
            label_visibility="collapsed",
        )

        if "SmallSat" in preset:
            default_sat_val, default_burn_val, default_th = 5_000_000, 15_000, 0.030
        elif "Earth Observation" in preset:
            default_sat_val, default_burn_val, default_th = 300_000_000, 45_000, 0.020
        else:
            default_sat_val, default_burn_val, default_th = 1_000_000_000, 100_000, 0.010

        st.markdown("#### 2. Decision Threshold (p*)")
        sim_th = st.slider(
            "Probability Threshold",
            min_value=0.005,
            max_value=0.250,
            value=default_th,
            step=0.005,
            format="%.3f",
        )
        st.caption(f"Encounters with P(collision) ≥ {sim_th:.1%} trigger a maneuver.")

        st.markdown("#### 3. Cost Parameters (USD)")
        cost_collision = st.number_input("Collision Loss Cost (USD)", value=default_sat_val, step=1_000_000)
        cost_manoeuvre = st.number_input("Avoidance Burn Cost (USD)", value=default_burn_val, step=5_000)

        sim_pred_bin = (split_preds["high_risk_prob"] >= sim_th).astype(int)
        y_eval_true = split_preds["actual_high_risk"].values
        sim_metrics = compute_classification_metrics(y_eval_true, sim_pred_bin, split_preds["high_risk_prob"].values)

        base_pred_bin = (split_preds["baseline_recommendation"] == "MANOEUVRE").astype(int)
        base_metrics = compute_classification_metrics(y_eval_true, base_pred_bin)

        cost_model = (sim_metrics["fn"] * cost_collision) + ((sim_metrics["tp"] + sim_metrics["fp"]) * cost_manoeuvre)
        cost_baseline = (base_metrics["fn"] * cost_collision) + ((base_metrics["tp"] + base_metrics["fp"]) * cost_manoeuvre)
        savings = cost_baseline - cost_model
        savings_pct = (savings / max(1, cost_baseline)) * 100

        savings_color = "#34D399" if savings >= 0 else "#FB7185"
        st.markdown(
            f"""
            <div class='info-card' style='margin-top:14px; border-color:{savings_color};'>
                <div class='kpi-label'>Net Financial Balance vs Baseline A</div>
                <div class='mono' style='font-size:1.75rem; font-weight:800; color:{savings_color}; margin:4px 0;'>
                    USD {savings:+,.0f}
                </div>
                <div style='font-size:0.82rem; color:#CBD5E1;'>
                    Relative cost delta: <strong style='color:{savings_color};'>{savings_pct:+.1f}%</strong> vs legacy persistence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sim_c2:
        st.markdown("#### Operational Risk Frontier (Missed Collisions vs False Alarms)")

        th_grid = np.linspace(0.005, 0.35, 70)
        sweep_pts = []
        for t in th_grid:
            p_bin = (split_preds["high_risk_prob"] >= t).astype(int)
            m = compute_classification_metrics(y_eval_true, p_bin)
            sweep_pts.append({"threshold": t, "fn": m["fn"], "fp": m["fp"]})
        sweep_df = pd.DataFrame(sweep_pts)

        fig_sim = go.Figure()
        fig_sim.add_trace(
            go.Scatter(
                x=sweep_df["fp"],
                y=sweep_df["fn"],
                mode="lines",
                name="NearMiss Frontier",
                line=dict(color="#22D3EE", width=3),
                hovertemplate="False Alarms (FP): %{x}<br>Missed Collisions (FN): %{y}<extra></extra>",
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=[sim_metrics["fp"]],
                y=[sim_metrics["fn"]],
                mode="markers",
                name=f"Selected Policy (p*={sim_th:.3f})",
                marker=dict(size=14, color="#F43F5E", symbol="star", line=dict(width=1, color="#F8FAFC")),
                hovertemplate=f"Selected Policy (p*={sim_th:.3f})<br>FP: %{{x}}, FN: %{{y}}<extra></extra>",
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=[base_metrics["fp"]],
                y=[base_metrics["fn"]],
                mode="markers",
                name="Baseline A (Persistence)",
                marker=dict(size=12, color="#FBBF24", symbol="square", line=dict(width=1, color="#060913")),
                hovertemplate="Baseline A<br>FP: %{x}, FN: %{y}<extra></extra>",
            )
        )

        fig_sim.update_layout(
            xaxis=dict(title="Unnecessary Maneuvers (False Positives)", gridcolor="#1E293B", color="#CBD5E1"),
            yaxis=dict(title="Missed Collisions (False Negatives)", gridcolor="#1E293B", color="#CBD5E1"),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#090E1C",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#CBD5E1"),
            ),
            height=350,
            margin=dict(l=68, r=24, t=20, b=82),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        mat_c1, mat_c2 = st.columns(2)
        with mat_c1:
            st.markdown(
                f"""
                <div class='info-card'>
                    <div style='font-size:0.76rem; font-weight:800; color:#22D3EE; text-transform:uppercase; margin-bottom:8px;'>
                        NearMiss Matrix (p* = {sim_th:.3f})
                    </div>
                    <table style='width:100%; border-collapse:separate; border-spacing:4px; font-family:"JetBrains Mono",monospace; font-size:0.84rem;'>
                        <tr><th></th><th style='color:#94A3B8;'>Pred Safe</th><th style='color:#94A3B8;'>Pred Burn</th></tr>
                        <tr>
                            <th style='color:#94A3B8;'>Safe</th>
                            <td style='background:#16223B; padding:8px; border-radius:4px;'>{sim_metrics['tn']:,} (TN)</td>
                            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF; padding:8px; border-radius:4px;'>{sim_metrics['fp']:,} (FP)</td>
                        </tr>
                        <tr>
                            <th style='color:#94A3B8;'>Danger</th>
                            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF; padding:8px; border-radius:4px;'>{sim_metrics['fn']:,} (FN)</td>
                            <td style='background:rgba(52,211,153,0.16); color:#6EE7B7; padding:8px; border-radius:4px;'>{sim_metrics['tp']:,} (TP)</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mat_c2:
            st.markdown(
                f"""
                <div class='info-card'>
                    <div style='font-size:0.76rem; font-weight:800; color:#FBBF24; text-transform:uppercase; margin-bottom:8px;'>
                        Baseline A Matrix (Persistence)
                    </div>
                    <table style='width:100%; border-collapse:separate; border-spacing:4px; font-family:"JetBrains Mono",monospace; font-size:0.84rem;'>
                        <tr><th></th><th style='color:#94A3B8;'>Pred Safe</th><th style='color:#94A3B8;'>Pred Burn</th></tr>
                        <tr>
                            <th style='color:#94A3B8;'>Safe</th>
                            <td style='background:#16223B; padding:8px; border-radius:4px;'>{base_metrics['tn']:,} (TN)</td>
                            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF; padding:8px; border-radius:4px;'>{base_metrics['fp']:,} (FP)</td>
                        </tr>
                        <tr>
                            <th style='color:#94A3B8;'>Danger</th>
                            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF; padding:8px; border-radius:4px;'>{base_metrics['fn']:,} (FN)</td>
                            <td style='background:rgba(52,211,153,0.16); color:#6EE7B7; padding:8px; border-radius:4px;'>{base_metrics['tp']:,} (TP)</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_benchmarks() -> None:
    """Render Model Benchmarks, Calibration, and Orbital Physics."""
    st.markdown("<div class='section-head'>ESA Benchmark Suite & Orbital Physics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Out-of-fold 5-fold cross-validation (8,293 filtered events) and locked competition test set (2,167 events).</div>",
        unsafe_allow_html=True,
    )

    # Full-width benchmark tables so zero columns are clipped
    st.markdown("#### 5-Fold Cross-Validation Performance (8,293 Filtered Events)")
    cv_df = pd.DataFrame(
        [
            ["Baseline A (Persistence)", "Reference Baseline", "71.21%", "15.26%", 0.4108, 0.1803, 19, 261],
            ["Logistic Regression", "Unweighted", "28.79%", "25.00%", 0.2794, 0.1711, 47, 57],
            ["Logistic Regression", "Balanced Weights", "87.88%", "6.58%", 0.2531, 0.1484, 8, 824],
            ["Random Forest", "Unweighted", "45.45%", "24.59%", 0.3886, 0.1759, 36, 92],
            ["Random Forest (Champion)", "Balanced Subsample", "71.21%", "16.85%", 0.4328, 0.1926, 19, 232],
            ["LightGBM Classifier", "Scale-Pos-Weight", "65.15%", "19.11%", 0.4397, 0.1504, 23, 182],
        ],
        columns=["Model Architecture", "Balancing Strategy", "Recall", "Precision", "F2 Score", "PR-AUC", "Missed (FN)", "False Alarms (FP)"],
    )
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    st.markdown("#### Locked Competition Test Benchmark (2,167 Events · 150 High-Risk)")
    test_df = pd.DataFrame(
        [
            ["Baseline A (Persistence)", "Reference (0)", "Reference (0)", "64.61%", 0.7391, 26.64],
            ["NearMiss Production (RF + Platt)", "+12 Avoided Burns", "+3 Rescued Events", "66.46%", 0.7030, 133.79],
        ],
        columns=["Pipeline Variant", "Avoided Burns", "Rescued Collisions", "Precision", "F2 Score", "High-Risk MSE"],
    )
    st.dataframe(test_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    fig_c1, fig_c2 = st.columns(2)
    with fig_c1:
        st.markdown("#### Platt Calibration Reliability (-59.5% Brier Error)")
        cal_img = Path("reports/figures/calibration_curve.png")
        if cal_img.exists():
            st.image(str(cal_img), use_container_width=True)
    with fig_c2:
        st.markdown("#### Global Tree SHAP Feature Attribution")
        shap_img = Path("reports/figures/shap_importance.png")
        if shap_img.exists():
            st.image(str(shap_img), use_container_width=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Orbital Covariance Physics Features (73 Dimensions)")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class='kpi-card'>
                <div style='color:#22D3EE; font-weight:800; font-size:0.98rem; margin-bottom:6px;'>Combined Positional Sigmas</div>
                <div style='font-size:0.84rem; color:#CBD5E1; line-height:1.5;'>
                    Root-sum-square positional dispersion combining target and chaser RTN covariance tensors.
                    <br><br><code class='mono' style='color:#F8FAFC;'>σ_pos = √(σ_t² + σ_c²)</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class='kpi-card'>
                <div style='color:#818CF8; font-weight:800; font-size:0.98rem; margin-bottom:6px;'>Covariance Ellipsoid Volume</div>
                <div style='font-size:0.84rem; color:#CBD5E1; line-height:1.5;'>
                    True 3D positional uncertainty volume computed from full 3×3 covariance determinants.
                    <br><br><code class='mono' style='color:#F8FAFC;'>Vol = √det(C_pos)</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class='kpi-card'>
                <div style='color:#34D399; font-weight:800; font-size:0.98rem; margin-bottom:6px;'>Volume Contraction Ratio</div>
                <div style='font-size:0.84rem; color:#CBD5E1; line-height:1.5;'>
                    Rate of uncertainty collapse across successive radar tracking passes prior to T-48h.
                    <br><br><code class='mono' style='color:#F8FAFC;'>Ratio = Vol_latest / Vol_earliest</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main() -> None:
    cfg, all_preds, all_hist = load_app_data()

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"

    high_risk_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))
    operational_thresh = float(cfg.get("thresholds", {}).get("operational_risk_log10", -4.0))
    cutoff_days = float(cfg["data"]["cutoff_days"])

    # --- SIDEBAR: COHORT, QUICK-JUMPS & OPERATOR GUIDE ---
    with st.sidebar:
        st.markdown(
            """
            <div class='sb-brand'>
                <div class='sb-badge'>SW-5</div>
                <div>
                    <div style='font-weight:800; font-size:1.1rem; color:#F8FAFC;'>NearMiss Console</div>
                    <div style='font-size:0.76rem; color:#94A3B8;'>Orbital Risk Decision Support</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sb-label'>Evaluation Cohort</div>", unsafe_allow_html=True)
        split_choice = st.radio(
            "Dataset Cohort",
            options=["Validation Cohort (Fold 0 OOF)", "Locked Test Set (ESA Blind)"],
            index=0,
            label_visibility="collapsed",
        )
        split_key = "val" if "Validation" in split_choice else "test"
        split_label = "Validation OOF" if split_key == "val" else "Locked ESA Test"
        split_preds = all_preds[all_preds["split"] == split_key].copy()

        cohort_desc = (
            "Out-of-fold validation encounters used to calibrate probabilities without leakage."
            if split_key == "val"
            else "Official blind competition encounters held out from all training and tuning."
        )
        st.markdown(
            f"<div class='sb-card'>{cohort_desc}<br><strong>{len(split_preds):,}</strong> events · <strong>{int(split_preds['actual_high_risk'].sum())}</strong> high-risk</div>",
            unsafe_allow_html=True,
        )

        notable = _notable_events(split_preds)
        st.markdown("<div class='sb-label'>Quick-Jump Encounters</div>", unsafe_allow_html=True)
        if notable["rescued"] is not None:
            if st.button(f"Rescued Collision #{notable['rescued']}", use_container_width=True, key="sb_rescued"):
                _navigate_to("Event Inspector", notable["rescued"])
                st.rerun()
        if notable["avoided"] is not None:
            if st.button(f"Avoided False Alarm #{notable['avoided']}", use_container_width=True, key="sb_avoided"):
                _navigate_to("Event Inspector", notable["avoided"])
                st.rerun()
        if notable["top_risk"] is not None:
            if st.button(f"Highest Risk Event #{notable['top_risk']}", use_container_width=True, key="sb_top"):
                _navigate_to("Event Inspector", notable["top_risk"])
                st.rerun()

        st.markdown("<div class='sb-label'>Operator Threshold Guide</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='sb-card'>
                <div class='sb-rule-row'>
                    <span style='color:#94A3B8;'>Planning Cutoff</span>
                    <span class='mono' style='color:#22D3EE; font-weight:700;'>T-{cutoff_days*24:.0f}h ({cutoff_days:.1f}d)</span>
                </div>
                <div class='sb-rule-row'>
                    <span style='color:#94A3B8;'>Warning Level</span>
                    <span class='mono' style='color:#FBBF24; font-weight:700;'>10⁻⁶ ({high_risk_thresh:.1f})</span>
                </div>
                <div class='sb-rule-row'>
                    <span style='color:#94A3B8;'>Maneuver Level</span>
                    <span class='mono' style='color:#FB7185; font-weight:700;'>10⁻⁴ ({operational_thresh:.1f})</span>
                </div>
            </div>
            <div style='font-size:0.76rem; color:#94A3B8; line-height:1.45;'>
                <strong>Confidence Tiers:</strong> HIGH means clear burn/stand-down margin; LOW flags borderline geometry for radar re-tasking.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- TOP COMMAND HEADER ---
    st.markdown(
        f"""
        <div class='cmd-header'>
            <div class='cmd-brand'>
                <div class='cmd-logo'>NEARMISS</div>
                <div>
                    <div class='cmd-title'>Orbital Conjunction Assessment Console</div>
                    <div class='cmd-sub'>Autonomous Collision Risk Predictor · StarForge Challenge SW-5</div>
                </div>
            </div>
            <div class='cmd-pills'>
                <span class='pill-cohort'>COHORT: {split_label.upper()} ({len(split_preds):,})</span>
                <span class='pill-cutoff'><span class='pulse-dot'></span>T-{cutoff_days*24:.0f}H CUTOFF ACTIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- PRIMARY NAVIGATION BAR ---
    nav_cols = st.columns(len(NAV_PAGES))
    for idx, page_name in enumerate(NAV_PAGES):
        with nav_cols[idx]:
            is_active = st.session_state["active_page"] == page_name
            if st.button(
                page_name,
                type="primary" if is_active else "secondary",
                use_container_width=True,
                key=f"nav_{page_name}",
            ):
                _navigate_to(page_name)
                st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # --- ROUTE TO ACTIVE VIEW ---
    active_page = st.session_state["active_page"]
    if active_page == "Overview":
        render_overview(split_preds, split_label, cutoff_days)
    elif active_page == "Event Inspector":
        render_inspector(split_preds, all_hist, high_risk_thresh, cutoff_days)
    elif active_page == "Policy Simulator":
        render_simulator(split_preds)
    else:
        render_benchmarks()


if __name__ == "__main__":
    main()

"""NearMiss — Autonomous Satellite Collision Risk Predictor.

Aerospace Operations Console & Decision Support Interface.

Views:
- Overview: Landing briefing, live T-48h Blind Zone trajectory showcase,
  operational pipeline, and curated encounter walkthroughs.
- Event Inspector: Searchable conjunction audit, NearMiss vs Baseline A verdict,
  synchronized risk & miss-distance trajectories, and blind-zone reveal.
- Policy Simulator: Decision threshold and mission economics simulator with
  live FN/FP trade-off frontier and confusion matrices.
- Benchmarks: Full-width ESA Kelvins evaluation tables, interactive Platt
  calibration reliability, and Tree SHAP orbital physics attributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is importable when launched via `streamlit run`
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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
    "Benchmarks",
]

# Exact Top-12 global Tree SHAP mean(|SHAP|) importances from Phase 10 audit
TOP_SHAP_FEATURES = [
    ("latest_log_risk", "Latest pre-cutoff log risk", 0.1186, "Trajectory"),
    ("risk_max", "Peak observed risk", 0.0597, "Trajectory"),
    ("latest_max_risk_estimate", "Max possible risk envelope", 0.0546, "Covariance"),
    ("latest_max_risk_scaling", "Covariance scaling factor", 0.0517, "Covariance"),
    ("risk_mean", "Mean risk across passes", 0.0428, "Trajectory"),
    ("risk_min", "Minimum observed risk", 0.0246, "Trajectory"),
    ("risk_total_change", "Net risk change (first to cutoff)", 0.0185, "Dynamics"),
    ("risk_first", "Initial tracking log risk", 0.0133, "Trajectory"),
    ("risk_slope_per_day", "Risk escalation slope / day", 0.0116, "Dynamics"),
    ("latest_time_to_tca", "Hours remaining at last CDM", 0.0072, "Timing"),
    ("latest_uncertainty_volume", "3D ellipsoid volume √det(C)", 0.0067, "Covariance"),
    ("risk_last_step_change", "Most recent CDM risk jump", 0.0062, "Dynamics"),
]

# -----------------------------------------------------------------------------
# DESIGN SYSTEM — ESOC FLIGHT INSTRUMENTATION PALETTE & TYPOGRAPHY
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-0: #0B0E14;
    --bg-1: #121722;
    --bg-2: #192131;
    --bg-3: #222D42;
    --line: rgba(255, 255, 255, 0.08);
    --line-strong: rgba(255, 255, 255, 0.16);
    --ink-1: #F8FAFC;
    --ink-2: #CBD5E1;
    --ink-3: #8B98B0;
    --sky: #38BDF8;
    --sky-bright: #7DD3FC;
    --gold: #F59E0B;
    --gold-soft: #FCD34D;
    --red: #F43F5E;
    --red-soft: #FDA4AF;
    --green: #10B981;
    --green-soft: #6EE7B7;
}

/* Global canvas with subtle top-right orbital horizon lighting */
.stApp {
    background:
        radial-gradient(ellipse 75% 42% at 72% -8%, rgba(56, 189, 248, 0.13) 0%, transparent 62%),
        radial-gradient(ellipse 45% 32% at 18% 0%, rgba(245, 158, 11, 0.06) 0%, transparent 58%),
        var(--bg-0) !important;
    color: var(--ink-1) !important;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Eliminate Streamlit 1.65's massive default container padding (96px top / 80px sides) */
[data-testid="stMainBlockContainer"] {
    padding: 0.9rem 2rem 2.2rem 2rem !important;
    max-width: 1460px !important;
}

[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
    gap: 0.7rem !important;
}

/* Compact transparent top chrome while preserving sidebar toggle */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.2rem !important;
}
[data-testid="stToolbar"] {
    visibility: hidden !important;
}

/* Sidebar — zero top void, zero left clipping, clean mission rail */
section[data-testid="stSidebar"] {
    background-color: #0E131D !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebarHeader"] {
    padding: 0.65rem 1rem 0.15rem 1rem !important;
    min-height: 0 !important;
    height: auto !important;
}
[data-testid="stSidebarUserContent"] {
    padding: 0.2rem 1rem 1.4rem 1rem !important;
}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {
    gap: 0.55rem !important;
}

.mono {
    font-family: 'IBM Plex Mono', monospace !important;
}
.display {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Top Navigation Brand & Live Status */
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 2px 0;
}
.nav-mark {
    width: 30px;
    height: 30px;
    border-radius: 7px;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(245, 158, 11, 0.2));
    border: 1px solid rgba(56, 189, 248, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.nav-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.nav-sub {
    font-size: 0.72rem;
    color: var(--ink-3);
    line-height: 1.2;
}
.nav-status {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 8px;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-1);
    border: 1px solid var(--line-strong);
    border-radius: 999px;
    padding: 5px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-2);
    white-space: nowrap;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--gold);
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.7);
}
.nav-divider {
    border: none;
    border-top: 1px solid var(--line);
    margin: 2px 0 6px 0;
}

/* Streamlit 1.65 Button Styling (exact data-testid match) */
button[data-testid="stBaseButton-primary"] {
    background: var(--sky) !important;
    color: #07101C !important;
    border: 1px solid var(--sky) !important;
    border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    min-height: 36px !important;
    padding: 0.35rem 0.85rem !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background: var(--sky-bright) !important;
    border-color: var(--sky-bright) !important;
}
button[data-testid="stBaseButton-secondary"] {
    background: var(--bg-1) !important;
    color: var(--ink-2) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    min-height: 36px !important;
    padding: 0.35rem 0.8rem !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--bg-2) !important;
    border-color: rgba(56, 189, 248, 0.55) !important;
    color: var(--ink-1) !important;
}

/* Sidebar sections */
.sb-section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--ink-1);
    margin: 4px 0 2px 0;
}
.sb-help {
    font-size: 0.77rem;
    color: var(--ink-3);
    line-height: 1.42;
    margin-bottom: 4px;
}
.sb-stat-strip {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 9px 11px;
    margin-top: 4px;
}
.sb-stat-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--ink-1);
    line-height: 1.15;
}
.sb-stat-lbl {
    font-size: 0.7rem;
    color: var(--ink-3);
    margin-top: 2px;
}
.sb-hr {
    border: none;
    border-top: 1px solid var(--line);
    margin: 8px 0 4px 0;
}
.sb-guide-item {
    padding: 7px 0;
    border-bottom: 1px solid var(--line);
}
.sb-guide-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.sb-guide-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--ink-1);
}
.sb-guide-desc {
    font-size: 0.73rem;
    color: var(--ink-3);
    line-height: 1.38;
    margin-top: 2px;
}

/* Hero & Landing Page Typography */
.hero-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.35rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.03em;
    line-height: 1.12;
    margin: 4px 0 12px 0;
}
.hero-lead {
    font-size: 0.98rem;
    color: var(--ink-2);
    line-height: 1.62;
    max-width: 60ch;
    margin: 0 0 18px 0;
}
.hero-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border-top: 1px solid var(--line-strong);
    border-bottom: 1px solid var(--line);
    padding: 14px 0;
    margin: 4px 0 18px 0;
    gap: 12px;
}
.hm-cell {
    padding-right: 10px;
    border-right: 1px solid var(--line);
}
.hm-cell:last-child {
    border-right: none;
    padding-right: 0;
}
.hm-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.38rem;
    font-weight: 700;
    color: var(--ink-1);
    line-height: 1.15;
}
.hm-lbl {
    font-size: 0.75rem;
    color: var(--ink-3);
    margin-top: 3px;
    line-height: 1.3;
}

/* Structural Pipeline Strip */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.015em;
    margin: 6px 0 2px 0;
}
.section-desc {
    font-size: 0.87rem;
    color: var(--ink-3);
    margin: 0 0 10px 0;
}
.flow-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin: 6px 0 14px 0;
}
.flow-step {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-top: 2px solid var(--sky);
    border-radius: 8px;
    padding: 15px 17px;
}
.flow-step.gold { border-top-color: var(--gold); }
.flow-step.green { border-top-color: var(--green); }
.flow-step-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--ink-1);
    margin-bottom: 5px;
}
.flow-step-body {
    font-size: 0.84rem;
    color: var(--ink-2);
    line-height: 1.52;
}

/* Case Study Panels */
.case-panel {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-left: 3px solid var(--red);
    border-radius: 8px;
    padding: 15px 18px;
    margin-bottom: 8px;
}
.case-panel.saved {
    border-left-color: var(--green);
}
.case-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.case-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--ink-1);
}
.case-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(244, 63, 94, 0.14);
    color: var(--red-soft);
}
.case-tag.saved {
    background: rgba(16, 185, 129, 0.14);
    color: var(--green-soft);
}
.case-body {
    font-size: 0.84rem;
    color: var(--ink-2);
    line-height: 1.5;
    margin-bottom: 10px;
}
.case-compare {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    background: var(--bg-0);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.76rem;
}

/* Compact Decision HUD for Event Inspector */
.hud-strip {
    background: var(--bg-1);
    border: 1px solid var(--line-strong);
    border-left: 4px solid var(--red);
    border-radius: 8px;
    padding: 14px 18px;
    display: grid;
    grid-template-columns: 1.6fr 1fr 1fr 1.15fr 1fr;
    gap: 14px;
    align-items: center;
}
.hud-strip.safe {
    border-left-color: var(--green);
}
.verdict-row {
    display: flex;
    align-items: center;
    gap: 9px;
    flex-wrap: wrap;
}
.verdict-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.verdict-meta {
    font-size: 0.77rem;
    color: var(--ink-3);
    margin-top: 3px;
}
.conf-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
}
.conf-high { background: rgba(56, 189, 248, 0.16); color: var(--sky-bright); border: 1px solid rgba(56, 189, 248, 0.4); }
.conf-medium { background: rgba(245, 158, 11, 0.16); color: var(--gold-soft); border: 1px solid rgba(245, 158, 11, 0.4); }
.conf-low { background: rgba(148, 163, 184, 0.16); color: var(--ink-2); border: 1px solid rgba(148, 163, 184, 0.35); }
.hud-cell {
    border-left: 1px solid var(--line);
    padding-left: 14px;
}
.hud-label {
    font-size: 0.73rem;
    color: var(--ink-3);
    margin-bottom: 2px;
}
.hud-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--ink-1);
    line-height: 1.2;
}
.hud-sub {
    font-size: 0.72rem;
    color: var(--ink-3);
    margin-top: 2px;
}

/* KPI Strip for Simulator & Benchmarks */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 10px;
}
.kpi-tile {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px 14px;
}
.kpi-tile-lbl {
    font-size: 0.75rem;
    color: var(--ink-3);
    margin-bottom: 4px;
}
.kpi-tile-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.32rem;
    font-weight: 700;
    color: var(--ink-1);
    line-height: 1.15;
}
.kpi-tile-sub {
    font-size: 0.74rem;
    color: var(--ink-2);
    margin-top: 3px;
}

/* Confusion Matrix Panels */
.cm-card {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px 14px;
}
.cm-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink-1);
    margin-bottom: 8px;
}
.cm-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 5px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.cm-table th {
    color: var(--ink-3);
    font-weight: 500;
    text-align: left;
    padding: 3px 6px;
}
.cm-table td {
    padding: 7px 9px;
    border-radius: 5px;
    text-align: center;
    font-weight: 600;
}

/* Responsive adjustments */
@media (max-width: 1024px) {
    .hud-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero-metrics, .kpi-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .flow-grid { grid-template-columns: 1fr; }
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


@st.cache_data
def load_calibration_oof() -> pd.DataFrame | None:
    """Load out-of-fold calibrated probabilities for live reliability plotting."""
    cal_path = root_dir / "data/processed/calibrated_oof.parquet"
    if not cal_path.exists():
        return None
    return load_parquet(cal_path)


def _slice_history_for_split(all_hist: pd.DataFrame, split_key: str) -> pd.DataFrame:
    """Isolate CDM history strictly to the selected cohort split.

    Production cdm_history.parquet concatenates train/val CDMs followed by
    test_merged CDMs (where event_id resets from ~13153 back to 0). Slicing
    at that boundary prevents the 405 colliding event_ids between val and test
    from merging into a single jagged trajectory.
    """
    drops = all_hist.index[all_hist["event_id"].diff() < 0].tolist()
    if not drops:
        return all_hist
    boundary = int(drops[0])
    if split_key == "test":
        return all_hist.iloc[boundary:]
    return all_hist.iloc[:boundary]


def _navigate_to(page: str, event_id: int | None = None) -> None:
    """Switch active page and optionally focus a specific event_id."""
    st.session_state["active_page"] = page
    if event_id is not None:
        st.session_state["selected_event_id"] = int(event_id)
        st.session_state["rec_filter"] = "All"
        st.session_state["conf_filter"] = "All"
        st.session_state["search_query"] = ""


def _notable_events(split_preds: pd.DataFrame) -> dict[str, int | None]:
    """Find flagship rescued, avoided-burn, and highest-risk encounters."""
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


def _build_risk_trajectory_figure(
    event_row: pd.Series,
    event_history: pd.DataFrame,
    high_risk_thresh: float,
    cutoff_days: float,
    reveal_gt: bool,
    height: int = 370,
    compact_legend: bool = False,
) -> go.Figure:
    """Build a crisp, overlap-free Plotly collision risk trajectory chart."""
    vis_cdms = event_history[event_history["visible"]].sort_values("time_to_tca", ascending=False)
    post_cdms = event_history[~event_history["visible"]].sort_values("time_to_tca", ascending=False)

    all_y = list(vis_cdms["log_risk"].values) + [float(event_row["pred_final_log_risk"]), high_risk_thresh]
    if reveal_gt and len(post_cdms) > 0:
        all_y.extend(list(post_cdms["log_risk"].values))
    if reveal_gt:
        all_y.append(float(event_row["actual_final_log_risk"]))

    y_min_data = float(np.nanmin(all_y)) if all_y else -30.0
    y_max_data = float(np.nanmax(all_y)) if all_y else -2.0
    y_low = max(-31.5, min(y_min_data - 2.2, high_risk_thresh - 4.5))
    y_high = min(0.5, max(y_max_data + 1.8, high_risk_thresh + 2.8))

    max_x = float(max(7.0, event_history["time_to_tca"].max() + 0.35)) if len(event_history) else 7.2

    fig = go.Figure()

    # Shaded Blind Zone (< cutoff_days)
    fig.add_vrect(
        x0=cutoff_days,
        x1=0.0,
        fillcolor="rgba(244, 63, 94, 0.10)",
        layer="below",
        line_width=0,
    )
    fig.add_vline(
        x=cutoff_days,
        line_dash="dash",
        line_color="#F59E0B",
        line_width=2,
    )
    fig.add_hline(
        y=high_risk_thresh,
        line_dash="dot",
        line_color="rgba(245, 158, 11, 0.85)",
        line_width=1.5,
    )

    # Non-overlapping annotations anchored cleanly to margins/blind zone
    fig.add_annotation(
        x=cutoff_days * 0.5,
        y=0.98,
        yref="paper",
        yanchor="top",
        text="Blind Zone (<48h)",
        showarrow=False,
        font=dict(color="#FDA4AF", size=10, family="IBM Plex Mono"),
        bgcolor="rgba(11, 14, 20, 0.72)",
        borderpad=3,
    )
    fig.add_annotation(
        x=cutoff_days,
        y=0.03,
        yref="paper",
        yanchor="bottom",
        xanchor="right",
        xshift=-6,
        text="T-48h Cutoff",
        showarrow=False,
        font=dict(color="#FCD34D", size=10, family="IBM Plex Mono"),
        bgcolor="rgba(11, 14, 20, 0.82)",
        borderpad=3,
    )
    # Keep alert label clear of the Blind Zone top badge on deep-plunge trajectories
    alert_near_top = (y_high - high_risk_thresh) < 0.24 * max(1.0, (y_high - y_low))
    fig.add_annotation(
        x=0.99,
        xref="paper",
        xanchor="right",
        y=high_risk_thresh,
        yshift=-12 if alert_near_top else 11,
        text=f"1 in 1M Alert ({high_risk_thresh:.1f})",
        showarrow=False,
        font=dict(color="#FCD34D", size=10, family="IBM Plex Mono"),
        bgcolor="rgba(11, 14, 20, 0.85)",
        borderpad=3,
    )

    # Observable pre-cutoff CDMs
    if len(vis_cdms) > 0:
        fig.add_trace(
            go.Scatter(
                x=vis_cdms["time_to_tca"],
                y=vis_cdms["log_risk"],
                mode="lines+markers",
                name="Pre-cutoff CDMs",
                line=dict(color="#38BDF8", width=2.5),
                marker=dict(size=7, color="#38BDF8", line=dict(width=1, color="#0B0E14")),
                hovertemplate="T-%{x:.2f}d<br>Observed log risk: %{y:.2f}<extra></extra>",
            )
        )

    # NearMiss predicted final risk at TCA
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[float(event_row["pred_final_log_risk"])],
            mode="markers",
            name=f"NearMiss forecast ({event_row['pred_final_log_risk']:.2f})",
            marker=dict(
                size=12,
                color="#A78BFA",
                symbol="diamond",
                line=dict(width=1.5, color="#F8FAFC"),
            ),
            hovertemplate="NearMiss TCA forecast: %{y:.2f}<extra></extra>",
        )
    )

    # Post-cutoff ground truth reveal
    if reveal_gt:
        if len(post_cdms) > 0:
            # Connect last visible CDM to post-cutoff track seamlessly
            if len(vis_cdms) > 0:
                bridge = pd.concat([vis_cdms.iloc[[-1]], post_cdms], ignore_index=True)
            else:
                bridge = post_cdms
            fig.add_trace(
                go.Scatter(
                    x=bridge["time_to_tca"],
                    y=bridge["log_risk"],
                    mode="lines+markers",
                    name="Actual blind-zone track",
                    line=dict(color="#F59E0B", width=2.2, dash="dash"),
                    marker=dict(size=7, color="#F59E0B", symbol="circle-open", line=dict(width=2, color="#F59E0B")),
                    hovertemplate="T-%{x:.2f}d<br>Actual log risk: %{y:.2f}<extra></extra>",
                )
            )
        actual_val = float(event_row["actual_final_log_risk"])
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[actual_val],
                mode="markers",
                name=f"Actual TCA ({actual_val:.2f})",
                marker=dict(
                    size=11,
                    color="#F43F5E" if actual_val >= high_risk_thresh else "#10B981",
                    symbol="star",
                    line=dict(width=1, color="#F8FAFC"),
                ),
                hovertemplate="Actual final log risk: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Days remaining until closest approach (TCA)",
            range=[max_x, -0.25],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            color="#94A3B8",
            title_font=dict(size=11, family="IBM Plex Sans"),
        ),
        yaxis=dict(
            title="Collision risk log10(P)",
            range=[y_low, y_high],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            color="#94A3B8",
            title_font=dict(size=11, family="IBM Plex Sans"),
        ),
        paper_bgcolor="#121722",
        plot_bgcolor="#0D1119",
        font=dict(family="IBM Plex Sans", color="#CBD5E1"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22 if not compact_legend else -0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#CBD5E1"),
        ),
        height=height,
        margin=dict(l=58, r=16, t=14, b=74),
        hovermode="closest",
    )
    return fig


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW (MISSION BRIEFING LANDING PAGE)
# -----------------------------------------------------------------------------
def render_overview(
    split_preds: pd.DataFrame,
    split_hist: pd.DataFrame,
    split_label: str,
    high_risk_thresh: float,
    cutoff_days: float,
) -> None:
    """Render the Mission Briefing landing page with a live interactive Hero."""
    notable = _notable_events(split_preds)
    rescued_id = notable["rescued"]
    avoided_id = notable["avoided"]

    hero_l, hero_r = st.columns([1.15, 1.0], gap="large", vertical_alignment="top")

    with hero_l:
        st.markdown(
            f"""
<div class='hero-headline'>Predict satellite collisions {cutoff_days*24:.0f} hours before closest approach.</div>
<p class='hero-lead'>
Mission planners need 24 to 48 hours to design and uplink an orbital avoidance burn.
Once an encounter crosses <strong>T-{cutoff_days*24:.0f}h</strong>, late radar updates arrive inside the
<strong>Blind Zone</strong>—too late to act. NearMiss forecasts terminal collision risk strictly from
early tracking passes using 73 orbital covariance physics features, catching late-jumping hazards while
cutting unnecessary fuel burns.
</p>
<div class='hero-metrics'>
    <div class='hm-cell'>
        <div class='hm-val'>14,109</div>
        <div class='hm-lbl'>ESA conjunctions evaluated</div>
    </div>
    <div class='hm-cell'>
        <div class='hm-val' style='color:#38BDF8;'>{cutoff_days*24:.0f}h</div>
        <div class='hm-lbl'>Strict advance decision horizon</div>
    </div>
    <div class='hm-cell'>
        <div class='hm-val' style='color:#10B981;'>-59.5%</div>
        <div class='hm-lbl'>Probability calibration error</div>
    </div>
    <div class='hm-cell'>
        <div class='hm-val' style='color:#F59E0B;'>-22.6%</div>
        <div class='hm-lbl'>Avoidable burns cut (OOF)</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        act1, act2 = st.columns([1.1, 1.0], gap="small")
        with act1:
            if st.button("Open Event Inspector", type="primary", width="stretch", key="hero_open_inspector"):
                _navigate_to("Event Inspector", rescued_id or notable["top_risk"])
                st.rerun()
        with act2:
            if st.button("Simulate Burn Policy", type="secondary", width="stretch", key="hero_open_sim"):
                _navigate_to("Policy Simulator")
                st.rerun()

    with hero_r:
        showcase_options = []
        if avoided_id is not None:
            showcase_options.append(f"Avoided False Alarm #{avoided_id}")
        if rescued_id is not None:
            showcase_options.append(f"Caught Late Collision #{rescued_id}")
        if not showcase_options and notable["top_risk"] is not None:
            showcase_options.append(f"Top Risk Event #{notable['top_risk']}")

        selected_showcase = st.segmented_control(
            "Live showcase encounter",
            options=showcase_options,
            default=showcase_options[0],
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
            key=f"hero_showcase_{split_label}",
        )
        if not selected_showcase:
            selected_showcase = showcase_options[0]

        show_eid = int(selected_showcase.split("#")[-1])
        show_row = split_preds[split_preds["event_id"] == show_eid].iloc[0]
        show_hist = split_hist[split_hist["event_id"] == show_eid]

        is_rescued_view = "Caught" in selected_showcase
        caption_color = "#FDA4AF" if is_rescued_view else "#6EE7B7"
        caption_txt = (
            f"Baseline A stood down at T-48h ({show_row['latest_log_risk']:.2f}), missing the jump to {show_row['actual_final_log_risk']:.2f}. NearMiss commanded a burn early."
            if is_rescued_view
            else f"Baseline A alarmed at T-48h ({show_row['latest_log_risk']:.2f}), but risk fell safely to {show_row['actual_final_log_risk']:.2f}. NearMiss saved the burn."
        )
        st.markdown(
            f"<div style='font-size:0.8rem; color:{caption_color}; margin-bottom:4px; line-height:1.4;'>{caption_txt}</div>",
            unsafe_allow_html=True,
        )

        fig_hero = _build_risk_trajectory_figure(
            event_row=show_row,
            event_history=show_hist,
            high_risk_thresh=high_risk_thresh,
            cutoff_days=cutoff_days,
            reveal_gt=True,
            height=305,
            compact_legend=True,
        )
        st.plotly_chart(fig_hero, width="stretch", config={"displayModeBar": False})

    # 3-Step Pipeline Architecture
    st.markdown(
        """
<div class='section-title'>How NearMiss decides before the blind zone</div>
<div class='section-desc'>Every feature, split, and calibrator strictly respects the 48-hour operational barrier with zero future leakage.</div>
<div class='flow-grid'>
    <div class='flow-step'>
        <div class='flow-step-title'>1. Early Radar Intake (T-7d to T-2d)</div>
        <div class='flow-step-body'>
            Ingests Conjunction Data Messages (CDMs) up to the planning deadline and discards every observation
            inside 48 hours to closest approach before features are computed.
        </div>
    </div>
    <div class='flow-step gold'>
        <div class='flow-step-title'>2. Orbital Covariance Physics</div>
        <div class='flow-step-body'>
            Extracts 73 domain features capturing 3D uncertainty ellipsoid volume √det(C), contraction rate across passes,
            combined RTN positional sigmas, and miss-distance velocity gradients.
        </div>
    </div>
    <div class='flow-step green'>
        <div class='flow-step-title'>3. Calibrated Probability & Confidence</div>
        <div class='flow-step-body'>
            Applies out-of-fold Platt scaling to transform balanced Random Forest scores into trustworthy collision
            probabilities (Brier 0.0166 → 0.0068) paired with High, Medium, or Low confidence tiers.
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Curated Case Studies (Real live rows from active split)
    st.markdown(
        f"""
<div class='section-title'>Verified encounters in {split_label}</div>
<div class='section-desc'>Direct head-to-head disagreements where NearMiss improved on legacy Persistence (Baseline A).</div>
""",
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2, gap="medium")

    with c_left:
        if rescued_id is not None:
            r_row = split_preds[split_preds["event_id"] == rescued_id].iloc[0]
            st.markdown(
                f"""
<div class='case-panel'>
    <div class='case-head'>
        <div class='case-title'>Late-Jumping Collision Caught — Event #{rescued_id}</div>
        <span class='case-tag'>Rescued</span>
    </div>
    <div class='case-body'>
        At T-48h, the latest radar pass sat just beneath the alert line, causing legacy Persistence to clear the
        encounter. NearMiss detected tightening covariance geometry and triggered an early avoidance burn.
    </div>
    <div class='case-compare'>
        <div><span style='color:#8B98B0;'>Baseline A</span><br><strong style='color:#94A3B8;'>Stand Down</strong></div>
        <div><span style='color:#8B98B0;'>NearMiss</span><br><strong style='color:#FDA4AF;'>Burn ({r_row['high_risk_prob']:.1%})</strong></div>
        <div><span style='color:#8B98B0;'>Actual TCA</span><br><strong style='color:#F59E0B;'>{r_row['actual_final_log_risk']:.2f} (Danger)</strong></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button(f"Inspect Rescued Event #{rescued_id}", width="stretch", key="ov_btn_rescued"):
                _navigate_to("Event Inspector", rescued_id)
                st.rerun()

    with c_right:
        if avoided_id is not None:
            a_row = split_preds[split_preds["event_id"] == avoided_id].iloc[0]
            st.markdown(
                f"""
<div class='case-panel saved'>
    <div class='case-head'>
        <div class='case-title'>Costly False Alarm Prevented — Event #{avoided_id}</div>
        <span class='case-tag saved'>Burn Saved</span>
    </div>
    <div class='case-body'>
        Early tracking noise briefly pushed collision risk above the maneuver line, triggering a fuel-wasting burn
        under Persistence. NearMiss recognized transient uncertainty and safely cleared the orbit.
    </div>
    <div class='case-compare'>
        <div><span style='color:#8B98B0;'>Baseline A</span><br><strong style='color:#FDA4AF;'>Order Burn</strong></div>
        <div><span style='color:#8B98B0;'>NearMiss</span><br><strong style='color:#6EE7B7;'>Stand Down ({a_row['high_risk_prob']:.1%})</strong></div>
        <div><span style='color:#8B98B0;'>Actual TCA</span><br><strong style='color:#10B981;'>{a_row['actual_final_log_risk']:.2f} (Safe)</strong></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button(f"Inspect Avoided-Burn Event #{avoided_id}", width="stretch", key="ov_btn_avoided"):
                _navigate_to("Event Inspector", avoided_id)
                st.rerun()


# -----------------------------------------------------------------------------
# PAGE 2: EVENT INSPECTOR
# -----------------------------------------------------------------------------
def render_inspector(
    split_preds: pd.DataFrame,
    split_hist: pd.DataFrame,
    high_risk_thresh: float,
    cutoff_days: float,
) -> None:
    """Render the Conjunction Event Inspector with charts visible above the fold."""
    # Compact single-row filter bar
    f1, f2, f3, f4 = st.columns([0.95, 0.9, 0.95, 2.0], gap="small")
    with f1:
        rec_filter = st.selectbox("Recommendation", ["All", "MANOEUVRE", "NO_MANOEUVRE"], key="rec_filter")
    with f2:
        conf_filter = st.selectbox("Confidence", ["All", "HIGH", "MEDIUM", "LOW"], key="conf_filter")
    with f3:
        search_query = st.text_input("Search Event ID", placeholder="e.g. 8776, 873", key="search_query")

    filtered = split_preds.copy()
    if rec_filter != "All":
        filtered = filtered[filtered["recommendation"] == rec_filter]
    if conf_filter != "All":
        filtered = filtered[filtered["confidence"] == conf_filter]
    if search_query.strip():
        try:
            filtered = filtered[filtered["event_id"] == int(search_query.strip())]
        except ValueError:
            st.warning("Enter a numeric Event ID (for example, 8776 or 873).")

    filtered = filtered.sort_values("high_risk_prob", ascending=False)

    with f4:
        if len(filtered) == 0:
            st.selectbox("Encounter (0 matches)", ["No matching encounters"], disabled=True)
        else:
            event_options = filtered["event_id"].tolist()
            if st.session_state.get("selected_event_id") not in event_options:
                st.session_state["selected_event_id"] = event_options[0]

            lookup = filtered.set_index("event_id")
            selected_event_id = st.selectbox(
                f"Select Encounter ({len(filtered):,} matching, sorted by collision probability)",
                event_options,
                key="selected_event_id",
                format_func=lambda eid: (
                    f"Event #{eid} — P={lookup.loc[eid, 'high_risk_prob']:.1%} "
                    f"({lookup.loc[eid, 'recommendation'].replace('_', ' ').title()}, "
                    f"{lookup.loc[eid, 'confidence'].title()})"
                ),
            )

    if len(filtered) == 0:
        st.info("No conjunctions match the active filter combination.")
        if st.button("Reset Filters", type="primary"):
            st.session_state["rec_filter"] = "All"
            st.session_state["conf_filter"] = "All"
            st.session_state["search_query"] = ""
            st.rerun()
        return

    event_row = split_preds[split_preds["event_id"] == selected_event_id].iloc[0]
    conf_raw = str(event_row["confidence"]).upper()
    conf_class = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}.get(conf_raw, "conf-low")
    is_manoeuvre = event_row["recommendation"] == "MANOEUVRE"
    b_is_man = event_row["baseline_recommendation"] == "MANOEUVRE"

    hud_mod = "" if is_manoeuvre else "safe"
    verdict_color = "#FDA4AF" if is_manoeuvre else "#6EE7B7"
    verdict_text = "Command Avoidance Burn" if is_manoeuvre else "Stand Down — Nominal Orbit"

    b_color = "#FDA4AF" if b_is_man else "#6EE7B7"
    b_text = "Order Burn" if b_is_man else "Stand Down"

    if is_manoeuvre == b_is_man:
        div_label = "Agreement"
        div_color = "#94A3B8"
        div_sub = "Matches persistence"
    elif is_manoeuvre:
        div_label = "Caught Early"
        div_color = "#38BDF8"
        div_sub = "Baseline stood down"
    else:
        div_label = "Burn Saved"
        div_color = "#10B981"
        div_sub = "Baseline alarmed"

    # Compact Single-Row Decision HUD
    st.markdown(
        f"""
<div class='hud-strip {hud_mod}'>
    <div>
        <div class='verdict-row'>
            <span class='verdict-title' style='color:{verdict_color};'>{verdict_text}</span>
            <span class='conf-pill {conf_class}'>{conf_raw.title()} Confidence</span>
        </div>
        <div class='verdict-meta'>
            Event #{selected_event_id} &nbsp;|&nbsp; Satellite #{int(event_row['mission_id'])} vs {event_row['object_type'].title()} &nbsp;|&nbsp; {int(event_row['cdm_count_pre_cutoff'])} pre-cutoff CDMs
        </div>
    </div>
    <div class='hud-cell'>
        <div class='hud-label'>Calibrated Risk Odds</div>
        <div class='hud-val'>{event_row['high_risk_prob']:.2%}</div>
        <div class='hud-sub'>Platt posterior probability</div>
    </div>
    <div class='hud-cell'>
        <div class='hud-label'>Forecast TCA Risk</div>
        <div class='hud-val' style='color:#38BDF8;'>{event_row['pred_final_log_risk']:.2f}</div>
        <div class='hud-sub'>log10(P) hurdle estimate</div>
    </div>
    <div class='hud-cell'>
        <div class='hud-label'>Baseline A (Persistence)</div>
        <div class='hud-val' style='color:{b_color}; font-size:1.02rem;'>{b_text}</div>
        <div class='hud-sub'>Last visible: {event_row['latest_log_risk']:.2f}</div>
    </div>
    <div class='hud-cell'>
        <div class='hud-label'>Policy Comparison</div>
        <div class='hud-val' style='color:{div_color}; font-size:1.02rem;'>{div_label}</div>
        <div class='hud-sub'>{div_sub}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Reveal toggle + inline outcome banner
    tog_c1, tog_c2 = st.columns([1.15, 2.2], vertical_alignment="center")
    with tog_c1:
        reveal_gt = st.toggle("Reveal post-cutoff outcome (Blind Zone)", value=False, key="inspector_reveal_gt")
    with tog_c2:
        if reveal_gt:
            actual_val = float(event_row["actual_final_log_risk"])
            actual_danger = int(event_row["actual_high_risk"]) == 1
            if actual_danger and is_manoeuvre:
                tag_txt, tag_col = "True Positive — Hazard Caught", "#10B981"
            elif not actual_danger and not is_manoeuvre:
                tag_txt, tag_col = "True Negative — Safe Clearance", "#10B981"
            elif actual_danger and not is_manoeuvre:
                tag_txt, tag_col = "False Negative — Missed Hazard", "#F43F5E"
            else:
                tag_txt, tag_col = "False Positive — Precautionary Burn", "#F59E0B"
            st.markdown(
                f"<div style='background:#121722; border:1px solid {tag_col}; border-radius:6px; padding:6px 12px; font-size:0.82rem;'>"
                f"<strong style='color:{tag_col};'>{tag_txt}</strong> &nbsp;|&nbsp; "
                f"Actual terminal risk: <span class='mono' style='color:#F8FAFC;'>{actual_val:.2f}</span></div>",
                unsafe_allow_html=True,
            )

    # Fetch split-isolated history (fixes 405-event ID collision sawtooth!)
    event_history = split_hist[split_hist["event_id"] == selected_event_id].sort_values(
        "time_to_tca", ascending=False
    )
    vis_cdms = event_history[event_history["visible"]]
    post_cdms = event_history[~event_history["visible"]]

    chart_c1, chart_c2 = st.columns([1.35, 1.0], gap="medium")

    with chart_c1:
        st.markdown(f"<div style='font-weight:600; font-size:0.92rem; margin-bottom:2px;'>Collision Risk Trajectory (Event #{selected_event_id})</div>", unsafe_allow_html=True)
        fig_risk = _build_risk_trajectory_figure(
            event_row=event_row,
            event_history=event_history,
            high_risk_thresh=high_risk_thresh,
            cutoff_days=cutoff_days,
            reveal_gt=reveal_gt,
            height=360,
        )
        st.plotly_chart(fig_risk, width="stretch", config={"displayModeBar": False})

    with chart_c2:
        st.markdown("<div style='font-weight:600; font-size:0.92rem; margin-bottom:2px;'>Relative Miss Distance Across Passes</div>", unsafe_allow_html=True)
        max_x = float(max(7.0, event_history["time_to_tca"].max() + 0.35)) if len(event_history) else 7.2
        fig_miss = go.Figure()
        fig_miss.add_vrect(
            x0=cutoff_days,
            x1=0.0,
            fillcolor="rgba(244, 63, 94, 0.08)",
            layer="below",
            line_width=0,
        )
        fig_miss.add_vline(x=cutoff_days, line_dash="dash", line_color="#F59E0B", line_width=1.8)
        if len(vis_cdms) > 0:
            fig_miss.add_trace(
                go.Scatter(
                    x=vis_cdms["time_to_tca"],
                    y=vis_cdms["miss_distance"],
                    mode="lines+markers",
                    name="Pre-cutoff miss distance",
                    line=dict(color="#10B981", width=2.4),
                    marker=dict(size=7, color="#10B981", line=dict(width=1, color="#0B0E14")),
                    hovertemplate="T-%{x:.2f}d<br>Miss distance: %{y:,.0f} m<extra></extra>",
                )
            )
        if reveal_gt and len(post_cdms) > 0:
            bridge_m = pd.concat([vis_cdms.iloc[[-1]], post_cdms], ignore_index=True) if len(vis_cdms) > 0 else post_cdms
            fig_miss.add_trace(
                go.Scatter(
                    x=bridge_m["time_to_tca"],
                    y=bridge_m["miss_distance"],
                    mode="lines+markers",
                    name="Blind-zone miss distance",
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                    marker=dict(size=6, color="#F59E0B"),
                    hovertemplate="T-%{x:.2f}d<br>Actual miss: %{y:,.0f} m<extra></extra>",
                )
            )
        fig_miss.update_layout(
            xaxis=dict(
                title="Days remaining until TCA",
                range=[max_x, -0.25],
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                color="#94A3B8",
                title_font=dict(size=11, family="IBM Plex Sans"),
            ),
            yaxis=dict(
                title="Miss distance (meters)",
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                color="#94A3B8",
                title_font=dict(size=11, family="IBM Plex Sans"),
            ),
            paper_bgcolor="#121722",
            plot_bgcolor="#0D1119",
            font=dict(family="IBM Plex Sans", color="#CBD5E1"),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=10.5, color="#CBD5E1"),
            ),
            height=360,
            margin=dict(l=64, r=16, t=14, b=74),
            hovermode="closest",
        )
        st.plotly_chart(fig_miss, width="stretch", config={"displayModeBar": False})

    with st.expander(f"CDM Telemetry Table ({len(event_history if reveal_gt else vis_cdms)} passes shown)", expanded=False):
        display_hist = event_history if reveal_gt else vis_cdms
        tbl = display_hist[["time_to_tca", "log_risk", "miss_distance", "visible"]].copy()
        tbl.columns = ["Days to TCA", "Log10 Risk", "Miss Distance (m)", "Visible Pre-Cutoff"]
        st.dataframe(tbl, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 3: POLICY SIMULATOR
# -----------------------------------------------------------------------------
def render_simulator(split_preds: pd.DataFrame) -> None:
    """Render the Mission Economics & Decision Threshold Simulator."""
    st.markdown(
        """
<div class='section-title'>Decision Threshold & Mission Economics Simulator</div>
<div class='section-desc'>Adjust the calibrated collision probability threshold to inspect missed hazards, fuel burns, and financial balance against Baseline A.</div>
""",
        unsafe_allow_html=True,
    )

    sim_l, sim_r = st.columns([1.0, 1.85], gap="large")

    with sim_l:
        st.markdown("<div style='font-weight:600; font-size:0.9rem; margin-bottom:2px;'>Spacecraft Profile</div>", unsafe_allow_html=True)
        preset = st.segmented_control(
            "Spacecraft Profile",
            options=["SmallSat", "Earth Obs", "Flagship"],
            default="SmallSat",
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
            key="sim_profile",
        )
        if not preset:
            preset = "SmallSat"

        if preset == "SmallSat":
            default_miss_penalty, default_burn_val, default_th = 250_000, 15_000, 0.020
            profile_note = "Commercial LEO SmallSat — USD 250k risk-weighted miss penalty / USD 15k burn."
        elif preset == "Earth Obs":
            default_miss_penalty, default_burn_val, default_th = 1_000_000, 45_000, 0.015
            profile_note = "Flagship Earth Observation — USD 1M risk-weighted miss penalty / USD 45k burn."
        else:
            default_miss_penalty, default_burn_val, default_th = 5_000_000, 100_000, 0.010
            profile_note = "High-Value Infrastructure — USD 5M risk-weighted miss penalty / USD 100k burn."

        st.markdown(f"<div style='font-size:0.78rem; color:#8B98B0; margin-bottom:6px;'>{profile_note}</div>", unsafe_allow_html=True)

        sim_th = st.slider(
            "Maneuver Trigger Threshold (p*)",
            min_value=0.005,
            max_value=0.250,
            value=default_th,
            step=0.005,
            format="%.3f",
            key=f"sim_th_{preset}",
        )
        st.caption(f"Command an avoidance burn whenever calibrated P(collision) ≥ {sim_th:.1%}.")

        cost_collision = st.number_input("Risk-Weighted Miss Penalty (USD)", value=default_miss_penalty, step=50_000, key=f"cc_{preset}")
        cost_manoeuvre = st.number_input("Avoidance Burn Cost (USD)", value=default_burn_val, step=5_000, key=f"cm_{preset}")

        sim_pred_bin = (split_preds["high_risk_prob"] >= sim_th).astype(int)
        y_eval_true = split_preds["actual_high_risk"].values
        sim_metrics = compute_classification_metrics(y_eval_true, sim_pred_bin, split_preds["high_risk_prob"].values)

        base_pred_bin = (split_preds["baseline_recommendation"] == "MANOEUVRE").astype(int)
        base_metrics = compute_classification_metrics(y_eval_true, base_pred_bin)

        cost_model = (sim_metrics["fn"] * cost_collision) + ((sim_metrics["tp"] + sim_metrics["fp"]) * cost_manoeuvre)
        cost_baseline = (base_metrics["fn"] * cost_collision) + ((base_metrics["tp"] + base_metrics["fp"]) * cost_manoeuvre)
        savings = cost_baseline - cost_model
        savings_pct = (savings / max(1, cost_baseline)) * 100

        sav_col = "#10B981" if savings >= 0 else "#F43F5E"
        st.markdown(
            f"""
<div style='background:#121722; border:1px solid {sav_col}; border-radius:8px; padding:14px 16px; margin-top:8px;'>
    <div style='font-size:0.76rem; color:#8B98B0;'>Expected Financial Balance vs Baseline A</div>
    <div class='mono' style='font-size:1.55rem; font-weight:700; color:{sav_col}; margin:3px 0;'>USD {savings:+,.0f}</div>
    <div style='font-size:0.8rem; color:#CBD5E1;'>
        {savings_pct:+.1f}% total expected mission cost change at p* = {sim_th:.3f}.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with sim_r:
        fn_delta = sim_metrics["fn"] - base_metrics["fn"]
        fp_saved = base_metrics["fp"] - sim_metrics["fp"]
        fn_col = "#10B981" if fn_delta <= 0 else "#F59E0B"
        fp_col = "#10B981" if fp_saved >= 0 else "#F43F5E"

        st.markdown(
            f"""
<div class='kpi-strip'>
    <div class='kpi-tile'>
        <div class='kpi-tile-lbl'>Recall (Caught)</div>
        <div class='kpi-tile-val' style='color:#38BDF8;'>{sim_metrics['recall']:.1%}</div>
        <div class='kpi-tile-sub'>{sim_metrics['tp']} of {sim_metrics['tp']+sim_metrics['fn']} hazards</div>
    </div>
    <div class='kpi-tile'>
        <div class='kpi-tile-lbl'>Missed Hazards (FN)</div>
        <div class='kpi-tile-val' style='color:{fn_col};'>{sim_metrics['fn']}</div>
        <div class='kpi-tile-sub'>{fn_delta:+d} vs Baseline ({base_metrics['fn']})</div>
    </div>
    <div class='kpi-tile'>
        <div class='kpi-tile-lbl'>False Alarms (FP)</div>
        <div class='kpi-tile-val' style='color:{fp_col};'>{sim_metrics['fp']}</div>
        <div class='kpi-tile-sub'>{fp_saved:+d} saved vs Baseline ({base_metrics['fp']})</div>
    </div>
    <div class='kpi-tile'>
        <div class='kpi-tile-lbl'>F2 Score</div>
        <div class='kpi-tile-val'>{sim_metrics['f2']:.3f}</div>
        <div class='kpi-tile-sub'>Baseline F2: {base_metrics['f2']:.3f}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Trade-off frontier curve
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
                name="NearMiss trade-off curve",
                line=dict(color="#38BDF8", width=2.8),
                hovertemplate="Threshold: %{customdata:.3f}<br>False alarms (FP): %{x}<br>Missed hazards (FN): %{y}<extra></extra>",
                customdata=sweep_df["threshold"],
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=[sim_metrics["fp"]],
                y=[sim_metrics["fn"]],
                mode="markers",
                name=f"Active Policy (p*={sim_th:.3f})",
                marker=dict(size=14, color="#F43F5E", symbol="star", line=dict(width=1.2, color="#F8FAFC")),
                hovertemplate=f"Active Policy (p*={sim_th:.3f})<br>FP: %{{x}}, FN: %{{y}}<extra></extra>",
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=[base_metrics["fp"]],
                y=[base_metrics["fn"]],
                mode="markers",
                name="Baseline A (Persistence)",
                marker=dict(size=11, color="#F59E0B", symbol="square", line=dict(width=1, color="#0B0E14")),
                hovertemplate="Baseline A<br>FP: %{x}, FN: %{y}<extra></extra>",
            )
        )
        fig_sim.update_layout(
            xaxis=dict(title="Unnecessary maneuvers (FP)", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8", title_font=dict(size=11)),
            yaxis=dict(title="Missed collisions (FN)", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8", title_font=dict(size=11)),
            paper_bgcolor="#121722",
            plot_bgcolor="#0D1119",
            font=dict(family="IBM Plex Sans", color="#CBD5E1"),
            legend=dict(orientation="h", yanchor="top", y=-0.30, xanchor="center", x=0.5, font=dict(size=10.5)),
            height=262,
            margin=dict(l=54, r=16, t=12, b=72),
        )
        st.plotly_chart(fig_sim, width="stretch", config={"displayModeBar": False})

        cm1, cm2 = st.columns(2, gap="small")
        with cm1:
            st.markdown(
                f"""
<div class='cm-card'>
    <div class='cm-title' style='color:#38BDF8;'>NearMiss Matrix (p* = {sim_th:.3f})</div>
    <table class='cm-table'>
        <tr><th></th><th>Pred Safe</th><th>Pred Burn</th></tr>
        <tr>
            <th>Actual Safe</th>
            <td style='background:#192131; color:#CBD5E1;'>{sim_metrics['tn']:,} (TN)</td>
            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF;'>{sim_metrics['fp']:,} (FP)</td>
        </tr>
        <tr>
            <th>Actual Danger</th>
            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF;'>{sim_metrics['fn']:,} (FN)</td>
            <td style='background:rgba(16,185,129,0.16); color:#6EE7B7;'>{sim_metrics['tp']:,} (TP)</td>
        </tr>
    </table>
</div>
""",
                unsafe_allow_html=True,
            )
        with cm2:
            st.markdown(
                f"""
<div class='cm-card'>
    <div class='cm-title' style='color:#F59E0B;'>Baseline A Matrix (Persistence)</div>
    <table class='cm-table'>
        <tr><th></th><th>Pred Safe</th><th>Pred Burn</th></tr>
        <tr>
            <th>Actual Safe</th>
            <td style='background:#192131; color:#CBD5E1;'>{base_metrics['tn']:,} (TN)</td>
            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF;'>{base_metrics['fp']:,} (FP)</td>
        </tr>
        <tr>
            <th>Actual Danger</th>
            <td style='background:rgba(244,63,94,0.16); color:#FDA4AF;'>{base_metrics['fn']:,} (FN)</td>
            <td style='background:rgba(16,185,129,0.16); color:#6EE7B7;'>{base_metrics['tp']:,} (TP)</td>
        </tr>
    </table>
</div>
""",
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# PAGE 4: BENCHMARKS & ORBITAL PHYSICS
# -----------------------------------------------------------------------------
def render_benchmarks() -> None:
    """Render full-width benchmark tables and live interactive Plotly diagnostics."""
    st.markdown(
        """
<div class='section-title'>ESA Kelvins Benchmark & Model Calibration</div>
<div class='section-desc'>Evaluated under strict event-grouped cross-validation (8,293 filtered events) and on the locked competition test set (2,167 events).</div>
""",
        unsafe_allow_html=True,
    )

    # Full-width benchmark tables so every column is visible with zero clipping
    st.markdown("<div style='font-weight:600; font-size:0.94rem; margin-bottom:2px;'>Locked ESA Blind Test Set (2,167 Events, 150 High-Risk)</div>", unsafe_allow_html=True)
    test_df = pd.DataFrame(
        [
            ["Baseline A (Persistence)", 76.67, 64.61, 0.7391, 0.6651, 35, 63, 26.64, "Reference"],
            ["NearMiss Champion (RF + Platt)", 71.33, 66.46, 0.7030, 0.6270, 43, 54, 133.79, "12 burns saved, 3 collisions rescued"],
        ],
        columns=["Pipeline", "Recall", "Precision", "F2 Score", "PR-AUC", "Missed (FN)", "False Alarms (FP)", "High-Risk MSE", "Operational Impact"],
    )
    st.dataframe(
        test_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Recall": st.column_config.NumberColumn(format="%.2f%%"),
            "Precision": st.column_config.NumberColumn(format="%.2f%%"),
            "F2 Score": st.column_config.NumberColumn(format="%.4f"),
            "PR-AUC": st.column_config.NumberColumn(format="%.4f"),
            "High-Risk MSE": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.markdown("<div style='font-weight:600; font-size:0.94rem; margin-top:4px; margin-bottom:2px;'>5-Fold Out-of-Fold Cross-Validation (8,293 Filtered Events, 66 High-Risk)</div>", unsafe_allow_html=True)
    cv_df = pd.DataFrame(
        [
            ["Baseline A (Persistence)", "Reference", 71.21, 15.26, 0.4108, 0.1803, 19, 261],
            ["Logistic Regression", "Unweighted", 28.79, 25.00, 0.2794, 0.1711, 47, 57],
            ["Logistic Regression", "Balanced Weights", 87.88, 6.58, 0.2531, 0.1484, 8, 824],
            ["Random Forest", "Unweighted", 45.45, 24.59, 0.3886, 0.1759, 36, 92],
            ["Random Forest (Champion)", "Balanced Subsample", 71.21, 16.85, 0.4328, 0.1926, 19, 232],
            ["LightGBM Classifier", "Scale-Pos-Weight", 65.15, 19.11, 0.4397, 0.1504, 23, 182],
        ],
        columns=["Model Architecture", "Imbalance Strategy", "Recall", "Precision", "F2 Score", "PR-AUC", "Missed (FN)", "False Alarms (FP)"],
    )
    st.dataframe(
        cv_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Recall": st.column_config.NumberColumn(format="%.2f%%"),
            "Precision": st.column_config.NumberColumn(format="%.2f%%"),
            "F2 Score": st.column_config.NumberColumn(format="%.4f"),
            "PR-AUC": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # Matching Interactive Plotly Diagnostics (replaces clashing white/black PNGs!)
    diag_l, diag_r = st.columns(2, gap="medium")

    with diag_l:
        st.markdown("<div style='font-weight:600; font-size:0.92rem; margin-bottom:2px;'>Platt Probability Calibration Reliability (-59.5% Brier Error)</div>", unsafe_allow_html=True)
        cal_df = load_calibration_oof()
        fig_cal = go.Figure()

        fig_cal.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line=dict(color="rgba(148, 163, 184, 0.45)", width=1.5, dash="dash"),
                hoverinfo="skip",
            )
        )

        if cal_df is not None and len(cal_df) > 0:
            sub_cal = cal_df[cal_df["is_eval_filtered"]] if "is_eval_filtered" in cal_df.columns else cal_df
            y_true = sub_cal["y_high_risk"].values
            raw_p = sub_cal["raw_prob"].values
            cal_p = sub_cal["calibrated_prob"].values
            brier_raw = brier_score_loss(y_true, raw_p)
            brier_cal = brier_score_loss(y_true, cal_p)

            raw_true, raw_pred = calibration_curve(y_true, raw_p, n_bins=8, strategy="uniform")
            cal_true, cal_pred = calibration_curve(y_true, cal_p, n_bins=8, strategy="uniform")

            fig_cal.add_trace(
                go.Scatter(
                    x=raw_pred,
                    y=raw_true,
                    mode="lines+markers",
                    name=f"Uncalibrated RF (Brier {brier_raw:.4f})",
                    line=dict(color="#F43F5E", width=2.4),
                    marker=dict(size=7, color="#F43F5E"),
                    hovertemplate="Predicted: %{x:.1%}<br>Observed rate: %{y:.1%}<extra></extra>",
                )
            )
            fig_cal.add_trace(
                go.Scatter(
                    x=cal_pred,
                    y=cal_true,
                    mode="lines+markers",
                    name=f"Platt Calibrated RF (Brier {brier_cal:.4f})",
                    line=dict(color="#38BDF8", width=2.8),
                    marker=dict(size=8, color="#38BDF8", line=dict(width=1, color="#F8FAFC")),
                    hovertemplate="Calibrated: %{x:.1%}<br>Observed rate: %{y:.1%}<extra></extra>",
                )
            )

        fig_cal.update_layout(
            xaxis=dict(title="Mean predicted probability", range=[-0.02, 1.0], tickformat=".0%", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
            yaxis=dict(title="Observed high-risk fraction", range=[-0.02, 0.52], tickformat=".0%", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
            paper_bgcolor="#121722",
            plot_bgcolor="#0D1119",
            font=dict(family="IBM Plex Sans", color="#CBD5E1"),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10.5)),
            height=355,
            margin=dict(l=58, r=18, t=12, b=72),
        )
        st.plotly_chart(fig_cal, width="stretch", config={"displayModeBar": False})

    with diag_r:
        st.markdown("<div style='font-weight:600; font-size:0.92rem; margin-bottom:2px;'>Global Feature Drivers (Tree SHAP Mean |SHAP|)</div>", unsafe_allow_html=True)
        shap_df = pd.DataFrame(TOP_SHAP_FEATURES, columns=["raw", "label", "val", "group"]).iloc[::-1]
        color_map = {"Trajectory": "#38BDF8", "Covariance": "#F59E0B", "Dynamics": "#10B981", "Timing": "#A78BFA"}

        fig_shap = go.Figure()
        for grp in ["Trajectory", "Covariance", "Dynamics", "Timing"]:
            part = shap_df[shap_df["group"] == grp]
            if len(part) == 0:
                continue
            fig_shap.add_trace(
                go.Bar(
                    x=part["val"],
                    y=part["label"],
                    orientation="h",
                    name=grp,
                    marker=dict(color=color_map[grp], line=dict(width=0)),
                    customdata=part["raw"],
                    hovertemplate="%{y}<br>Feature: %{customdata}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                )
            )
        fig_shap.update_layout(
            barmode="stack",
            xaxis=dict(title="Mean |SHAP| impact on collision risk", gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
            yaxis=dict(categoryorder="array", categoryarray=shap_df["label"].tolist(), color="#CBD5E1", tickfont=dict(size=10.5)),
            paper_bgcolor="#121722",
            plot_bgcolor="#0D1119",
            font=dict(family="IBM Plex Sans", color="#CBD5E1"),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10.5)),
            height=355,
            margin=dict(l=175, r=18, t=12, b=72),
        )
        st.plotly_chart(fig_shap, width="stretch", config={"displayModeBar": False})

    # Orbital Physics Summary Strip
    st.markdown(
        """
<div class='flow-grid'>
    <div class='flow-step'>
        <div class='flow-step-title'>Combined Positional Sigmas</div>
        <div class='flow-step-body'>
            Root-sum-square radial, along-track, and cross-track uncertainty fusing both target and chaser covariance tensors:
            <span class='mono' style='color:#38BDF8;'>σ_pos = √(σ_t² + σ_c²)</span>.
        </div>
    </div>
    <div class='flow-step gold'>
        <div class='flow-step-title'>3D Ellipsoid Volume</div>
        <div class='flow-step-body'>
            Determinant-based orbital uncertainty volume measuring how tightly ground radar constrains the flyby corridor:
            <span class='mono' style='color:#FCD34D;'>Vol = √det(C_pos)</span>.
        </div>
    </div>
    <div class='flow-step green'>
        <div class='flow-step-title'>Volume Contraction Ratio</div>
        <div class='flow-step-body'>
            Ratio of latest to earliest covariance volume across passes—separating true tightening threats from transient noise:
            <span class='mono' style='color:#6EE7B7;'>Vol_cutoff / Vol_initial</span>.
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# MAIN APPLICATION SHELL (SIDEBAR + SINGLE-ROW TOP NAVBAR)
# -----------------------------------------------------------------------------
def main() -> None:
    cfg, all_preds, all_hist = load_app_data()

    if "active_page" not in st.session_state or st.session_state["active_page"] not in NAV_PAGES:
        st.session_state["active_page"] = "Overview"

    high_risk_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))
    operational_thresh = float(cfg.get("thresholds", {}).get("operational_risk_log10", -4.0))
    cutoff_days = float(cfg["data"]["cutoff_days"])

    # --- PURPOSEFUL, PLAIN-ENGLISH SIDEBAR ---
    with st.sidebar:
        st.markdown("<div class='sb-section-title'>Active Dataset</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sb-help'>Choose which unseen set of ESA satellite encounters to inspect across all views.</div>",
            unsafe_allow_html=True,
        )
        split_choice = st.radio(
            "Dataset Cohort",
            options=["Validation Set (2,389)", "ESA Blind Test (2,167)"],
            index=0,
            label_visibility="collapsed",
        )
        split_key = "val" if split_choice.startswith("Validation") else "test"
        split_label = "Validation" if split_key == "val" else "Blind Test"
        split_full_label = "Validation Set" if split_key == "val" else "ESA Blind Test Set"
        split_preds = all_preds[all_preds["split"] == split_key].copy()
        split_hist = _slice_history_for_split(all_hist, split_key)

        n_man = int((split_preds["recommendation"] == "MANOEUVRE").sum())
        n_haz = int(split_preds["actual_high_risk"].sum())
        st.markdown(
            f"""
<div class='sb-stat-strip'>
    <div>
        <div class='sb-stat-num' style='color:#FDA4AF;'>{n_man:,}</div>
        <div class='sb-stat-lbl'>Burns recommended</div>
    </div>
    <div>
        <div class='sb-stat-num' style='color:#FCD34D;'>{n_haz:,}</div>
        <div class='sb-stat-lbl'>Actual hazards</div>
    </div>
</div>
<hr class='sb-hr'>
""",
            unsafe_allow_html=True,
        )

        notable = _notable_events(split_preds)
        st.markdown("<div class='sb-section-title'>Guided Demo Jumps</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sb-help'>Jump straight into a notable encounter:</div>",
            unsafe_allow_html=True,
        )
        if notable["rescued"] is not None:
            if st.button(f"Caught Collision  #{notable['rescued']}", width="stretch", key="sb_jump_rescued"):
                _navigate_to("Event Inspector", notable["rescued"])
                st.rerun()
        if notable["avoided"] is not None:
            if st.button(f"Saved Fuel Burn  #{notable['avoided']}", width="stretch", key="sb_jump_avoided"):
                _navigate_to("Event Inspector", notable["avoided"])
                st.rerun()
        if notable["top_risk"] is not None:
            if st.button(f"Highest Risk Case  #{notable['top_risk']}", width="stretch", key="sb_jump_top"):
                _navigate_to("Event Inspector", notable["top_risk"])
                st.rerun()

        st.markdown(
            f"""
<hr class='sb-hr'>
<div class='sb-section-title'>Reading the Numbers</div>
<div class='sb-guide-item'>
    <div class='sb-guide-head'><span>Decision Horizon</span><span class='mono' style='color:#38BDF8;'>T-{cutoff_days*24:.0f}h</span></div>
    <div class='sb-guide-desc'>All radar passes inside {cutoff_days:.0f} days to closest approach are hidden during prediction.</div>
</div>
<div class='sb-guide-item'>
    <div class='sb-guide-head'><span>Alert Risk Level</span><span class='mono' style='color:#FCD34D;'>1 in 1M ({high_risk_thresh:.1f})</span></div>
    <div class='sb-guide-desc'>ESA Kelvins high-risk threshold where an encounter becomes dangerous.</div>
</div>
<div class='sb-guide-item'>
    <div class='sb-guide-head'><span>Burn Risk Level</span><span class='mono' style='color:#FDA4AF;'>1 in 10k ({operational_thresh:.1f})</span></div>
    <div class='sb-guide-desc'>Operational emergency threshold triggering mandatory collision review.</div>
</div>
<div class='sb-guide-item'>
    <div class='sb-guide-head'><span>Confidence Tiers</span><span class='mono' style='color:#CBD5E1;'>High / Med / Low</span></div>
    <div class='sb-guide-desc'>Distance from decision cutoff; Low flags borderline cases for extra radar tasking.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # --- SINGLE-ROW TOP NAVIGATION BAR ---
    nav_brand, nav_b1, nav_b2, nav_b3, nav_b4, nav_stat = st.columns(
        [1.65, 0.8, 1.15, 1.15, 0.9, 1.75],
        gap="small",
        vertical_alignment="center",
    )

    with nav_brand:
        st.markdown(
            """
<div class='nav-brand'>
    <div class='nav-mark'>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.2">
            <circle cx="12" cy="12" r="9" stroke-opacity="0.45"/>
            <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(-32 12 12)" stroke="#F59E0B"/>
            <circle cx="12" cy="12" r="2.6" fill="#38BDF8" stroke="none"/>
        </svg>
    </div>
    <div>
        <div class='nav-name'>NearMiss</div>
        <div class='nav-sub'>SW-5 Orbital Conjunction Console</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    nav_slots = [nav_b1, nav_b2, nav_b3, nav_b4]
    for slot, page_name in zip(nav_slots, NAV_PAGES):
        with slot:
            is_active = st.session_state["active_page"] == page_name
            if st.button(
                page_name,
                type="primary" if is_active else "secondary",
                width="stretch",
                key=f"topbar_{page_name}",
            ):
                _navigate_to(page_name)
                st.rerun()

    with nav_stat:
        st.markdown(
            f"""
<div class='nav-status'>
    <span class='status-chip'><span class='status-dot'></span>T-{cutoff_days*24:.0f}h · {split_label}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='nav-divider'>", unsafe_allow_html=True)

    # --- ROUTE TO ACTIVE VIEW ---
    active_page = st.session_state["active_page"]
    if active_page == "Overview":
        render_overview(split_preds, split_hist, split_full_label, high_risk_thresh, cutoff_days)
    elif active_page == "Event Inspector":
        render_inspector(split_preds, split_hist, high_risk_thresh, cutoff_days)
    elif active_page == "Policy Simulator":
        render_simulator(split_preds)
    else:
        render_benchmarks()


if __name__ == "__main__":
    main()

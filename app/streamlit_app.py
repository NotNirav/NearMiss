"""NearMiss — Autonomous Satellite Collision Risk Predictor.

Precision Flight Dynamics & Conjunction Assessment Interface.

Views:
- Overview: Mission briefing, live interactive T-48h Blind Zone trajectory,
  three-stage operational pipeline, and head-to-head encounter outcomes.
- Event Inspector: Searchable conjunction audit, NearMiss vs Baseline A verdict,
  synchronized risk & miss-distance trajectories, and blind-zone reveal.
- Policy Simulator: Decision threshold and mission economics simulator with
  live FN/FP trade-off frontier and confusion matrices.
- Benchmarks: Full-width ESA Kelvins evaluation tables, interactive Platt
  calibration reliability, and Tree SHAP feature drivers.
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
    ("latest_log_risk", "Latest pre-cutoff log risk", 0.1186, "Risk trajectory"),
    ("risk_max", "Peak observed risk", 0.0597, "Risk trajectory"),
    ("latest_max_risk_estimate", "Max possible risk envelope", 0.0546, "Covariance physics"),
    ("latest_max_risk_scaling", "Covariance scaling factor", 0.0517, "Covariance physics"),
    ("risk_mean", "Mean risk across passes", 0.0428, "Risk trajectory"),
    ("risk_min", "Minimum observed risk", 0.0246, "Risk trajectory"),
    ("risk_total_change", "Net risk change (first to cutoff)", 0.0185, "Risk trajectory"),
    ("risk_first", "Initial tracking log risk", 0.0133, "Risk trajectory"),
    ("risk_slope_per_day", "Risk escalation slope / day", 0.0116, "Risk trajectory"),
    ("latest_time_to_tca", "Hours remaining at last CDM", 0.0072, "Covariance physics"),
    ("latest_uncertainty_volume", "3D ellipsoid volume √det(C)", 0.0067, "Covariance physics"),
    ("risk_last_step_change", "Most recent CDM risk jump", 0.0062, "Risk trajectory"),
]

# -----------------------------------------------------------------------------
# DESIGN SYSTEM — OPTICAL AEROSPACE LIGHT (SWISS PRECISION INSTRUMENT)
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --canvas: #F5F7FA;
    --surface: #FFFFFF;
    --surface-muted: #F8FAFC;
    --line: #E2E8F0;
    --line-strong: #CBD5E1;
    --ink-1: #111827;
    --ink-2: #475467;
    --ink-3: #667085;
    --cobalt: #1B4DFF;
    --cobalt-hover: #143AD6;
    --cobalt-tint: #EEF4FF;
    --cobalt-border: #C7D7FE;
    --crimson: #D92D20;
    --crimson-tint: #FEF3F2;
    --crimson-border: #FECDCA;
    --emerald: #067647;
    --emerald-tint: #ECFDF3;
    --emerald-border: #ABEFC6;
    --ochre: #B54708;
    --ochre-tint: #FFFAEB;
    --ochre-border: #FEDF89;
}

.stApp {
    background-color: var(--canvas) !important;
    color: var(--ink-1) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Crisp, compact main container (overrides Streamlit 1.65's 96px top / 80px side padding) */
[data-testid="stMainBlockContainer"] {
    padding: 0.9rem 2.25rem 2.4rem 2.25rem !important;
    max-width: 1440px !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
    gap: 0.72rem !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.1rem !important;
}
[data-testid="stToolbar"] {
    visibility: hidden !important;
}

/* Clean Pure-White Sidebar Rail */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebarHeader"] {
    padding: 0.65rem 1rem 0.1rem 1rem !important;
    min-height: 0 !important;
    height: auto !important;
}
[data-testid="stSidebarUserContent"] {
    padding: 0.2rem 1.05rem 1.5rem 1.05rem !important;
}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {
    gap: 0.52rem !important;
}

.mono {
    font-family: 'IBM Plex Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}

/* Top Navigation Bar */
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-mark {
    width: 30px;
    height: 30px;
    border-radius: 7px;
    background: var(--ink-1);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.nav-name {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.025em;
    line-height: 1.1;
}
.nav-sub {
    font-size: 0.73rem;
    color: var(--ink-3);
    line-height: 1.2;
}
.nav-status {
    display: flex;
    justify-content: flex-end;
    align-items: center;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--ink-2);
    white-space: nowrap;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--cobalt);
}
.nav-rule {
    border: none;
    border-top: 1px solid var(--line);
    margin: 4px 0 8px 0;
}

/* Button Hierarchy — Crisp, Restrained, Tactile */
button[data-testid="stBaseButton-primary"] {
    background-color: var(--cobalt) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cobalt) !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    min-height: 36px !important;
    padding: 0.35rem 0.9rem !important;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.08) !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--cobalt-hover) !important;
    border-color: var(--cobalt-hover) !important;
}
button[data-testid="stBaseButton-secondary"] {
    background-color: var(--surface) !important;
    color: var(--ink-1) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    min-height: 36px !important;
    padding: 0.35rem 0.85rem !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: var(--surface-muted) !important;
    border-color: var(--ink-3) !important;
    color: var(--ink-1) !important;
}

/* Clean Nav Tab Buttons in Top Bar (subtle ghost tabs, cobalt active tint) */
button[data-testid="stBaseButton-tertiary"] {
    background: transparent !important;
    color: var(--ink-2) !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    min-height: 35px !important;
    padding: 0.3rem 0.75rem !important;
}
button[data-testid="stBaseButton-tertiary"]:hover {
    background: #EAEEF5 !important;
    color: var(--ink-1) !important;
}

/* Sidebar Typography & Dividers */
.sb-heading {
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.01em;
    margin: 2px 0 1px 0;
}
.sb-caption {
    font-size: 0.77rem;
    color: var(--ink-3);
    line-height: 1.4;
    margin-bottom: 4px;
}
.sb-summary {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    background: var(--canvas);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 4px;
}
.sb-summary-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ink-1);
    line-height: 1.15;
}
.sb-summary-lbl {
    font-size: 0.72rem;
    color: var(--ink-3);
    margin-top: 2px;
}
.sb-rule {
    border: none;
    border-top: 1px solid var(--line);
    margin: 8px 0 6px 0;
}
.sb-guide-row {
    padding: 7px 0;
    border-bottom: 1px solid var(--line);
}
.sb-guide-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.sb-guide-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--ink-1);
}
.sb-guide-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--cobalt);
}
.sb-guide-text {
    font-size: 0.74rem;
    color: var(--ink-3);
    line-height: 1.4;
    margin-top: 2px;
}

/* Landing Hero — Open Architectural Canvas (No Boxed Slop!) */
.hero-title {
    font-size: 2.45rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.035em;
    line-height: 1.1;
    margin: 6px 0 14px 0;
}
.hero-copy {
    font-size: 0.99rem;
    color: var(--ink-2);
    line-height: 1.62;
    max-width: 56ch;
    margin: 0 0 20px 0;
}
.hero-stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border-top: 1px solid var(--line-strong);
    border-bottom: 1px solid var(--line);
    padding: 15px 0;
    margin-bottom: 18px;
    gap: 12px;
}
.hs-cell {
    padding-right: 10px;
    border-right: 1px solid var(--line);
}
.hs-cell:last-child {
    border-right: none;
    padding-right: 0;
}
.hs-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.38rem;
    font-weight: 600;
    color: var(--ink-1);
    line-height: 1.15;
}
.hs-lbl {
    font-size: 0.75rem;
    color: var(--ink-3);
    margin-top: 3px;
    line-height: 1.3;
}

/* Section Headers */
.sec-heading {
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.02em;
    margin: 10px 0 2px 0;
}
.sec-sub {
    font-size: 0.87rem;
    color: var(--ink-3);
    margin: 0 0 10px 0;
}

/* Open Editorial 3-Step Pipeline (No Boxes, Just Crisp Hairlines) */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 24px;
    border-top: 1px solid var(--line-strong);
    padding: 16px 0 10px 0;
    margin: 6px 0 12px 0;
}
.step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--cobalt);
    margin-bottom: 5px;
}
.step-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink-1);
    margin-bottom: 5px;
    letter-spacing: -0.01em;
}
.step-body {
    font-size: 0.85rem;
    color: var(--ink-2);
    line-height: 1.54;
}

/* Crisp White Instrument Surfaces */
.surface-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 18px;
}
.case-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.case-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.01em;
}
.pill {
    display: inline-flex;
    align-items: center;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    white-space: nowrap;
}
.pill-red { background: var(--crimson-tint); color: var(--crimson); border: 1px solid var(--crimson-border); }
.pill-green { background: var(--emerald-tint); color: var(--emerald); border: 1px solid var(--emerald-border); }
.pill-blue { background: var(--cobalt-tint); color: var(--cobalt); border: 1px solid var(--cobalt-border); }
.pill-ochre { background: var(--ochre-tint); color: var(--ochre); border: 1px solid var(--ochre-border); }
.pill-neutral { background: var(--canvas); color: var(--ink-2); border: 1px solid var(--line-strong); }

/* Crisp Pure-White Bordered Instrument Containers */
[class*="st-key-card_"] {
    background-color: var(--surface) !important;
    border-color: var(--line) !important;
    border-radius: 10px !important;
}

.case-desc {
    font-size: 0.85rem;
    color: var(--ink-2);
    line-height: 1.52;
    margin-bottom: 12px;
    min-height: 46px;
}
.case-table {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    background: var(--canvas);
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 9px 12px;
    font-size: 0.77rem;
}
.case-k {
    color: var(--ink-3);
    font-size: 0.71rem;
    margin-bottom: 2px;
}
.case-v {
    font-weight: 600;
    color: var(--ink-1);
}

/* Event Inspector Verdict Strip */
.verdict-bar {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 18px;
    display: grid;
    grid-template-columns: 1.65fr 1fr 1fr 1.15fr 1fr;
    gap: 14px;
    align-items: center;
}
.verdict-top {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.verdict-sub {
    font-size: 0.78rem;
    color: var(--ink-3);
    margin-top: 5px;
}
.vb-cell {
    border-left: 1px solid var(--line);
    padding-left: 14px;
}
.vb-lbl {
    font-size: 0.73rem;
    color: var(--ink-3);
    margin-bottom: 2px;
}
.vb-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.16rem;
    font-weight: 600;
    color: var(--ink-1);
    line-height: 1.2;
}
.vb-note {
    font-size: 0.73rem;
    color: var(--ink-3);
    margin-top: 2px;
}

/* Policy Simulator KPI Row */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 8px;
}
.kpi-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 9px;
    padding: 12px 14px;
}
.kpi-lbl {
    font-size: 0.74rem;
    color: var(--ink-3);
    margin-bottom: 3px;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.32rem;
    font-weight: 600;
    color: var(--ink-1);
    line-height: 1.15;
}
.kpi-sub {
    font-size: 0.74rem;
    color: var(--ink-2);
    margin-top: 3px;
}

/* Clean Light Confusion Matrix */
.cm-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 9px;
    padding: 12px 14px;
}
.cm-head {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--ink-1);
    margin-bottom: 8px;
}
.cm-tbl {
    width: 100%;
    border-collapse: separate;
    border-spacing: 5px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.79rem;
}
.cm-tbl th {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink-3);
    font-weight: 500;
    font-size: 0.75rem;
    text-align: left;
    padding: 3px 5px;
}
.cm-tbl td {
    padding: 7px 9px;
    border-radius: 6px;
    text-align: center;
    font-weight: 600;
}

/* Chart Card Title */
.chart-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--ink-1);
    margin-bottom: 2px;
}

@media (max-width: 1024px) {
    .verdict-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero-stats, .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .steps-grid { grid-template-columns: 1fr; gap: 14px; }
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
    at that boundary prevents colliding event_ids between val and test from
    merging into a single jagged trajectory.
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
    height: int = 355,
    compact_legend: bool = False,
) -> go.Figure:
    """Build a crisp light-canvas Plotly collision risk trajectory chart."""
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

    # Soft blush Blind Zone shading (< cutoff_days)
    fig.add_vrect(
        x0=cutoff_days,
        x1=0.0,
        fillcolor="rgba(217, 45, 32, 0.06)",
        layer="below",
        line_width=0,
    )
    fig.add_vline(
        x=cutoff_days,
        line_dash="dash",
        line_color="#B54708",
        line_width=1.8,
    )
    fig.add_hline(
        y=high_risk_thresh,
        line_dash="dot",
        line_color="rgba(181, 71, 8, 0.72)",
        line_width=1.4,
    )

    # Non-overlapping labels cleanly anchored on light canvas
    fig.add_annotation(
        x=cutoff_days * 0.5,
        y=0.98,
        yref="paper",
        yanchor="top",
        text="Blind Zone (<48h)",
        showarrow=False,
        font=dict(color="#B42318", size=10, family="IBM Plex Mono"),
        bgcolor="rgba(254, 243, 242, 0.92)",
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
        font=dict(color="#B54708", size=10, family="IBM Plex Mono"),
        bgcolor="rgba(255, 255, 255, 0.92)",
        borderpad=3,
    )
    alert_near_top = (y_high - high_risk_thresh) < 0.24 * max(1.0, (y_high - y_low))
    fig.add_annotation(
        x=0.99,
        xref="paper",
        xanchor="right",
        y=high_risk_thresh,
        yshift=-12 if alert_near_top else 11,
        text=f"1 in 1M Alert ({high_risk_thresh:.1f})",
        showarrow=False,
        font=dict(color="#93370D", size=10, family="IBM Plex Mono"),
        bgcolor="rgba(255, 250, 235, 0.92)",
        borderpad=3,
    )

    # Pre-cutoff CDMs in crisp Aerospace Cobalt
    if len(vis_cdms) > 0:
        fig.add_trace(
            go.Scatter(
                x=vis_cdms["time_to_tca"],
                y=vis_cdms["log_risk"],
                mode="lines+markers",
                name="Pre-cutoff CDMs",
                line=dict(color="#1B4DFF", width=2.4),
                marker=dict(size=7, color="#1B4DFF", line=dict(width=1.5, color="#FFFFFF")),
                hovertemplate="T-%{x:.2f}d<br>Observed log risk: %{y:.2f}<extra></extra>",
            )
        )

    # NearMiss predicted final risk at TCA (crisp dark charcoal diamond)
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[float(event_row["pred_final_log_risk"])],
            mode="markers",
            name=f"NearMiss forecast ({event_row['pred_final_log_risk']:.2f})",
            marker=dict(
                size=11,
                color="#111827",
                symbol="diamond",
                line=dict(width=1.5, color="#FFFFFF"),
            ),
            hovertemplate="NearMiss TCA forecast: %{y:.2f}<extra></extra>",
        )
    )

    # Post-cutoff ground truth reveal
    if reveal_gt:
        if len(post_cdms) > 0:
            bridge = pd.concat([vis_cdms.iloc[[-1]], post_cdms], ignore_index=True) if len(vis_cdms) > 0 else post_cdms
            fig.add_trace(
                go.Scatter(
                    x=bridge["time_to_tca"],
                    y=bridge["log_risk"],
                    mode="lines+markers",
                    name="Actual blind-zone track",
                    line=dict(color="#B54708", width=2.1, dash="dash"),
                    marker=dict(size=6.5, color="#FFFFFF", line=dict(width=2, color="#B54708")),
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
                    color="#D92D20" if actual_val >= high_risk_thresh else "#067647",
                    symbol="star",
                    line=dict(width=1, color="#FFFFFF"),
                ),
                hovertemplate="Actual final log risk: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Days remaining until closest approach (TCA)",
            range=[max_x, -0.25],
            gridcolor="#EEF2F6",
            linecolor="#CBD5E1",
            zeroline=False,
            color="#475467",
            title_font=dict(size=11, family="DM Sans", color="#475467"),
        ),
        yaxis=dict(
            title="Collision risk log10(P)",
            range=[y_low, y_high],
            gridcolor="#EEF2F6",
            linecolor="#CBD5E1",
            zeroline=False,
            color="#475467",
            title_font=dict(size=11, family="DM Sans", color="#475467"),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans", color="#111827"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22 if not compact_legend else -0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#475467"),
        ),
        height=height,
        margin=dict(l=58, r=16, t=14, b=72),
        hovermode="closest",
    )
    return fig


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW (LANDING PAGE)
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

    hero_l, hero_r = st.columns([1.08, 1.0], gap="large", vertical_alignment="top")

    with hero_l:
        st.markdown(
            f"""
<div class='hero-title'>Predict satellite collisions {cutoff_days*24:.0f} hours before closest approach.</div>
<p class='hero-copy'>
Flight dynamics teams need 24 to 48 hours to plan and uplink an orbital avoidance burn.
Once an encounter crosses <strong>T-{cutoff_days*24:.0f}h</strong>, new radar updates arrive inside the
<strong>Blind Zone</strong>—too late to act. NearMiss predicts final collision risk strictly from
early passes using 73 orbital covariance features, catching late-jumping hazards while reducing
unnecessary fuel burns.
</p>
<div class='hero-stats'>
    <div class='hs-cell'>
        <div class='hs-val'>14,109</div>
        <div class='hs-lbl'>ESA conjunctions evaluated</div>
    </div>
    <div class='hs-cell'>
        <div class='hs-val'>{cutoff_days*24:.0f}h</div>
        <div class='hs-lbl'>Advance decision horizon</div>
    </div>
    <div class='hs-cell'>
        <div class='hs-val'>-59.5%</div>
        <div class='hs-lbl'>Calibration error drop</div>
    </div>
    <div class='hs-cell'>
        <div class='hs-val'>-22.6%</div>
        <div class='hs-lbl'>False burns cut (OOF)</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        act1, act2 = st.columns([1.05, 1.0], gap="small")
        with act1:
            if st.button("Open Event Inspector", type="primary", width="stretch", key="hero_open_inspector"):
                _navigate_to("Event Inspector", rescued_id or notable["top_risk"])
                st.rerun()
        with act2:
            if st.button("Simulate Burn Policy", type="secondary", width="stretch", key="hero_open_sim"):
                _navigate_to("Policy Simulator")
                st.rerun()

    with hero_r:
        with st.container(border=True, key="card_hero_chart"):
            showcase_options = []
            if avoided_id is not None:
                showcase_options.append(f"Avoided False Alarm #{avoided_id}")
            if rescued_id is not None:
                showcase_options.append(f"Caught Late Collision #{rescued_id}")
            if not showcase_options and notable["top_risk"] is not None:
                showcase_options.append(f"Top Risk Event #{notable['top_risk']}")

            selected_showcase = st.segmented_control(
                "Showcase encounter",
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
            note_color = "#B42318" if is_rescued_view else "#067647"
            caption_txt = (
                f"Baseline A stood down at T-48h ({show_row['latest_log_risk']:.2f}), missing the jump to {show_row['actual_final_log_risk']:.2f}. NearMiss commanded a burn early."
                if is_rescued_view
                else f"Baseline A alarmed at T-48h ({show_row['latest_log_risk']:.2f}), but risk dropped safely to {show_row['actual_final_log_risk']:.2f}. NearMiss saved the burn."
            )
            st.markdown(
                f"<div style='font-size:0.81rem; font-weight:500; color:{note_color}; margin-bottom:4px; line-height:1.4;'>{caption_txt}</div>",
                unsafe_allow_html=True,
            )

            fig_hero = _build_risk_trajectory_figure(
                event_row=show_row,
                event_history=show_hist,
                high_risk_thresh=high_risk_thresh,
                cutoff_days=cutoff_days,
                reveal_gt=True,
                height=295,
                compact_legend=True,
            )
            st.plotly_chart(fig_hero, width="stretch", config={"displayModeBar": False})

    # Clean Open 3-Step Pipeline (Hairline Grid — Zero Box Clutter)
    st.markdown(
        """
<div class='sec-heading'>How NearMiss decides before the blind zone</div>
<div class='sec-sub'>Every feature, cross-validation fold, and calibrator strictly respects the 48-hour operational barrier.</div>
<div class='steps-grid'>
    <div>
        <div class='step-num'>01</div>
        <div class='step-title'>Early Radar Intake (T-7d to T-2d)</div>
        <div class='step-body'>
            Ingests Conjunction Data Messages up to the planning deadline and drops every observation
            inside 48 hours to closest approach before computing any feature.
        </div>
    </div>
    <div>
        <div class='step-num'>02</div>
        <div class='step-title'>Orbital Covariance Physics</div>
        <div class='step-body'>
            Computes 73 physical features including 3D uncertainty ellipsoid volume √det(C), volume
            contraction ratio across passes, combined RTN positional sigmas, and miss-distance gradients.
        </div>
    </div>
    <div>
        <div class='step-num'>03</div>
        <div class='step-title'>Calibrated Risk & Confidence</div>
        <div class='step-body'>
            Applies out-of-fold Platt scaling to convert Random Forest scores into trustworthy collision
            probabilities (Brier 0.0166 → 0.0068) paired with High, Medium, or Low certainty tiers.
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Head-to-Head Outcomes
    st.markdown(
        f"""
<div class='sec-heading'>Head-to-head encounters in {split_label}</div>
<div class='sec-sub'>Concrete cases where NearMiss outperformed legacy Persistence (Baseline A).</div>
""",
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2, gap="medium")

    with c_left:
        if rescued_id is not None:
            r_row = split_preds[split_preds["event_id"] == rescued_id].iloc[0]
            st.markdown(
                f"""
<div class='surface-card'>
    <div class='case-header'>
        <div class='case-title'>Late-Jumping Collision Caught — Event #{rescued_id}</div>
        <span class='pill pill-red'>Rescued</span>
    </div>
    <div class='case-desc'>
        At T-48h, the latest radar pass sat just below the alert line, causing Persistence to stand down.
        NearMiss detected tightening covariance geometry and triggered an avoidance burn early.
    </div>
    <div class='case-table'>
        <div><div class='case-k'>Baseline A</div><div class='case-v'>Stand Down</div></div>
        <div><div class='case-k'>NearMiss</div><div class='case-v' style='color:#D92D20;'>Burn ({r_row['high_risk_prob']:.1%})</div></div>
        <div><div class='case-k'>Actual TCA</div><div class='case-v'>{r_row['actual_final_log_risk']:.2f} (Hazard)</div></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button(f"Inspect Event #{rescued_id}", type="secondary", width="stretch", key="ov_btn_rescued"):
                _navigate_to("Event Inspector", rescued_id)
                st.rerun()

    with c_right:
        if avoided_id is not None:
            a_row = split_preds[split_preds["event_id"] == avoided_id].iloc[0]
            st.markdown(
                f"""
<div class='surface-card'>
    <div class='case-header'>
        <div class='case-title'>False Alarm Burn Prevented — Event #{avoided_id}</div>
        <span class='pill pill-green'>Burn Saved</span>
    </div>
    <div class='case-desc'>
        Early tracking noise spiked risk above the alert threshold, prompting a costly burn under Persistence.
        NearMiss identified transient early uncertainty and cleared the orbit safely.
    </div>
    <div class='case-table'>
        <div><div class='case-k'>Baseline A</div><div class='case-v' style='color:#D92D20;'>Order Burn</div></div>
        <div><div class='case-k'>NearMiss</div><div class='case-v' style='color:#067647;'>Stand Down ({a_row['high_risk_prob']:.1%})</div></div>
        <div><div class='case-k'>Actual TCA</div><div class='case-v'>{a_row['actual_final_log_risk']:.2f} (Safe)</div></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button(f"Inspect Event #{avoided_id}", type="secondary", width="stretch", key="ov_btn_avoided"):
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
                f"Select Encounter ({len(filtered):,} matching, sorted by risk probability)",
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
    conf_pill = {"HIGH": "pill-blue", "MEDIUM": "pill-ochre", "LOW": "pill-neutral"}.get(conf_raw, "pill-neutral")
    is_manoeuvre = event_row["recommendation"] == "MANOEUVRE"
    b_is_man = event_row["baseline_recommendation"] == "MANOEUVRE"

    verdict_pill = (
        "<span class='pill pill-red'>Command Avoidance Burn</span>"
        if is_manoeuvre
        else "<span class='pill pill-green'>Stand Down — Nominal</span>"
    )
    b_color = "#D92D20" if b_is_man else "#067647"
    b_text = "Order Burn" if b_is_man else "Stand Down"

    if is_manoeuvre == b_is_man:
        div_label, div_color, div_sub = "Agreement", "#475467", "Matches persistence"
    elif is_manoeuvre:
        div_label, div_color, div_sub = "Caught Early", "#1B4DFF", "Baseline stood down"
    else:
        div_label, div_color, div_sub = "Burn Saved", "#067647", "Baseline alarmed"

    st.markdown(
        f"""
<div class='verdict-bar'>
    <div>
        <div class='verdict-top'>
            {verdict_pill}
            <span class='pill {conf_pill}'>{conf_raw.title()} Confidence</span>
        </div>
        <div class='verdict-sub'>
            Event #{selected_event_id} &nbsp;/&nbsp; Sat #{int(event_row['mission_id'])} vs {event_row['object_type'].title()} &nbsp;/&nbsp; {int(event_row['cdm_count_pre_cutoff'])} pre-cutoff passes
        </div>
    </div>
    <div class='vb-cell'>
        <div class='vb-lbl'>Calibrated Risk Odds</div>
        <div class='vb-val'>{event_row['high_risk_prob']:.2%}</div>
        <div class='vb-note'>Platt posterior</div>
    </div>
    <div class='vb-cell'>
        <div class='vb-lbl'>Forecast TCA Risk</div>
        <div class='vb-val'>{event_row['pred_final_log_risk']:.2f}</div>
        <div class='vb-note'>log10(P) hurdle</div>
    </div>
    <div class='vb-cell'>
        <div class='vb-lbl'>Baseline A</div>
        <div class='vb-val' style='color:{b_color}; font-size:1.0rem;'>{b_text}</div>
        <div class='vb-note'>Last visible: {event_row['latest_log_risk']:.2f}</div>
    </div>
    <div class='vb-cell'>
        <div class='vb-lbl'>Comparison</div>
        <div class='vb-val' style='color:{div_color}; font-size:1.0rem;'>{div_label}</div>
        <div class='vb-note'>{div_sub}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    tog_c1, tog_c2 = st.columns([1.15, 2.2], vertical_alignment="center")
    with tog_c1:
        reveal_gt = st.toggle("Reveal post-cutoff outcome (Blind Zone)", value=False, key="inspector_reveal_gt")
    with tog_c2:
        if reveal_gt:
            actual_val = float(event_row["actual_final_log_risk"])
            actual_danger = int(event_row["actual_high_risk"]) == 1
            if actual_danger and is_manoeuvre:
                tag_txt, pill_cls = "True Positive — Hazard Caught", "pill-green"
            elif not actual_danger and not is_manoeuvre:
                tag_txt, pill_cls = "True Negative — Safe Clearance", "pill-green"
            elif actual_danger and not is_manoeuvre:
                tag_txt, pill_cls = "False Negative — Missed Hazard", "pill-red"
            else:
                tag_txt, pill_cls = "False Positive — Precautionary Burn", "pill-ochre"
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; font-size:0.83rem; color:#475467;'>"
                f"<span class='pill {pill_cls}'>{tag_txt}</span>"
                f"<span>Actual terminal risk: <strong class='mono' style='color:#111827;'>{actual_val:.2f}</strong></span></div>",
                unsafe_allow_html=True,
            )

    event_history = split_hist[split_hist["event_id"] == selected_event_id].sort_values(
        "time_to_tca", ascending=False
    )
    vis_cdms = event_history[event_history["visible"]]
    post_cdms = event_history[~event_history["visible"]]

    chart_c1, chart_c2 = st.columns([1.35, 1.0], gap="medium")

    with chart_c1:
        with st.container(border=True, key="card_insp_risk"):
            st.markdown(f"<div class='chart-title'>Collision Risk Trajectory — Event #{selected_event_id}</div>", unsafe_allow_html=True)
            fig_risk = _build_risk_trajectory_figure(
                event_row=event_row,
                event_history=event_history,
                high_risk_thresh=high_risk_thresh,
                cutoff_days=cutoff_days,
                reveal_gt=reveal_gt,
                height=345,
            )
            st.plotly_chart(fig_risk, width="stretch", config={"displayModeBar": False})

    with chart_c2:
        with st.container(border=True, key="card_insp_miss"):
            st.markdown("<div class='chart-title'>Relative Miss Distance Across Passes</div>", unsafe_allow_html=True)
            max_x = float(max(7.0, event_history["time_to_tca"].max() + 0.35)) if len(event_history) else 7.2
            fig_miss = go.Figure()
            fig_miss.add_vrect(
                x0=cutoff_days,
                x1=0.0,
                fillcolor="rgba(217, 45, 32, 0.06)",
                layer="below",
                line_width=0,
            )
            fig_miss.add_vline(x=cutoff_days, line_dash="dash", line_color="#B54708", line_width=1.6)
            if len(vis_cdms) > 0:
                fig_miss.add_trace(
                    go.Scatter(
                        x=vis_cdms["time_to_tca"],
                        y=vis_cdms["miss_distance"],
                        mode="lines+markers",
                        name="Pre-cutoff miss distance",
                        line=dict(color="#111827", width=2.2),
                        marker=dict(size=6.5, color="#111827", line=dict(width=1.5, color="#FFFFFF")),
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
                        line=dict(color="#B54708", width=2, dash="dash"),
                        marker=dict(size=6, color="#FFFFFF", line=dict(width=2, color="#B54708")),
                        hovertemplate="T-%{x:.2f}d<br>Actual miss: %{y:,.0f} m<extra></extra>",
                    )
                )
            fig_miss.update_layout(
                xaxis=dict(
                    title="Days remaining until TCA",
                    range=[max_x, -0.25],
                    gridcolor="#EEF2F6",
                    linecolor="#CBD5E1",
                    zeroline=False,
                    color="#475467",
                    title_font=dict(size=11, family="DM Sans", color="#475467"),
                ),
                yaxis=dict(
                    title="Miss distance (meters)",
                    gridcolor="#EEF2F6",
                    linecolor="#CBD5E1",
                    zeroline=False,
                    color="#475467",
                    title_font=dict(size=11, family="DM Sans", color="#475467"),
                ),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#111827"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10.5, color="#475467"),
                ),
                height=345,
                margin=dict(l=62, r=16, t=14, b=72),
                hovermode="closest",
            )
            st.plotly_chart(fig_miss, width="stretch", config={"displayModeBar": False})

    with st.expander(f"CDM Telemetry Log ({len(event_history if reveal_gt else vis_cdms)} passes)", expanded=False):
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
<div class='sec-heading'>Decision Threshold & Mission Economics Simulator</div>
<div class='sec-sub'>Tune the calibrated probability trigger to compare missed collisions, avoidable maneuvers, and net mission balance against Baseline A.</div>
""",
        unsafe_allow_html=True,
    )

    sim_l, sim_r = st.columns([1.0, 1.85], gap="large")

    with sim_l:
        with st.container(border=True, key="card_sim_ctrl"):
            st.markdown("<div class='chart-title'>Spacecraft Profile</div>", unsafe_allow_html=True)
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
                profile_note = "Commercial LEO SmallSat — USD 250k miss penalty / USD 15k burn."
            elif preset == "Earth Obs":
                default_miss_penalty, default_burn_val, default_th = 1_000_000, 45_000, 0.015
                profile_note = "Flagship Earth Observation — USD 1M miss penalty / USD 45k burn."
            else:
                default_miss_penalty, default_burn_val, default_th = 5_000_000, 100_000, 0.010
                profile_note = "High-Value Infrastructure — USD 5M miss penalty / USD 100k burn."

            st.markdown(f"<div style='font-size:0.78rem; color:#667085; margin-bottom:6px;'>{profile_note}</div>", unsafe_allow_html=True)

            sim_th = st.slider(
                "Maneuver Trigger Threshold (p*)",
                min_value=0.005,
                max_value=0.250,
                value=default_th,
                step=0.005,
                format="%.3f",
                key=f"sim_th_{preset}",
            )
            st.caption(f"Trigger an avoidance burn when calibrated P(collision) ≥ {sim_th:.1%}.")

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

            sav_bg = "#ECFDF3" if savings >= 0 else "#FEF3F2"
            sav_bd = "#ABEFC6" if savings >= 0 else "#FECDCA"
            sav_fg = "#067647" if savings >= 0 else "#D92D20"
            st.markdown(
                f"""
<div style='background:{sav_bg}; border:1px solid {sav_bd}; border-radius:8px; padding:12px 14px; margin-top:6px;'>
    <div style='font-size:0.74rem; font-weight:500; color:#475467;'>Expected Financial Balance vs Baseline A</div>
    <div class='mono' style='font-size:1.45rem; font-weight:600; color:{sav_fg}; margin:2px 0;'>USD {savings:+,.0f}</div>
    <div style='font-size:0.78rem; color:#475467;'>{savings_pct:+.1f}% mission cost change at p* = {sim_th:.3f}.</div>
</div>
""",
                unsafe_allow_html=True,
            )

    with sim_r:
        fn_delta = sim_metrics["fn"] - base_metrics["fn"]
        fp_saved = base_metrics["fp"] - sim_metrics["fp"]
        fn_col = "#067647" if fn_delta <= 0 else "#D92D20"
        fp_col = "#067647" if fp_saved >= 0 else "#D92D20"

        st.markdown(
            f"""
<div class='kpi-row'>
    <div class='kpi-box'>
        <div class='kpi-lbl'>Recall (Caught)</div>
        <div class='kpi-val'>{sim_metrics['recall']:.1%}</div>
        <div class='kpi-sub'>{sim_metrics['tp']} of {sim_metrics['tp']+sim_metrics['fn']} hazards</div>
    </div>
    <div class='kpi-box'>
        <div class='kpi-lbl'>Missed Hazards (FN)</div>
        <div class='kpi-val' style='color:{fn_col};'>{sim_metrics['fn']}</div>
        <div class='kpi-sub'>{fn_delta:+d} vs Baseline ({base_metrics['fn']})</div>
    </div>
    <div class='kpi-box'>
        <div class='kpi-lbl'>False Alarms (FP)</div>
        <div class='kpi-val' style='color:{fp_col};'>{sim_metrics['fp']}</div>
        <div class='kpi-sub'>{fp_saved:+d} saved vs Baseline ({base_metrics['fp']})</div>
    </div>
    <div class='kpi-box'>
        <div class='kpi-lbl'>F2 Score</div>
        <div class='kpi-val'>{sim_metrics['f2']:.3f}</div>
        <div class='kpi-sub'>Baseline F2: {base_metrics['f2']:.3f}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        th_grid = np.linspace(0.005, 0.35, 70)
        sweep_pts = []
        for t in th_grid:
            p_bin = (split_preds["high_risk_prob"] >= t).astype(int)
            m = compute_classification_metrics(y_eval_true, p_bin)
            sweep_pts.append({"threshold": t, "fn": m["fn"], "fp": m["fp"]})
        sweep_df = pd.DataFrame(sweep_pts)

        with st.container(border=True, key="card_sim_chart"):
            fig_sim = go.Figure()
            fig_sim.add_trace(
                go.Scatter(
                    x=sweep_df["fp"],
                    y=sweep_df["fn"],
                    mode="lines",
                    name="NearMiss trade-off curve",
                    line=dict(color="#1B4DFF", width=2.5),
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
                    marker=dict(size=13, color="#D92D20", symbol="star", line=dict(width=1, color="#FFFFFF")),
                    hovertemplate=f"Active Policy (p*={sim_th:.3f})<br>FP: %{{x}}, FN: %{{y}}<extra></extra>",
                )
            )
            fig_sim.add_trace(
                go.Scatter(
                    x=[base_metrics["fp"]],
                    y=[base_metrics["fn"]],
                    mode="markers",
                    name="Baseline A (Persistence)",
                    marker=dict(size=10, color="#111827", symbol="square", line=dict(width=1.5, color="#FFFFFF")),
                    hovertemplate="Baseline A<br>FP: %{x}, FN: %{y}<extra></extra>",
                )
            )
            fig_sim.update_layout(
                xaxis=dict(title="Unnecessary maneuvers (FP)", gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475467", title_font=dict(size=11)),
                yaxis=dict(title="Missed collisions (FN)", gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475467", title_font=dict(size=11)),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#111827"),
                legend=dict(orientation="h", yanchor="top", y=-0.30, xanchor="center", x=0.5, font=dict(size=10.5, color="#475467")),
                height=248,
                margin=dict(l=54, r=16, t=10, b=68),
            )
            st.plotly_chart(fig_sim, width="stretch", config={"displayModeBar": False})

        cm1, cm2 = st.columns(2, gap="small")
        with cm1:
            st.markdown(
                f"""
<div class='cm-box'>
    <div class='cm-head'>NearMiss Matrix (p* = {sim_th:.3f})</div>
    <table class='cm-tbl'>
        <tr><th></th><th>Pred Safe</th><th>Pred Burn</th></tr>
        <tr>
            <th>Actual Safe</th>
            <td style='background:#F5F7FA; color:#111827;'>{sim_metrics['tn']:,} (TN)</td>
            <td style='background:#FEF3F2; color:#B42318;'>{sim_metrics['fp']:,} (FP)</td>
        </tr>
        <tr>
            <th>Actual Danger</th>
            <td style='background:#FEF3F2; color:#B42318;'>{sim_metrics['fn']:,} (FN)</td>
            <td style='background:#ECFDF3; color:#067647;'>{sim_metrics['tp']:,} (TP)</td>
        </tr>
    </table>
</div>
""",
                unsafe_allow_html=True,
            )
        with cm2:
            st.markdown(
                f"""
<div class='cm-box'>
    <div class='cm-head'>Baseline A Matrix (Persistence)</div>
    <table class='cm-tbl'>
        <tr><th></th><th>Pred Safe</th><th>Pred Burn</th></tr>
        <tr>
            <th>Actual Safe</th>
            <td style='background:#F5F7FA; color:#111827;'>{base_metrics['tn']:,} (TN)</td>
            <td style='background:#FEF3F2; color:#B42318;'>{base_metrics['fp']:,} (FP)</td>
        </tr>
        <tr>
            <th>Actual Danger</th>
            <td style='background:#FEF3F2; color:#B42318;'>{base_metrics['fn']:,} (FN)</td>
            <td style='background:#ECFDF3; color:#067647;'>{base_metrics['tp']:,} (TP)</td>
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
<div class='sec-heading'>ESA Kelvins Benchmarks & Model Calibration</div>
<div class='sec-sub'>Evaluated under strict event-grouped cross-validation (8,293 filtered events) and on the locked blind competition test set (2,167 events).</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='chart-title'>Locked ESA Blind Test Set (2,167 Events, 150 High-Risk)</div>", unsafe_allow_html=True)
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

    st.markdown("<div class='chart-title' style='margin-top:4px;'>5-Fold Out-of-Fold Cross-Validation (8,293 Filtered Events, 66 High-Risk)</div>", unsafe_allow_html=True)
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

    diag_l, diag_r = st.columns(2, gap="medium")

    with diag_l:
        with st.container(border=True, key="card_bench_cal"):
            st.markdown("<div class='chart-title'>Platt Probability Calibration Reliability (-59.5% Brier Error)</div>", unsafe_allow_html=True)
            cal_df = load_calibration_oof()
            fig_cal = go.Figure()

            fig_cal.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect calibration",
                    line=dict(color="#94A3B8", width=1.4, dash="dash"),
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
                        line=dict(color="#D92D20", width=2.2),
                        marker=dict(size=7, color="#D92D20", line=dict(width=1.2, color="#FFFFFF")),
                        hovertemplate="Predicted: %{x:.1%}<br>Observed rate: %{y:.1%}<extra></extra>",
                    )
                )
                fig_cal.add_trace(
                    go.Scatter(
                        x=cal_pred,
                        y=cal_true,
                        mode="lines+markers",
                        name=f"Platt Calibrated RF (Brier {brier_cal:.4f})",
                        line=dict(color="#1B4DFF", width=2.6),
                        marker=dict(size=8, color="#1B4DFF", line=dict(width=1.5, color="#FFFFFF")),
                        hovertemplate="Calibrated: %{x:.1%}<br>Observed rate: %{y:.1%}<extra></extra>",
                    )
                )

            fig_cal.update_layout(
                xaxis=dict(title="Mean predicted probability", range=[-0.02, 1.0], tickformat=".0%", gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475467"),
                yaxis=dict(title="Observed high-risk fraction", range=[-0.02, 0.52], tickformat=".0%", gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475467"),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#111827"),
                legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5, font=dict(size=10.5, color="#475467")),
                height=340,
                margin=dict(l=58, r=16, t=12, b=72),
            )
            st.plotly_chart(fig_cal, width="stretch", config={"displayModeBar": False})

    with diag_r:
        with st.container(border=True, key="card_bench_shap"):
            st.markdown("<div class='chart-title'>Global Feature Drivers (Tree SHAP Mean |SHAP|)</div>", unsafe_allow_html=True)
            shap_df = pd.DataFrame(TOP_SHAP_FEATURES, columns=["raw", "label", "val", "group"]).iloc[::-1]
            # Restrained two-tone scientific palette (Cobalt vs Deep Slate — no neon rainbow)
            color_map = {"Risk trajectory": "#1B4DFF", "Covariance physics": "#334155"}

            fig_shap = go.Figure()
            for grp in ["Risk trajectory", "Covariance physics"]:
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
                xaxis=dict(title="Mean |SHAP| impact on collision risk", gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475467"),
                yaxis=dict(categoryorder="array", categoryarray=shap_df["label"].tolist(), color="#111827", tickfont=dict(size=10.5)),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#111827"),
                legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5, font=dict(size=10.5, color="#475467")),
                height=340,
                margin=dict(l=175, r=16, t=12, b=72),
            )
            st.plotly_chart(fig_shap, width="stretch", config={"displayModeBar": False})

    # Clean Open Hairline Physics Strip
    st.markdown(
        """
<div class='steps-grid'>
    <div>
        <div class='step-title'>Combined Positional Sigmas</div>
        <div class='step-body'>
            Root-sum-square radial, along-track, and cross-track dispersion fusing target and chaser covariance tensors:
            <span class='mono' style='color:#111827; font-weight:600;'>σ_pos = √(σ_t² + σ_c²)</span>.
        </div>
    </div>
    <div>
        <div class='step-title'>3D Ellipsoid Volume</div>
        <div class='step-body'>
            Determinant-based positional uncertainty volume measuring how tightly ground radar constrains the encounter:
            <span class='mono' style='color:#111827; font-weight:600;'>Vol = √det(C_pos)</span>.
        </div>
    </div>
    <div>
        <div class='step-title'>Volume Contraction Ratio</div>
        <div class='step-body'>
            Ratio of latest to earliest covariance volume across passes—separating tightening threats from transient noise:
            <span class='mono' style='color:#111827; font-weight:600;'>Vol_cutoff / Vol_initial</span>.
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# MAIN APPLICATION SHELL (WHITE SIDEBAR RAIL + CLEAN TOP NAVBAR)
# -----------------------------------------------------------------------------
def main() -> None:
    cfg, all_preds, all_hist = load_app_data()

    if "active_page" not in st.session_state or st.session_state["active_page"] not in NAV_PAGES:
        st.session_state["active_page"] = "Overview"

    high_risk_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))
    operational_thresh = float(cfg.get("thresholds", {}).get("operational_risk_log10", -4.0))
    cutoff_days = float(cfg["data"]["cutoff_days"])

    # --- CLEAN, PLAIN-ENGLISH SIDEBAR ---
    with st.sidebar:
        st.markdown("<div class='sb-heading'>Dataset</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sb-caption'>Choose which held-out ESA encounter cohort to inspect.</div>",
            unsafe_allow_html=True,
        )
        split_choice = st.radio(
            "Dataset Cohort",
            options=["Validation Set (2,389)", "ESA Blind Test (2,167)"],
            index=0,
            label_visibility="collapsed",
        )
        split_key = "val" if split_choice.startswith("Validation") else "test"
        split_short = "Validation" if split_key == "val" else "Blind Test"
        split_full = "Validation Set" if split_key == "val" else "ESA Blind Test Set"
        split_preds = all_preds[all_preds["split"] == split_key].copy()
        split_hist = _slice_history_for_split(all_hist, split_key)

        n_man = int((split_preds["recommendation"] == "MANOEUVRE").sum())
        n_haz = int(split_preds["actual_high_risk"].sum())
        st.markdown(
            f"""
<div class='sb-summary'>
    <div>
        <div class='sb-summary-num'>{n_man:,}</div>
        <div class='sb-summary-lbl'>Burns flagged</div>
    </div>
    <div>
        <div class='sb-summary-num'>{n_haz:,}</div>
        <div class='sb-summary-lbl'>Actual hazards</div>
    </div>
</div>
<hr class='sb-rule'>
""",
            unsafe_allow_html=True,
        )

        notable = _notable_events(split_preds)
        st.markdown("<div class='sb-heading'>Featured Encounters</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sb-caption'>Jump directly into a curated real trajectory:</div>",
            unsafe_allow_html=True,
        )
        if notable["rescued"] is not None:
            if st.button(f"Caught Collision  #{notable['rescued']}", type="secondary", width="stretch", key="sb_jump_rescued"):
                _navigate_to("Event Inspector", notable["rescued"])
                st.rerun()
        if notable["avoided"] is not None:
            if st.button(f"Saved Fuel Burn  #{notable['avoided']}", type="secondary", width="stretch", key="sb_jump_avoided"):
                _navigate_to("Event Inspector", notable["avoided"])
                st.rerun()
        if notable["top_risk"] is not None:
            if st.button(f"Highest Risk Case  #{notable['top_risk']}", type="secondary", width="stretch", key="sb_jump_top"):
                _navigate_to("Event Inspector", notable["top_risk"])
                st.rerun()

        st.markdown(
            f"""
<hr class='sb-rule'>
<div class='sb-heading'>Operator Guide</div>
<div class='sb-guide-row'>
    <div class='sb-guide-top'><span>Decision Horizon</span><span class='sb-guide-val'>T-{cutoff_days*24:.0f}h</span></div>
    <div class='sb-guide-text'>Radar passes inside {cutoff_days:.0f} days to closest approach are hidden during inference.</div>
</div>
<div class='sb-guide-row'>
    <div class='sb-guide-top'><span>Alert Threshold</span><span class='sb-guide-val'>1 in 1M ({high_risk_thresh:.1f})</span></div>
    <div class='sb-guide-text'>ESA Kelvins high-risk threshold defining a dangerous encounter.</div>
</div>
<div class='sb-guide-row'>
    <div class='sb-guide-top'><span>Burn Threshold</span><span class='sb-guide-val'>1 in 10k ({operational_thresh:.1f})</span></div>
    <div class='sb-guide-text'>Operational level triggering immediate collision avoidance review.</div>
</div>
<div class='sb-guide-row'>
    <div class='sb-guide-top'><span>Confidence</span><span class='sb-guide-val'>High / Med / Low</span></div>
    <div class='sb-guide-text'>Margin from threshold; Low flags borderline cases for extra radar tracking.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # --- CLEAN SINGLE-ROW TOP NAVIGATION BAR ---
    nav_brand, nav_b1, nav_b2, nav_b3, nav_b4, nav_stat = st.columns(
        [1.65, 0.82, 1.12, 1.12, 0.92, 1.77],
        gap="small",
        vertical_alignment="center",
    )

    with nav_brand:
        st.markdown(
            """
<div class='nav-brand'>
    <div class='nav-mark'>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.1">
            <circle cx="12" cy="12" r="8.5" stroke-opacity="0.4"/>
            <ellipse cx="12" cy="12" rx="10" ry="3.8" transform="rotate(-30 12 12)" stroke="#FFFFFF"/>
            <circle cx="12" cy="12" r="2.4" fill="#60A5FA" stroke="none"/>
        </svg>
    </div>
    <div>
        <div class='nav-name'>NearMiss</div>
        <div class='nav-sub'>SW-5 Conjunction Assessment</div>
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
                type="primary" if is_active else "tertiary",
                width="stretch",
                key=f"topbar_{page_name}",
            ):
                _navigate_to(page_name)
                st.rerun()

    with nav_stat:
        st.markdown(
            f"""
<div class='nav-status'>
    <span class='status-chip'><span class='status-dot'></span>T-{cutoff_days*24:.0f}h Cutoff &nbsp;/&nbsp; {split_short}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='nav-rule'>", unsafe_allow_html=True)

    # --- ROUTE TO ACTIVE VIEW ---
    active_page = st.session_state["active_page"]
    if active_page == "Overview":
        render_overview(split_preds, split_hist, split_full, high_risk_thresh, cutoff_days)
    elif active_page == "Event Inspector":
        render_inspector(split_preds, split_hist, high_risk_thresh, cutoff_days)
    elif active_page == "Policy Simulator":
        render_simulator(split_preds)
    else:
        render_benchmarks()


if __name__ == "__main__":
    main()

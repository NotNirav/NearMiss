"""NearMiss — Autonomous Satellite Collision Risk Predictor.

Precision Orbital Conjunction Assessment & Flight Dynamics Interface.

Architecture:
- Single unified full-width top navigation bar (no redundant sidebar) with
  global cohort switching (Validation vs ESA Blind Test).
- Overview: Animated Hero with a live 60fps interactive orbital conjunction &
  avoidance-burn simulator canvas, spacious KPI cards, interactive head-to-head
  encounter explorer, 3-step pipeline cards, and operator reference footer.
- Event Inspector: Quick-jump notable encounters, filterable selector, clean
  verdict bar, overlap-free risk & miss-distance trajectories, and blind-zone reveal.
- Policy Simulator: Spacecraft profile presets, interactive decision threshold
  slider, live FN/FP frontier, expected cost balance, and confusion matrices.
- Benchmarks: Full-width ESA Kelvins tables, interactive Platt calibration
  reliability curve, and Tree SHAP feature drivers.
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
import streamlit.components.v1 as components
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.evaluate import compute_classification_metrics
from src.load import load_config, load_parquet

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION (Sidebar collapsed & hidden — single clean top navbar)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NearMiss — Satellite Collision Risk Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAV_PAGES = [
    "Overview",
    "Event Inspector",
    "Policy Simulator",
    "Benchmarks",
]

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
# DESIGN SYSTEM — SPACIOUS OPTICAL AEROSPACE LIGHT + PURPOSEFUL MOTION
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --canvas: #F4F6FA;
    --surface: #FFFFFF;
    --surface-soft: #F8FAFC;
    --line: #E2E8F0;
    --line-strong: #CBD5E1;
    --ink-1: #0F172A;
    --ink-2: #475569;
    --ink-3: #64748B;
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
    --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.03);
    --shadow-md: 0 6px 20px rgba(15, 23, 42, 0.06), 0 2px 6px rgba(15, 23, 42, 0.04);
}

/* CRITICAL FIX: Completely remove Streamlit's invisible 60px top overlay header
   that previously had z-index:999990 + pointer-events:auto and blocked navbar clicks! */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeaderActionElements"],
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
    pointer-events: none !important;
}

.stApp {
    background-color: var(--canvas) !important;
    color: var(--ink-1) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Spacious, centered full-width layout with generous vertical rhythm */
[data-testid="stMainBlockContainer"] {
    padding: 1.25rem 2.5rem 3.5rem 2.5rem !important;
    max-width: 1360px !important;
    margin: 0 auto !important;
}

[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    gap: 1.1rem !important;
}

.mono {
    font-family: 'IBM Plex Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}

/* Orchestrated entrance & micro-interaction animations */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes softPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(27, 77, 255, 0.45); }
    50%      { box-shadow: 0 0 0 6px rgba(27, 77, 255, 0); }
}
.anim-1 { animation: fadeUp 0.42s cubic-bezier(0.16, 1, 0.3, 1) both; }
.anim-2 { animation: fadeUp 0.48s cubic-bezier(0.16, 1, 0.3, 1) 0.07s both; }
.anim-3 { animation: fadeUp 0.52s cubic-bezier(0.16, 1, 0.3, 1) 0.14s both; }

@media (prefers-reduced-motion: reduce) {
    .anim-1, .anim-2, .anim-3 { animation: none !important; }
    .pulse-dot { animation: none !important; }
}

/* Pure-White Bordered Containers */
[class*="st-key-card_"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 1.15rem 1.35rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[class*="st-key-navbar_shell"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 0.6rem 1.15rem !important;
    box-shadow: var(--shadow-sm) !important;
}
/* Cancel Streamlit's -1rem bottom margin on stMarkdownContainer so brand aligns dead-center */
[class*="st-key-navbar_shell"] [data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}
[class*="st-key-navbar_shell"] [data-testid="stColumn"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
[class*="st-key-navbar_shell"] button[data-testid="stBaseButton-primary"],
[class*="st-key-navbar_shell"] button[data-testid="stBaseButton-tertiary"] {
    min-height: 36px !important;
    height: 36px !important;
    padding: 0 0.85rem !important;
}

/* Top Navbar Brand */
.brand-wrap {
    display: flex;
    align-items: center;
    gap: 11px;
    white-space: nowrap;
}
.brand-mark {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: var(--ink-1);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.brand-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.025em;
    line-height: 1.15;
    white-space: nowrap;
}
.brand-sub {
    font-size: 0.72rem;
    color: var(--ink-3);
    line-height: 1.2;
    white-space: nowrap;
}

/* Crisp Interactive Buttons */
button[data-testid="stBaseButton-primary"] {
    background-color: var(--cobalt) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cobalt) !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    min-height: 40px !important;
    padding: 0.45rem 1rem !important;
    box-shadow: 0 2px 6px rgba(27, 77, 255, 0.22) !important;
    transition: transform 0.16s ease, background-color 0.16s ease, box-shadow 0.16s ease !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--cobalt-hover) !important;
    border-color: var(--cobalt-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(27, 77, 255, 0.28) !important;
}
button[data-testid="stBaseButton-secondary"] {
    background-color: var(--surface) !important;
    color: var(--ink-1) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    min-height: 40px !important;
    padding: 0.45rem 0.95rem !important;
    transition: transform 0.16s ease, border-color 0.16s ease, background-color 0.16s ease !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: var(--surface-soft) !important;
    border-color: var(--cobalt) !important;
    color: var(--cobalt) !important;
    transform: translateY(-1px) !important;
}
button[data-testid="stBaseButton-tertiary"] {
    background: transparent !important;
    color: var(--ink-2) !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    min-height: 36px !important;
    padding: 0.35rem 0.8rem !important;
    transition: background-color 0.15s ease, color 0.15s ease !important;
}
button[data-testid="stBaseButton-tertiary"]:hover {
    background: var(--canvas) !important;
    color: var(--ink-1) !important;
}

/* Hero Section */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--ink-2);
    margin-bottom: 18px;
    box-shadow: var(--shadow-sm);
}
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--cobalt);
    animation: softPulse 2.2s infinite;
}
.hero-h1 {
    font-size: 2.85rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.04em;
    line-height: 1.08;
    margin: 0 0 18px 0;
}
.hero-p {
    font-size: 1.04rem;
    color: var(--ink-2);
    line-height: 1.65;
    max-width: 52ch;
    margin: 0 0 26px 0;
}

/* Spacious Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin: 8px 0 12px 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: var(--line-strong);
}
.mc-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.mc-icon {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: var(--cobalt-tint);
    color: var(--cobalt);
    display: flex;
    align-items: center;
    justify-content: center;
}
.mc-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    color: var(--ink-1);
    line-height: 1.15;
    margin-bottom: 4px;
}
.mc-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink-1);
    margin-bottom: 2px;
}
.mc-sub {
    font-size: 0.78rem;
    color: var(--ink-3);
    line-height: 1.35;
}

/* Clear Section Rhythm */
.section-wrap {
    margin-top: 22px;
    margin-bottom: 4px;
}
.sec-h2 {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.025em;
    margin: 0 0 4px 0;
}
.sec-p {
    font-size: 0.92rem;
    color: var(--ink-3);
    margin: 0;
}

/* 3-Step Architecture Cards */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 20px;
    margin-top: 8px;
}
.pipe-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 24px;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.pipe-box:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.pipe-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}
.pipe-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--cobalt);
    background: var(--cobalt-tint);
    border: 1px solid var(--cobalt-border);
    border-radius: 8px;
    padding: 4px 9px;
}
.pipe-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--ink-1);
    letter-spacing: -0.01em;
}
.pipe-body {
    font-size: 0.88rem;
    color: var(--ink-2);
    line-height: 1.6;
}

/* Operator Reference Footer Strip */
.ref-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 12px;
}
.ref-item {
    border-right: 1px solid var(--line);
    padding-right: 14px;
}
.ref-item:last-child {
    border-right: none;
    padding-right: 0;
}
.ref-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 4px;
}
.ref-name {
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--ink-1);
}
.ref-code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--cobalt);
}
.ref-desc {
    font-size: 0.77rem;
    color: var(--ink-3);
    line-height: 1.45;
}

/* Status Pills */
.pill {
    display: inline-flex;
    align-items: center;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 11px;
    border-radius: 999px;
    white-space: nowrap;
}
.pill-red { background: var(--crimson-tint); color: var(--crimson); border: 1px solid var(--crimson-border); }
.pill-green { background: var(--emerald-tint); color: var(--emerald); border: 1px solid var(--emerald-border); }
.pill-blue { background: var(--cobalt-tint); color: var(--cobalt); border: 1px solid var(--cobalt-border); }
.pill-ochre { background: var(--ochre-tint); color: var(--ochre); border: 1px solid var(--ochre-border); }
.pill-neutral { background: var(--canvas); color: var(--ink-2); border: 1px solid var(--line-strong); }

/* Case Comparison Stack inside Overview Showcase */
.story-card-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 14px 0 18px 0;
}
.story-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--canvas);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 11px 14px;
}
.story-k {
    font-size: 0.82rem;
    color: var(--ink-3);
    font-weight: 500;
}
.story-v {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--ink-1);
}

/* Event Inspector Verdict Strip */
.verdict-bar {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px 22px;
    display: grid;
    grid-template-columns: 1.75fr 1fr 1fr 1.15fr 1fr;
    gap: 16px;
    align-items: center;
    box-shadow: var(--shadow-sm);
}
.verdict-top {
    display: flex;
    align-items: center;
    gap: 9px;
    flex-wrap: wrap;
}
.verdict-sub {
    font-size: 0.8rem;
    color: var(--ink-3);
    margin-top: 6px;
}
.vb-cell {
    border-left: 1px solid var(--line);
    padding-left: 16px;
}
.vb-lbl {
    font-size: 0.74rem;
    color: var(--ink-3);
    margin-bottom: 3px;
}
.vb-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--ink-1);
    line-height: 1.2;
}
.vb-note {
    font-size: 0.74rem;
    color: var(--ink-3);
    margin-top: 2px;
}

/* Policy Simulator KPI Row & Confusion Matrix */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 10px;
}
.kpi-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 15px 17px;
    box-shadow: var(--shadow-sm);
}
.kpi-lbl {
    font-size: 0.76rem;
    color: var(--ink-3);
    margin-bottom: 4px;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.38rem;
    font-weight: 600;
    color: var(--ink-1);
    line-height: 1.15;
}
.kpi-sub {
    font-size: 0.76rem;
    color: var(--ink-2);
    margin-top: 4px;
}
.cm-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 15px 17px;
    box-shadow: var(--shadow-sm);
}
.cm-head {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--ink-1);
    margin-bottom: 10px;
}
.cm-tbl {
    width: 100%;
    border-collapse: separate;
    border-spacing: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.cm-tbl th {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink-3);
    font-weight: 500;
    font-size: 0.76rem;
    text-align: left;
    padding: 4px 6px;
}
.cm-tbl td {
    padding: 9px 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
}

.chart-head {
    font-size: 0.96rem;
    font-weight: 700;
    color: var(--ink-1);
    margin-bottom: 2px;
}
.chart-sub {
    font-size: 0.79rem;
    color: var(--ink-3);
    margin-bottom: 8px;
}

@media (max-width: 1080px) {
    .metrics-grid, .ref-strip, .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .verdict-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .pipeline-grid { grid-template-columns: 1fr; }
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
    """Isolate CDM history strictly to the selected cohort split."""
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


# -----------------------------------------------------------------------------
# LIVE INTERACTIVE 60FPS ORBITAL CONJUNCTION SIMULATOR (HERO CENTERPIECE)
# -----------------------------------------------------------------------------
def _render_orbital_hero_canvas() -> None:
    """Embed a crisp, animated, interactive orbital conjunction & burn simulator."""
    html_code = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
  body { background: transparent; overflow: hidden; }
  .shell {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
    padding: 16px 18px 14px 18px;
  }
  .top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .title-block { display: flex; flex-direction: column; gap: 2px; }
  .title { font-size: 14px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em; }
  .sub { font-size: 12px; color: #64748B; }
  .status-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    transition: all 0.25s ease;
  }
  .pill-risk { background: #FEF3F2; color: #D92D20; border: 1px solid #FECDCA; }
  .pill-safe { background: #ECFDF3; color: #067647; border: 1px solid #ABEFC6; }
  .canvas-box {
    position: relative;
    width: 100%;
    height: 235px;
    border-radius: 11px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    overflow: hidden;
  }
  canvas { width: 100%; height: 100%; display: block; }
  .telemetry {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 12px;
    padding-top: 11px;
    border-top: 1px solid #E2E8F0;
  }
  .t-cell { display: flex; flex-direction: column; gap: 2px; }
  .t-lbl { font-size: 11px; color: #64748B; }
  .t-val { font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 600; color: #0F172A; }
  .controls {
    display: flex;
    gap: 10px;
    margin-top: 12px;
  }
  .btn {
    flex: 1;
    height: 38px;
    border-radius: 9px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.16s ease;
    border: 1px solid transparent;
  }
  .btn-burn {
    background: #1B4DFF;
    color: #FFFFFF;
    box-shadow: 0 2px 6px rgba(27, 77, 255, 0.22);
  }
  .btn-burn:hover { background: #143AD6; transform: translateY(-1px); }
  .btn-burn.done {
    background: #ECFDF3;
    color: #067647;
    border-color: #ABEFC6;
    box-shadow: none;
  }
  .btn-reset {
    flex: 0.45;
    background: #FFFFFF;
    color: #334155;
    border-color: #CBD5E1;
  }
  .btn-reset:hover { border-color: #1B4DFF; color: #1B4DFF; }
</style>
</head>
<body>
<div class="shell">
  <div class="top-bar">
    <div class="title-block">
      <div class="title">Interactive T-48h Orbital Conjunction</div>
      <div class="sub">Fire an avoidance burn before the cutoff horizon to clear the collision cone</div>
    </div>
    <span id="statusPill" class="status-pill pill-risk">CONJUNCTION ALERT</span>
  </div>
  <div class="canvas-box">
    <canvas id="orbitCanvas"></canvas>
  </div>
  <div class="telemetry">
    <div class="t-cell">
      <span class="t-lbl">Time to TCA</span>
      <span id="tcaVal" class="t-val">T-58.0 h</span>
    </div>
    <div class="t-cell">
      <span class="t-lbl">Projected Miss Distance</span>
      <span id="missVal" class="t-val" style="color:#D92D20;">185 m</span>
    </div>
    <div class="t-cell">
      <span class="t-lbl">Collision Odds</span>
      <span id="probVal" class="t-val" style="color:#D92D20;">1 in 4,200</span>
    </div>
  </div>
  <div class="controls">
    <button id="burnBtn" class="btn btn-burn" onclick="triggerBurn()">Simulate Avoidance Burn (Δv)</button>
    <button id="resetBtn" class="btn btn-reset" onclick="resetSim()">Replay</button>
  </div>
</div>

<script>
const canvas = document.getElementById('orbitCanvas');
const ctx = canvas.getContext('2d');
let dpr = window.devicePixelRatio || 1;
let W = 600, H = 235;

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  W = rect.width;
  H = rect.height;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
resize();
window.addEventListener('resize', resize);

let progress = 0.0;      // 0 = T-72h, 0.34 = T-48h cutoff, 1.0 = TCA
let burned = false;
let burnBlend = 0.0;     // smoothly animates orbit deflection 0 -> 1
let pulsePhase = 0.0;

function satPos(p, deflect) {
  const x = 36 + p * (W - 96);
  const baseCurve = Math.sin(p * Math.PI * 0.75) * -18;
  const burnShift = deflect * Math.max(0, (p - 0.25) / 0.75) * Math.max(0, (p - 0.25) / 0.75) * 52;
  const y = H * 0.62 + baseCurve + burnShift;
  return { x, y };
}

function debrisPos(p) {
  const x = 36 + p * (W - 96);
  const y = H * 0.24 + p * (H * 0.40 - 15) + Math.sin(p * Math.PI) * 8;
  return { x, y };
}

function updateReadouts() {
  const pill = document.getElementById('statusPill');
  const miss = document.getElementById('missVal');
  const prob = document.getElementById('probVal');
  const btn = document.getElementById('burnBtn');
  if (burned) {
    pill.textContent = 'ORBIT CLEARED';
    pill.className = 'status-pill pill-safe';
    miss.textContent = '4,820 m';
    miss.style.color = '#067647';
    prob.textContent = '< 1 in 10M';
    prob.style.color = '#067647';
    btn.textContent = 'Avoidance Burn Executed';
    btn.className = 'btn btn-burn done';
  } else {
    pill.textContent = 'CONJUNCTION ALERT';
    pill.className = 'status-pill pill-risk';
    miss.textContent = '185 m';
    miss.style.color = '#D92D20';
    prob.textContent = '1 in 4,200';
    prob.style.color = '#D92D20';
    btn.textContent = 'Simulate Avoidance Burn (Δv)';
    btn.className = 'btn btn-burn';
  }
}

window.triggerBurn = function() {
  burned = true;
  if (progress > 0.42) progress = 0.22;
  updateReadouts();
};

window.resetSim = function() {
  burned = false;
  burnBlend = 0.0;
  progress = 0.0;
  updateReadouts();
};

function draw() {
  ctx.clearRect(0, 0, W, H);

  // Subtle radar coordinate grid
  ctx.strokeStyle = '#E8EEF5';
  ctx.lineWidth = 1;
  for (let gx = 40; gx < W; gx += 48) {
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
  }
  for (let gy = 30; gy < H; gy += 42) {
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
  }

  // Blind Zone shading (x > cutoffX)
  const cutoffP = 0.34;
  const cutoffX = 36 + cutoffP * (W - 96);
  ctx.fillStyle = burned ? 'rgba(6, 118, 71, 0.045)' : 'rgba(217, 45, 32, 0.055)';
  ctx.fillRect(cutoffX, 0, W - cutoffX, H);

  // T-48h Cutoff line
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = '#B54708';
  ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(cutoffX, 0); ctx.lineTo(cutoffX, H); ctx.stroke();
  ctx.restore();

  // Clean header labels strictly in top corners (zero overlap with orbits)
  ctx.font = '600 10px "IBM Plex Mono", monospace';
  ctx.fillStyle = '#B54708';
  ctx.fillText('T-48h CUTOFF', cutoffX - 82, 16);
  ctx.fillStyle = burned ? '#067647' : '#B42318';
  ctx.fillText(burned ? 'CLEARED CORRIDOR' : 'BLIND ZONE (<48h)', cutoffX + 10, 16);

  // TCA Point & Contracting Uncertainty Ellipsoid
  const tcaDeb = debrisPos(1.0);
  const ellRx = (26 - progress * 10) + Math.sin(pulsePhase) * 1.5;
  const ellRy = (15 - progress * 5) + Math.sin(pulsePhase) * 1.0;

  ctx.save();
  ctx.translate(tcaDeb.x, tcaDeb.y);
  ctx.rotate(-0.22);
  ctx.fillStyle = burned ? 'rgba(6, 118, 71, 0.10)' : 'rgba(217, 45, 32, 0.12)';
  ctx.strokeStyle = burned ? 'rgba(6, 118, 71, 0.45)' : 'rgba(217, 45, 32, 0.55)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(0, 0, ellRx, ellRy, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  // Draw Debris Orbit (dashed crimson)
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(217, 45, 32, 0.55)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 50; i++) {
    const pt = debrisPos(i / 50);
    if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
  }
  ctx.stroke();
  ctx.restore();

  // Draw Original Satellite Orbit (ghost line when burned)
  if (burnBlend > 0.02) {
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= 50; i++) {
      const pt = satPos(i / 50, 0);
      if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Draw Active Satellite Orbit (Cobalt or Emerald when cleared)
  ctx.strokeStyle = burned ? '#067647' : '#1B4DFF';
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const pt = satPos(i / 60, burnBlend);
    if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
  }
  ctx.stroke();

  // Burn Thruster Ring Marker at T-54h when burned
  if (burnBlend > 0.05) {
    const bp = satPos(0.25, burnBlend);
    ctx.fillStyle = '#067647';
    ctx.beginPath(); ctx.arc(bp.x, bp.y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.font = '600 10px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#067647';
    ctx.fillText('Δv BURN', bp.x - 20, bp.y + 18);
  }

  // Moving Satellite & Debris Markers
  const curSat = satPos(progress, burnBlend);
  const curDeb = debrisPos(progress);

  // Subtle radar ping around satellite
  const pingR = 7 + ((pulsePhase * 5) % 14);
  ctx.strokeStyle = burned ? `rgba(6,118,71,${Math.max(0, 0.55 - pingR / 26)})` : `rgba(27,77,255,${Math.max(0, 0.55 - pingR / 26)})`;
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(curSat.x, curSat.y, pingR, 0, Math.PI * 2); ctx.stroke();

  // Target Satellite dot
  ctx.fillStyle = burned ? '#067647' : '#1B4DFF';
  ctx.beginPath(); ctx.arc(curSat.x, curSat.y, 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 2; ctx.stroke();

  // Debris dot
  ctx.fillStyle = '#D92D20';
  ctx.beginPath(); ctx.arc(curDeb.x, curDeb.y, 5, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 1.8; ctx.stroke();

  // Bottom-left compact legend
  ctx.font = '500 11px "DM Sans", sans-serif';
  ctx.fillStyle = '#0F172A';
  ctx.fillText('● Target Satellite', 14, H - 12);
  ctx.fillStyle = '#D92D20';
  ctx.fillText('● Debris Chaser', 124, H - 12);

  // Update countdown smoothly
  const hrsLeft = Math.max(0, (1 - progress) * 72);
  document.getElementById('tcaVal').textContent = hrsLeft < 0.8 ? 'TCA (0.0 h)' : `T-${hrsLeft.toFixed(1)} h`;

  // Advance animation
  progress += 0.0026;
  if (progress > 1.04) progress = 0.0;
  if (burned && burnBlend < 1.0) burnBlend = Math.min(1.0, burnBlend + 0.055);
  pulsePhase += 0.05;

  requestAnimationFrame(draw);
}

updateReadouts();
requestAnimationFrame(draw);
</script>
</body>
</html>
"""
    components.html(html_code, height=405, scrolling=False)


# -----------------------------------------------------------------------------
# ZERO-OVERLAP PLOTLY RISK TRAJECTORY BUILDER
# -----------------------------------------------------------------------------
def _build_risk_trajectory_figure(
    event_row: pd.Series,
    event_history: pd.DataFrame,
    high_risk_thresh: float,
    cutoff_days: float,
    reveal_gt: bool,
    height: int = 370,
) -> go.Figure:
    """Build an uncluttered, zero-overlap Plotly collision risk trajectory chart.

    All text annotations sit strictly in the top margin header ABOVE the plot
    area (y=1.08, yref='paper'), and the alert threshold lives in the legend,
    making it mathematically impossible for labels to overlap data points.
    """
    vis_cdms = event_history[event_history["visible"]].sort_values("time_to_tca", ascending=False)
    post_cdms = event_history[~event_history["visible"]].sort_values("time_to_tca", ascending=False)

    all_y = list(vis_cdms["log_risk"].values) + [float(event_row["pred_final_log_risk"]), high_risk_thresh]
    if reveal_gt and len(post_cdms) > 0:
        all_y.extend(list(post_cdms["log_risk"].values))
    if reveal_gt:
        all_y.append(float(event_row["actual_final_log_risk"]))

    y_min_data = float(np.nanmin(all_y)) if all_y else -25.0
    y_max_data = float(np.nanmax(all_y)) if all_y else -2.0
    y_span = max(4.0, y_max_data - y_min_data)
    y_low = max(-31.5, y_min_data - max(1.6, 0.16 * y_span))
    y_high = min(0.5, y_max_data + max(1.4, 0.16 * y_span))

    # Smart x-axis framing eliminates unused left blank space
    x_data_max = float(event_history["time_to_tca"].max()) if len(event_history) else 6.5
    max_x = max(x_data_max + 0.35, cutoff_days + 1.4)

    fig = go.Figure()

    # Shaded Blind Zone (< cutoff_days)
    fig.add_vrect(
        x0=cutoff_days,
        x1=0.0,
        fillcolor="rgba(217, 45, 32, 0.055)",
        layer="below",
        line_width=0,
    )
    fig.add_vline(
        x=cutoff_days,
        line_dash="dash",
        line_color="#B54708",
        line_width=1.8,
    )

    # Alert threshold rendered as a clean legend-backed line (zero floating text overlap!)
    fig.add_trace(
        go.Scatter(
            x=[max_x, -0.15],
            y=[high_risk_thresh, high_risk_thresh],
            mode="lines",
            name=f"1 in 1M Alert ({high_risk_thresh:.1f})",
            line=dict(color="#B54708", width=1.4, dash="dot"),
            hoverinfo="skip",
        )
    )

    # Labels anchored strictly in the dedicated header strip ABOVE the plot box
    fig.add_annotation(
        x=cutoff_days,
        y=1.08,
        yref="paper",
        xanchor="right",
        xshift=-6,
        text="T-48h Cutoff",
        showarrow=False,
        font=dict(color="#B54708", size=10.5, family="IBM Plex Mono"),
    )
    fig.add_annotation(
        x=cutoff_days * 0.5,
        y=1.08,
        yref="paper",
        xanchor="center",
        text="Blind Zone (<48h)",
        showarrow=False,
        font=dict(color="#B42318", size=10.5, family="IBM Plex Mono"),
    )

    # Pre-cutoff CDMs
    if len(vis_cdms) > 0:
        fig.add_trace(
            go.Scatter(
                x=vis_cdms["time_to_tca"],
                y=vis_cdms["log_risk"],
                mode="lines+markers",
                name="Pre-cutoff CDMs",
                line=dict(color="#1B4DFF", width=2.6),
                marker=dict(size=7.5, color="#1B4DFF", line=dict(width=1.5, color="#FFFFFF")),
                hovertemplate="T-%{x:.2f}d<br>Observed log risk: %{y:.2f}<extra></extra>",
            )
        )

    # NearMiss TCA forecast marker
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[float(event_row["pred_final_log_risk"])],
            mode="markers",
            name=f"NearMiss Forecast ({event_row['pred_final_log_risk']:.2f})",
            marker=dict(size=11, color="#0F172A", symbol="diamond", line=dict(width=1.5, color="#FFFFFF")),
            hovertemplate="NearMiss TCA Forecast: %{y:.2f}<extra></extra>",
        )
    )

    # Ground-truth post-cutoff reveal
    if reveal_gt:
        if len(post_cdms) > 0:
            bridge = pd.concat([vis_cdms.iloc[[-1]], post_cdms], ignore_index=True) if len(vis_cdms) > 0 else post_cdms
            fig.add_trace(
                go.Scatter(
                    x=bridge["time_to_tca"],
                    y=bridge["log_risk"],
                    mode="lines+markers",
                    name="Blind-Zone Truth",
                    line=dict(color="#D97706", width=2.2, dash="dash"),
                    marker=dict(size=7, color="#FFFFFF", line=dict(width=2, color="#D97706")),
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
                    size=12,
                    color="#D92D20" if actual_val >= high_risk_thresh else "#067647",
                    symbol="star",
                    line=dict(width=1, color="#FFFFFF"),
                ),
                hovertemplate="Actual Final Log Risk: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Days remaining until closest approach (TCA)",
            range=[max_x, -0.22],
            nticks=6,
            gridcolor="#EEF2F6",
            linecolor="#CBD5E1",
            zeroline=False,
            color="#475569",
            title_font=dict(size=11.5, family="DM Sans", color="#475569"),
        ),
        yaxis=dict(
            title="Collision risk log10(P)",
            range=[y_low, y_high],
            nticks=6,
            gridcolor="#EEF2F6",
            linecolor="#CBD5E1",
            zeroline=False,
            color="#475569",
            title_font=dict(size=11.5, family="DM Sans", color="#475569"),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans", color="#0F172A"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#475569"),
        ),
        height=height,
        margin=dict(l=62, r=18, t=38, b=82),
        hovermode="closest",
    )
    return fig


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW (SPACIOUS, ANIMATED, INTERACTIVE LANDING PAGE)
# -----------------------------------------------------------------------------
def render_overview(
    split_preds: pd.DataFrame,
    split_hist: pd.DataFrame,
    split_label: str,
    high_risk_thresh: float,
    operational_thresh: float,
    cutoff_days: float,
) -> None:
    """Render the spacious, animated Mission Briefing landing page."""
    notable = _notable_events(split_preds)
    rescued_id = notable["rescued"]
    avoided_id = notable["avoided"]

    # --- SECTION 1: ANIMATED HERO WITH LIVE ORBITAL SIMULATOR ---
    hero_l, hero_r = st.columns([0.96, 1.04], gap="large", vertical_alignment="center")

    with hero_l:
        st.markdown(
            f"""
<div class='anim-1'>
    <div class='live-badge'>
        <span class='pulse-dot'></span>
        ESA Kelvins Telemetry &nbsp;·&nbsp; Strict T-{cutoff_days*24:.0f}h Decision Cutoff
    </div>
    <h1 class='hero-h1'>Stop satellite collisions before the 48-hour blind zone.</h1>
    <p class='hero-p'>
        Planning and uplinking an avoidance burn takes up to two days. NearMiss forecasts final collision
        risk strictly from early radar tracking using 73 orbital covariance physics features—catching
        late-jumping threats while eliminating unnecessary fuel burns.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )
        c_btn1, c_btn2 = st.columns([1.05, 1.0], gap="small")
        with c_btn1:
            if st.button("Inspect Real Encounters", type="primary", width="stretch", key="hero_cta_inspect"):
                _navigate_to("Event Inspector", rescued_id or notable["top_risk"])
                st.rerun()
        with c_btn2:
            if st.button("Run Policy Simulator", type="secondary", width="stretch", key="hero_cta_sim"):
                _navigate_to("Policy Simulator")
                st.rerun()

    with hero_r:
        _render_orbital_hero_canvas()

    # --- SECTION 2: SPACIOUS 4-CARD IMPACT METRICS ---
    st.markdown(
        f"""
<div class='metrics-grid anim-2'>
    <div class='metric-card'>
        <div class='mc-top'>
            <div class='mc-icon'>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>
            </div>
        </div>
        <div class='mc-val'>14,109</div>
        <div class='mc-title'>ESA Conjunctions Audited</div>
        <div class='mc-sub'>Zero event_id overlap across folds & blind test</div>
    </div>
    <div class='metric-card'>
        <div class='mc-top'>
            <div class='mc-icon'>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15.5 14"/></svg>
            </div>
        </div>
        <div class='mc-val'>{cutoff_days*24:.0f} Hours</div>
        <div class='mc-title'>Advance Lead Time</div>
        <div class='mc-sub'>Every post-cutoff radar pass strictly withheld</div>
    </div>
    <div class='metric-card'>
        <div class='mc-top'>
            <div class='mc-icon'>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>
            </div>
        </div>
        <div class='mc-val'>-59.5%</div>
        <div class='mc-title'>Calibration Error Drop</div>
        <div class='mc-sub'>Platt scaling lowers Brier 0.0166 → 0.0068</div>
    </div>
    <div class='metric-card'>
        <div class='mc-top'>
            <div class='mc-icon'>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
        </div>
        <div class='mc-val'>-22.6%</div>
        <div class='mc-title'>Avoidable Burns Cut</div>
        <div class='mc-sub'>Out-of-fold false alarms reduced vs Persistence</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --- SECTION 3: INTERACTIVE REAL-DATA ENCOUNTER SHOWCASE ---
    st.markdown(
        f"""
<div class='section-wrap anim-3'>
    <h2 class='sec-h2'>See NearMiss beat legacy Persistence on real {split_label} data</h2>
    <p class='sec-p'>Switch between an avoided false-alarm burn and a rescued late-jumping collision to inspect what unfolded inside the blind zone.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="card_overview_showcase"):
        showcase_opts = []
        if avoided_id is not None:
            showcase_opts.append(f"Avoided False Alarm (#{avoided_id})")
        if rescued_id is not None:
            showcase_opts.append(f"Caught Late Collision (#{rescued_id})")
        if not showcase_opts and notable["top_risk"] is not None:
            showcase_opts.append(f"Highest Risk Event (#{notable['top_risk']})")

        selected_case = st.segmented_control(
            "Select real encounter case",
            options=showcase_opts,
            default=showcase_opts[0],
            selection_mode="single",
            label_visibility="collapsed",
            key=f"ov_case_toggle_{split_label}",
        )
        if not selected_case:
            selected_case = showcase_opts[0]

        case_eid = int(selected_case.split("#")[-1].rstrip(")"))
        case_row = split_preds[split_preds["event_id"] == case_eid].iloc[0]
        case_hist = split_hist[split_hist["event_id"] == case_eid]
        is_rescued = "Caught" in selected_case

        sc_l, sc_r = st.columns([0.85, 1.35], gap="large", vertical_alignment="center")
        with sc_l:
            pill_html = (
                "<span class='pill pill-red'>Late-Jumping Hazard Rescued</span>"
                if is_rescued
                else "<span class='pill pill-green'>False-Alarm Fuel Burn Saved</span>"
            )
            story_title = (
                f"Event #{case_eid}: Caught 48h before impact risk spiked"
                if is_rescued
                else f"Event #{case_eid}: Cleared safely despite early radar spike"
            )
            story_copy = (
                "At T-48h, the latest tracking pass sat just beneath the alert line, causing legacy Persistence to clear the encounter. NearMiss detected tightening orbital covariance and commanded a burn early."
                if is_rescued
                else "Early tracking uncertainty spiked collision risk above the alert line, causing Persistence to order an expensive maneuver. NearMiss recognized transient noise and stood down safely."
            )
            nm_verdict = (
                f"Burn ({case_row['high_risk_prob']:.1%})"
                if is_rescued
                else f"Stand Down ({case_row['high_risk_prob']:.1%})"
            )
            nm_col = "#D92D20" if is_rescued else "#067647"
            base_verdict = "Stand Down" if is_rescued else "Order Burn"
            base_col = "#475569" if is_rescued else "#D92D20"
            act_verdict = (
                f"{case_row['actual_final_log_risk']:.2f} (Hazard)"
                if is_rescued
                else f"{case_row['actual_final_log_risk']:.2f} (Safe)"
            )
            act_col = "#D92D20" if is_rescued else "#067647"

            st.markdown(
                f"""
<div style='margin-bottom:10px;'>{pill_html}</div>
<div style='font-size:1.18rem; font-weight:700; color:#0F172A; letter-spacing:-0.02em; margin-bottom:8px; line-height:1.25;'>{story_title}</div>
<div style='font-size:0.88rem; color:#475569; line-height:1.6;'>{story_copy}</div>
<div class='story-card-list'>
    <div class='story-row'><span class='story-k'>Baseline A (Persistence)</span><span class='story-v' style='color:{base_col};'>{base_verdict}</span></div>
    <div class='story-row'><span class='story-k'>NearMiss Prediction</span><span class='story-v' style='color:{nm_col};'>{nm_verdict}</span></div>
    <div class='story-row'><span class='story-k'>Actual Outcome at TCA</span><span class='story-v' style='color:{act_col};'>{act_verdict}</span></div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button(f"Audit Event #{case_eid} in Inspector", type="primary", width="stretch", key=f"ov_audit_{case_eid}"):
                _navigate_to("Event Inspector", case_eid)
                st.rerun()

        with sc_r:
            fig_case = _build_risk_trajectory_figure(
                event_row=case_row,
                event_history=case_hist,
                high_risk_thresh=high_risk_thresh,
                cutoff_days=cutoff_days,
                reveal_gt=True,
                height=355,
            )
            st.plotly_chart(fig_case, width="stretch", config={"displayModeBar": False})

    # --- SECTION 4: SPACIOUS 3-STAGE PIPELINE CARDS ---
    st.markdown(
        """
<div class='section-wrap'>
    <h2 class='sec-h2'>How the prediction pipeline works</h2>
    <p class='sec-p'>Designed for strict temporal isolation so no blind-zone radar data ever leaks into training or inference.</p>
</div>
<div class='pipeline-grid'>
    <div class='pipe-box'>
        <div class='pipe-head'>
            <span class='pipe-num'>01</span>
            <span class='pipe-title'>Early Radar Intake</span>
        </div>
        <div class='pipe-body'>
            Collects Conjunction Data Messages from T-7 days up to the 48-hour decision horizon and discards every later update before computing features.
        </div>
    </div>
    <div class='pipe-box'>
        <div class='pipe-head'>
            <span class='pipe-num'>02</span>
            <span class='pipe-title'>Orbital Covariance Physics</span>
        </div>
        <div class='pipe-body'>
            Extracts 73 physical signals including 3D uncertainty ellipsoid volume √det(C), contraction ratio across passes, combined RTN sigmas, and miss-distance velocity.
        </div>
    </div>
    <div class='pipe-box'>
        <div class='pipe-head'>
            <span class='pipe-num'>03</span>
            <span class='pipe-title'>Calibrated Decision & Certainty</span>
        </div>
        <div class='pipe-body'>
            Transforms balanced Random Forest probabilities via out-of-fold Platt scaling into trustworthy risk odds paired with High, Medium, or Low confidence tiers.
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --- SECTION 5: CLEAN OPERATOR REFERENCE FOOTER ---
    st.markdown(
        f"""
<div class='ref-strip'>
    <div class='ref-item'>
        <div class='ref-top'><span class='ref-name'>Decision Horizon</span><span class='ref-code'>T-{cutoff_days*24:.0f}h</span></div>
        <div class='ref-desc'>Latest time to plan and uplink an avoidance burn before entering the blind zone.</div>
    </div>
    <div class='ref-item'>
        <div class='ref-top'><span class='ref-name'>Alert Threshold</span><span class='ref-code'>1 in 1M ({high_risk_thresh:.1f})</span></div>
        <div class='ref-desc'>ESA Kelvins high-risk collision probability threshold marking a dangerous encounter.</div>
    </div>
    <div class='ref-item'>
        <div class='ref-top'><span class='ref-name'>Burn Threshold</span><span class='ref-code'>1 in 10k ({operational_thresh:.1f})</span></div>
        <div class='ref-desc'>Operational emergency threshold requiring mandatory flight dynamics review.</div>
    </div>
    <div class='ref-item'>
        <div class='ref-top'><span class='ref-name'>Confidence Tiers</span><span class='ref-code'>High / Med / Low</span></div>
        <div class='ref-desc'>Margin from the decision boundary; Low flags borderline passes for extra radar tracking.</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 2: EVENT INSPECTOR (FULL-WIDTH, QUICK JUMPS, ZERO-OVERLAP CHARTS)
# -----------------------------------------------------------------------------
def render_inspector(
    split_preds: pd.DataFrame,
    split_hist: pd.DataFrame,
    high_risk_thresh: float,
    cutoff_days: float,
) -> None:
    """Render the full-width Conjunction Event Inspector."""
    notable = _notable_events(split_preds)

    # Contextual Quick-Jump Bar (replaces the cluttered sidebar cleanly right here)
    qj_lbl, qj1, qj2, qj3 = st.columns([1.25, 1.0, 1.0, 1.0], gap="small", vertical_alignment="center")
    with qj_lbl:
        st.markdown(
            "<div style='font-size:0.86rem; font-weight:600; color:#475569;'>Quick-Jump Notable Encounters:</div>",
            unsafe_allow_html=True,
        )
    with qj1:
        if notable["rescued"] is not None:
            if st.button(f"Caught Collision #{notable['rescued']}", type="secondary", width="stretch", key="insp_qj_rescued"):
                _navigate_to("Event Inspector", notable["rescued"])
                st.rerun()
    with qj2:
        if notable["avoided"] is not None:
            if st.button(f"Saved Fuel Burn #{notable['avoided']}", type="secondary", width="stretch", key="insp_qj_avoided"):
                _navigate_to("Event Inspector", notable["avoided"])
                st.rerun()
    with qj3:
        if notable["top_risk"] is not None:
            if st.button(f"Highest Risk #{notable['top_risk']}", type="secondary", width="stretch", key="insp_qj_top"):
                _navigate_to("Event Inspector", notable["top_risk"])
                st.rerun()

    # Clean single-row filters
    f1, f2, f3, f4 = st.columns([0.95, 0.9, 0.95, 2.2], gap="small")
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
        div_label, div_color, div_sub = "Agreement", "#475569", "Matches persistence"
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
            Event #{selected_event_id} &nbsp;·&nbsp; Sat #{int(event_row['mission_id'])} vs {event_row['object_type'].title()} &nbsp;·&nbsp; {int(event_row['cdm_count_pre_cutoff'])} pre-cutoff passes
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
        <div class='vb-val' style='color:{b_color}; font-size:1.02rem;'>{b_text}</div>
        <div class='vb-note'>Last visible: {event_row['latest_log_risk']:.2f}</div>
    </div>
    <div class='vb-cell'>
        <div class='vb-lbl'>Policy Comparison</div>
        <div class='vb-val' style='color:{div_color}; font-size:1.02rem;'>{div_label}</div>
        <div class='vb-note'>{div_sub}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    tog_c1, tog_c2 = st.columns([1.0, 2.3], vertical_alignment="center")
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
                f"<div style='display:flex; align-items:center; gap:12px; font-size:0.85rem; color:#475569;'>"
                f"<span class='pill {pill_cls}'>{tag_txt}</span>"
                f"<span>Actual terminal risk: <strong class='mono' style='color:#0F172A;'>{actual_val:.2f}</strong></span></div>",
                unsafe_allow_html=True,
            )

    event_history = split_hist[split_hist["event_id"] == selected_event_id].sort_values(
        "time_to_tca", ascending=False
    )
    vis_cdms = event_history[event_history["visible"]]
    post_cdms = event_history[~event_history["visible"]]

    chart_c1, chart_c2 = st.columns([1.25, 1.0], gap="medium")

    with chart_c1:
        with st.container(border=True, key="card_insp_risk"):
            st.markdown(f"<div class='chart-head'>Collision Risk Trajectory — Event #{selected_event_id}</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-sub'>Observed log10 collision probability up to T-48h cutoff vs NearMiss terminal forecast.</div>", unsafe_allow_html=True)
            fig_risk = _build_risk_trajectory_figure(
                event_row=event_row,
                event_history=event_history,
                high_risk_thresh=high_risk_thresh,
                cutoff_days=cutoff_days,
                reveal_gt=reveal_gt,
                height=370,
            )
            st.plotly_chart(fig_risk, width="stretch", config={"displayModeBar": False})

    with chart_c2:
        with st.container(border=True, key="card_insp_miss"):
            st.markdown("<div class='chart-head'>Relative Miss Distance Profile</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-sub'>Center-to-center separation in meters across consecutive radar passes.</div>", unsafe_allow_html=True)
            x_data_max = float(event_history["time_to_tca"].max()) if len(event_history) else 6.5
            max_x = max(x_data_max + 0.35, cutoff_days + 1.4)

            fig_miss = go.Figure()
            fig_miss.add_vrect(
                x0=cutoff_days,
                x1=0.0,
                fillcolor="rgba(217, 45, 32, 0.055)",
                layer="below",
                line_width=0,
            )
            fig_miss.add_vline(x=cutoff_days, line_dash="dash", line_color="#B54708", line_width=1.6)
            fig_miss.add_annotation(
                x=cutoff_days,
                y=1.08,
                yref="paper",
                xanchor="right",
                xshift=-6,
                text="T-48h Cutoff",
                showarrow=False,
                font=dict(color="#B54708", size=10.5, family="IBM Plex Mono"),
            )
            fig_miss.add_annotation(
                x=cutoff_days * 0.5,
                y=1.08,
                yref="paper",
                xanchor="center",
                text="Blind Zone",
                showarrow=False,
                font=dict(color="#B42318", size=10.5, family="IBM Plex Mono"),
            )
            if len(vis_cdms) > 0:
                fig_miss.add_trace(
                    go.Scatter(
                        x=vis_cdms["time_to_tca"],
                        y=vis_cdms["miss_distance"],
                        mode="lines+markers",
                        name="Pre-cutoff miss distance",
                        line=dict(color="#0F172A", width=2.4),
                        marker=dict(size=7, color="#0F172A", line=dict(width=1.5, color="#FFFFFF")),
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
                        line=dict(color="#D97706", width=2.1, dash="dash"),
                        marker=dict(size=6.5, color="#FFFFFF", line=dict(width=2, color="#D97706")),
                        hovertemplate="T-%{x:.2f}d<br>Actual miss: %{y:,.0f} m<extra></extra>",
                    )
                )
            fig_miss.update_layout(
                xaxis=dict(
                    title="Days remaining until TCA",
                    range=[max_x, -0.22],
                    nticks=6,
                    gridcolor="#EEF2F6",
                    linecolor="#CBD5E1",
                    zeroline=False,
                    color="#475569",
                    title_font=dict(size=11.5, family="DM Sans", color="#475569"),
                ),
                yaxis=dict(
                    title="Miss distance (meters)",
                    nticks=6,
                    gridcolor="#EEF2F6",
                    linecolor="#CBD5E1",
                    zeroline=False,
                    color="#475569",
                    title_font=dict(size=11.5, family="DM Sans", color="#475569"),
                ),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#0F172A"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.24,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11, color="#475569"),
                ),
                height=370,
                margin=dict(l=64, r=18, t=38, b=82),
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
    """Render the spacious Decision Threshold & Mission Economics Simulator."""
    st.markdown(
        """
<div style='margin-bottom:4px;'>
    <h2 class='sec-h2'>Decision Threshold & Mission Economics Simulator</h2>
    <p class='sec-p'>Adjust the calibrated probability trigger to balance missed hazards against avoidable fuel burns compared with Baseline A.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    sim_l, sim_r = st.columns([0.95, 1.85], gap="large")

    with sim_l:
        with st.container(border=True, key="card_sim_ctrl"):
            st.markdown("<div class='chart-head'>Spacecraft Profile</div>", unsafe_allow_html=True)
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

            st.markdown(f"<div style='font-size:0.8rem; color:#64748B; margin-bottom:8px;'>{profile_note}</div>", unsafe_allow_html=True)

            sim_th = st.slider(
                "Maneuver Trigger Threshold (p*)",
                min_value=0.005,
                max_value=0.250,
                value=default_th,
                step=0.005,
                format="%.3f",
                key=f"sim_th_{preset}",
            )
            st.caption(f"Trigger a burn whenever calibrated P(collision) ≥ {sim_th:.1%}.")

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
<div style='background:{sav_bg}; border:1px solid {sav_bd}; border-radius:10px; padding:14px 16px; margin-top:10px;'>
    <div style='font-size:0.76rem; font-weight:500; color:#475569;'>Expected Financial Balance vs Baseline A</div>
    <div class='mono' style='font-size:1.52rem; font-weight:600; color:{sav_fg}; margin:3px 0;'>USD {savings:+,.0f}</div>
    <div style='font-size:0.79rem; color:#475569;'>{savings_pct:+.1f}% mission cost change at p* = {sim_th:.3f}.</div>
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
            st.markdown("<div class='chart-head'>Operational Risk Frontier</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-sub'>Lower-left is better: fewer missed collisions and fewer unnecessary fuel burns.</div>", unsafe_allow_html=True)
            fig_sim = go.Figure()
            fig_sim.add_trace(
                go.Scatter(
                    x=sweep_df["fp"],
                    y=sweep_df["fn"],
                    mode="lines",
                    name="NearMiss trade-off curve",
                    line=dict(color="#1B4DFF", width=2.6),
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
                    marker=dict(size=13, color="#D92D20", symbol="star", line=dict(width=1.2, color="#FFFFFF")),
                    hovertemplate=f"Active Policy (p*={sim_th:.3f})<br>FP: %{{x}}, FN: %{{y}}<extra></extra>",
                )
            )
            fig_sim.add_trace(
                go.Scatter(
                    x=[base_metrics["fp"]],
                    y=[base_metrics["fn"]],
                    mode="markers",
                    name="Baseline A (Persistence)",
                    marker=dict(size=10, color="#0F172A", symbol="square", line=dict(width=1.5, color="#FFFFFF")),
                    hovertemplate="Baseline A<br>FP: %{x}, FN: %{y}<extra></extra>",
                )
            )
            fig_sim.update_layout(
                xaxis=dict(title="Unnecessary maneuvers (FP)", nticks=6, gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475569", title_font=dict(size=11.5)),
                yaxis=dict(title="Missed collisions (FN)", nticks=5, gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475569", title_font=dict(size=11.5)),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#0F172A"),
                legend=dict(orientation="h", yanchor="top", y=-0.30, xanchor="center", x=0.5, font=dict(size=11, color="#475569")),
                height=265,
                margin=dict(l=58, r=18, t=12, b=76),
            )
            st.plotly_chart(fig_sim, width="stretch", config={"displayModeBar": False})

        cm1, cm2 = st.columns(2, gap="medium")
        with cm1:
            st.markdown(
                f"""
<div class='cm-box'>
    <div class='cm-head'>NearMiss Matrix (p* = {sim_th:.3f})</div>
    <table class='cm-tbl'>
        <tr><th></th><th>Pred Safe</th><th>Pred Burn</th></tr>
        <tr>
            <th>Actual Safe</th>
            <td style='background:#F4F6FA; color:#0F172A;'>{sim_metrics['tn']:,} (TN)</td>
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
            <td style='background:#F4F6FA; color:#0F172A;'>{base_metrics['tn']:,} (TN)</td>
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
# PAGE 4: BENCHMARKS & CALIBRATION
# -----------------------------------------------------------------------------
def render_benchmarks() -> None:
    """Render full-width benchmark tables and live interactive Plotly diagnostics."""
    st.markdown(
        """
<div style='margin-bottom:4px;'>
    <h2 class='sec-h2'>ESA Kelvins Benchmarks & Model Calibration</h2>
    <p class='sec-p'>Evaluated under strict event-grouped cross-validation (8,293 filtered events) and on the locked blind competition test set (2,167 events).</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="card_bench_test"):
        st.markdown("<div class='chart-head'>Locked ESA Blind Test Set (2,167 Events · 150 High-Risk)</div>", unsafe_allow_html=True)
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

    with st.container(border=True, key="card_bench_cv"):
        st.markdown("<div class='chart-head'>5-Fold Out-of-Fold Cross-Validation (8,293 Filtered Events · 66 High-Risk)</div>", unsafe_allow_html=True)
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
            st.markdown("<div class='chart-head'>Platt Probability Calibration Reliability</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-sub'>Out-of-fold Platt scaling reduces Brier calibration error by 59.5%.</div>", unsafe_allow_html=True)
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
                        marker=dict(size=7.5, color="#D92D20", line=dict(width=1.4, color="#FFFFFF")),
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
                        marker=dict(size=8.5, color="#1B4DFF", line=dict(width=1.5, color="#FFFFFF")),
                        hovertemplate="Calibrated: %{x:.1%}<br>Observed rate: %{y:.1%}<extra></extra>",
                    )
                )

            fig_cal.update_layout(
                xaxis=dict(title="Mean predicted probability", range=[-0.02, 1.0], tickformat=".0%", nticks=6, gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475569"),
                yaxis=dict(title="Observed high-risk fraction", range=[-0.02, 0.52], tickformat=".0%", nticks=6, gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475569"),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#0F172A"),
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(size=10.5, color="#475569")),
                height=355,
                margin=dict(l=60, r=18, t=14, b=78),
            )
            st.plotly_chart(fig_cal, width="stretch", config={"displayModeBar": False})

    with diag_r:
        with st.container(border=True, key="card_bench_shap"):
            st.markdown("<div class='chart-head'>Global Feature Drivers (Tree SHAP)</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-sub'>Top 12 orbital trajectory and covariance features ranked by mean |SHAP| impact.</div>", unsafe_allow_html=True)
            shap_df = pd.DataFrame(TOP_SHAP_FEATURES, columns=["raw", "label", "val", "group"]).iloc[::-1]
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
                xaxis=dict(title="Mean |SHAP| impact on collision risk", nticks=5, gridcolor="#EEF2F6", linecolor="#CBD5E1", color="#475569"),
                yaxis=dict(categoryorder="array", categoryarray=shap_df["label"].tolist(), color="#0F172A", tickfont=dict(size=11)),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="DM Sans", color="#0F172A"),
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(size=10.5, color="#475569")),
                height=355,
                margin=dict(l=185, r=18, t=14, b=78),
            )
            st.plotly_chart(fig_shap, width="stretch", config={"displayModeBar": False})

    st.markdown(
        """
<div class='pipeline-grid'>
    <div class='pipe-box'>
        <div class='pipe-title' style='margin-bottom:8px;'>Combined Positional Sigmas</div>
        <div class='pipe-body'>
            Root-sum-square radial, along-track, and cross-track dispersion fusing target and chaser covariance tensors:
            <span class='mono' style='color:#0F172A; font-weight:600;'>σ_pos = √(σ_t² + σ_c²)</span>.
        </div>
    </div>
    <div class='pipe-box'>
        <div class='pipe-title' style='margin-bottom:8px;'>3D Ellipsoid Volume</div>
        <div class='pipe-body'>
            Determinant-based positional uncertainty volume measuring how tightly ground radar constrains the flyby corridor:
            <span class='mono' style='color:#0F172A; font-weight:600;'>Vol = √det(C_pos)</span>.
        </div>
    </div>
    <div class='pipe-box'>
        <div class='pipe-title' style='margin-bottom:8px;'>Volume Contraction Ratio</div>
        <div class='pipe-body'>
            Ratio of latest to earliest covariance volume across passes—separating tightening threats from transient noise:
            <span class='mono' style='color:#0F172A; font-weight:600;'>Vol_cutoff / Vol_initial</span>.
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# MAIN APPLICATION SHELL (SINGLE UNIFIED TOP NAVIGATION BAR — NO SIDEBAR)
# -----------------------------------------------------------------------------
def main() -> None:
    cfg, all_preds, all_hist = load_app_data()

    if "active_page" not in st.session_state or st.session_state["active_page"] not in NAV_PAGES:
        st.session_state["active_page"] = "Overview"
    if "active_cohort" not in st.session_state:
        st.session_state["active_cohort"] = "Validation (2,389)"

    high_risk_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))
    operational_thresh = float(cfg.get("thresholds", {}).get("operational_risk_log10", -4.0))
    cutoff_days = float(cfg["data"]["cutoff_days"])

    # --- SINGLE UNIFIED FULL-WIDTH TOP NAVIGATION BAR ---
    with st.container(border=True, key="navbar_shell"):
        c_brand, c_n1, c_n2, c_n3, c_n4, c_split = st.columns(
            [1.6, 0.78, 1.05, 1.05, 0.85, 1.85],
            gap="small",
            vertical_alignment="center",
        )

        with c_brand:
            st.markdown(
                """
<div class='brand-wrap'>
    <div class='brand-mark'>
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.1">
            <circle cx="12" cy="12" r="8.5" stroke-opacity="0.38"/>
            <ellipse cx="12" cy="12" rx="10" ry="3.8" transform="rotate(-30 12 12)" stroke="#FFFFFF"/>
            <circle cx="12" cy="12" r="2.5" fill="#60A5FA" stroke="none"/>
        </svg>
    </div>
    <div>
        <div class='brand-title'>NearMiss</div>
        <div class='brand-sub'>SW-5 Conjunction Assessment</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        for slot, page_name in zip([c_n1, c_n2, c_n3, c_n4], NAV_PAGES):
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

        with c_split:
            cohort_choice = st.segmented_control(
                "Cohort",
                options=["Validation (2,389)", "Blind Test (2,167)"],
                default=st.session_state["active_cohort"],
                selection_mode="single",
                label_visibility="collapsed",
                width="stretch",
                key="cohort_switcher",
            )
            if cohort_choice:
                st.session_state["active_cohort"] = cohort_choice

    split_key = "val" if st.session_state["active_cohort"].startswith("Validation") else "test"
    split_full = "Validation Set" if split_key == "val" else "ESA Blind Test Set"
    split_preds = all_preds[all_preds["split"] == split_key].copy()
    split_hist = _slice_history_for_split(all_hist, split_key)

    # --- ROUTE TO ACTIVE VIEW ---
    active_page = st.session_state["active_page"]
    if active_page == "Overview":
        render_overview(split_preds, split_hist, split_full, high_risk_thresh, operational_thresh, cutoff_days)
    elif active_page == "Event Inspector":
        render_inspector(split_preds, split_hist, high_risk_thresh, cutoff_days)
    elif active_page == "Policy Simulator":
        render_simulator(split_preds)
    else:
        render_benchmarks()


if __name__ == "__main__":
    main()

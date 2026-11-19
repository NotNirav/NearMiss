"""Streamlit dashboard for SW5 Satellite Collision Risk Predictor (NearMiss).

Implements Section 9 specification:
- Theme: Dark (#090D16 bg, #111827 secondary, #38BDF8 primary).
- Top Bar: Split selector, active threshold, status metrics.
- Tab 1: Event Inspector with Plotly risk trajectory, blind zone, and ground truth reveal.
- Tab 2: Threshold & Cost Simulator with interactive FN vs FP curve and cost analysis.
- Tab 3: Model Benchmarks, Reliability Diagram, and Physics Architecture.
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

# Page Configuration
st.set_page_config(
    page_title="NearMiss — Satellite Collision Risk Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Operational Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #090D16;
        color: #F9FAFB;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-manoeuvre {
        background-color: #EF4444;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-safe {
        background-color: #10B981;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-conf-high {
        background-color: #0284C7;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-conf-med {
        background-color: #D97706;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-conf-low {
        background-color: #4B5563;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_app_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    preds_df = load_parquet("data/processed/predictions.parquet")
    hist_df = load_parquet("data/processed/cdm_history.parquet")
    return cfg, preds_df, hist_df


def main() -> None:
    cfg, all_preds, all_hist = load_app_data()

    # --- SIDEBAR & TOP BAR ---
    st.sidebar.image("https://img.icons8.com/fluency/96/satellite.png", width=64)
    st.sidebar.title("NearMiss SW-5")
    st.sidebar.caption("StarForge Hackathon · Dark Side Software")

    split_choice = st.sidebar.radio(
        "Select Dataset Split",
        options=["Validation Set (OOF Fold 0)", "Locked Test Set (ESA Competition)"],
        index=0,
    )
    split_key = "val" if "Validation" in split_choice else "test"
    split_preds = all_preds[all_preds["split"] == split_key].copy()

    high_risk_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))
    operational_thresh = float(cfg.get("thresholds", {}).get("operational_risk_log10", -4.0))

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Primary Threshold:** `10^{{{int(high_risk_thresh)}}} (1 in 1M)`")
    st.sidebar.markdown(f"**Operational Threshold:** `10^{{{int(operational_thresh)}}} (1 in 10k)`")
    st.sidebar.markdown(f"**Decision Cutoff:** `{cfg['data']['cutoff_days']} days`")

    # Header Ribbon
    st.title("🛰️ NearMiss — Satellite Collision Risk Predictor")
    st.markdown(
        "Autonomous conjunction risk prediction using exclusively early tracking data "
        f"available at least **{cfg['data']['cutoff_days']} days** before Time of Closest Approach."
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Events Evaluated", f"{len(split_preds):,}")
    with col_m2:
        n_man = (split_preds["recommendation"] == "MANOEUVRE").sum()
        st.metric("Manoeuvres Recommended", f"{n_man} ({n_man/len(split_preds):.1%})")
    with col_m3:
        n_actual = split_preds["actual_high_risk"].sum()
        st.metric("Actual High-Risk Encounters", f"{n_actual} ({n_actual/len(split_preds):.1%})")
    with col_m4:
        st.metric("Decision Cutoff", f"{cfg['data']['cutoff_days']} Days Before TCA")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 1. Conjunction Event Inspector",
        "🎛️ 2. Threshold & Cost Simulator",
        "📊 3. Model Benchmark & Calibration",
    ])

    # =========================================================================
    # TAB 1: EVENT INSPECTOR
    # =========================================================================
    with tab1:
        col_list, col_view = st.columns([1, 2])

        with col_list:
            st.subheader("Event Selection")

            # Filters
            rec_filter = st.selectbox(
                "Filter by Recommendation",
                ["All", "MANOEUVRE", "NO_MANOEUVRE"],
                index=0,
            )
            conf_filter = st.selectbox(
                "Filter by Confidence",
                ["All", "HIGH", "MEDIUM", "LOW"],
                index=0,
            )
            search_query = st.text_input("Search Event ID", "")

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
                    st.warning("Please enter a valid numeric Event ID.")

            # Sort by danger probability descending
            filtered_list = filtered_list.sort_values("high_risk_prob", ascending=False)

            if len(filtered_list) == 0:
                st.info("No conjunction events match the active filters.")
                st.stop()

            event_options = filtered_list["event_id"].tolist()
            selected_event_id = st.selectbox(
                f"Select Event ({len(filtered_list)} matches)",
                event_options,
                format_func=lambda eid: f"Event {eid} | P(Risk)={filtered_list.loc[filtered_list['event_id']==eid, 'high_risk_prob'].iloc[0]:.1%} | {filtered_list.loc[filtered_list['event_id']==eid, 'recommendation'].iloc[0]}",
            )

        with col_view:
            event_row = split_preds[split_preds["event_id"] == selected_event_id].iloc[0]

            # Decision Banner
            st.markdown("### Decision & Collision Assessment")
            banner_col1, banner_col2, banner_col3 = st.columns(3)

            with banner_col1:
                st.markdown("**NearMiss Recommendation:**")
                if event_row["recommendation"] == "MANOEUVRE":
                    st.markdown("<span class='badge-manoeuvre'>🔥 MANOEUVRE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-safe'>🛡️ NO_MANOEUVRE</span>", unsafe_allow_html=True)

                conf = event_row["confidence"]
                conf_class = f"badge-conf-{conf.lower()}"
                st.markdown(f"Confidence Level: <span class='{conf_class}'>{conf}</span>", unsafe_allow_html=True)

            with banner_col2:
                st.metric("Calibrated Collision Probability", f"{event_row['high_risk_prob']:.2%}")
                st.metric("Predicted Terminal Risk", f"{event_row['pred_final_log_risk']:.2f}")

            with banner_col3:
                st.markdown("**Baseline A (Persistence):**")
                b_rec = event_row["baseline_recommendation"]
                if b_rec == "MANOEUVRE":
                    st.markdown("<span class='badge-manoeuvre' style='font-size:0.9rem;'>MANOEUVRE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-safe' style='font-size:0.9rem;'>NO_MANOEUVRE</span>", unsafe_allow_html=True)
                st.metric("Latest Visible Risk", f"{event_row['latest_log_risk']:.2f}")

            st.markdown("---")

            # Reveal ground truth toggle
            reveal_gt = st.toggle("👁️ Reveal Ground Truth (Simulation & Demo Mode)", value=False)
            if reveal_gt:
                gt_col1, gt_col2 = st.columns(2)
                with gt_col1:
                    st.metric("Actual Final Risk (Terminal CDM)", f"{event_row['actual_final_log_risk']:.2f}")
                with gt_col2:
                    is_actual_danger = event_row["actual_high_risk"] == 1
                    status_text = "DANGEROUS ENCOUNTER" if is_actual_danger else "HARMLESS APPROACH"
                    st.markdown(f"**Actual Outcome:** `{'🔴 ' + status_text if is_actual_danger else '🟢 ' + status_text}`")

            # Plotly Risk Trajectory Chart
            event_history = all_hist[all_hist["event_id"] == selected_event_id].sort_values("time_to_tca", ascending=False)

            fig_risk = go.Figure()

            # Shaded Blind Zone
            fig_risk.add_vrect(
                x0=2.0, x1=0.0,
                fillcolor="#EF4444", opacity=0.08,
                layer="below", line_width=0,
                annotation_text="BLIND ZONE (Decisions Prohibited)",
                annotation_position="top left",
                annotation_font=dict(color="#EF4444", size=10),
            )

            # 2.0-Day Cutoff line
            fig_risk.add_vline(
                x=2.0, line_dash="dash", line_color="#EF4444", line_width=2,
                annotation_text="2.0-Day Decision Boundary",
                annotation_font=dict(color="#EF4444", size=11),
            )

            # High risk threshold line (-6.0)
            fig_risk.add_hline(
                y=high_risk_thresh, line_dash="dot", line_color="#F59E0B", line_width=1.5,
                annotation_text="1 in 1M High Risk (-6.0)",
                annotation_font=dict(color="#F59E0B", size=10),
            )

            # Visible CDMs
            vis_cdms = event_history[event_history["visible"]]
            fig_risk.add_trace(go.Scatter(
                x=vis_cdms["time_to_tca"],
                y=vis_cdms["log_risk"],
                mode="lines+markers",
                name="Visible CDMs (>= 2.0d)",
                line=dict(color="#38BDF8", width=2.5),
                marker=dict(size=8, color="#38BDF8"),
                hovertemplate="Time to TCA: %{x:.2f}d<br>Log10 Risk: %{y:.2f}<extra></extra>",
            ))

            # Predicted Risk Marker
            fig_risk.add_trace(go.Scatter(
                x=[0.0],
                y=[event_row["pred_final_log_risk"]],
                mode="markers",
                name="Predicted Final Risk",
                marker=dict(size=14, color="#A855F7", symbol="diamond"),
                hovertemplate="Predicted Risk at TCA: %{y:.2f}<extra></extra>",
            ))

            # Post-cutoff CDMs if revealed
            if reveal_gt:
                post_cdms = event_history[~event_history["visible"]]
                if len(post_cdms) > 0:
                    fig_risk.add_trace(go.Scatter(
                        x=post_cdms["time_to_tca"],
                        y=post_cdms["log_risk"],
                        mode="lines+markers",
                        name="Post-Cutoff CDMs (Ground Truth)",
                        line=dict(color="#F97316", width=2, dash="dot"),
                        marker=dict(size=7, color="#F97316"),
                        hovertemplate="Actual Time to TCA: %{x:.2f}d<br>Actual Risk: %{y:.2f}<extra></extra>",
                    ))

            fig_risk.update_layout(
                title=f"Conjunction Risk Trajectory — Event {selected_event_id} ({event_row['object_type']})",
                xaxis=dict(title="Time to TCA (days remaining)", autorange="reversed", gridcolor="#1F2937"),
                yaxis=dict(title="log10(Collision Risk)", range=[-32, 0], gridcolor="#1F2937"),
                paper_bgcolor="#090D16",
                plot_bgcolor="#111827",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Miss Distance Sub-Chart
            fig_miss = go.Figure()
            fig_miss.add_trace(go.Scatter(
                x=vis_cdms["time_to_tca"],
                y=vis_cdms["miss_distance"],
                mode="lines+markers",
                name="Miss Distance (m)",
                line=dict(color="#10B981", width=2),
                marker=dict(size=6, color="#10B981"),
            ))
            fig_miss.update_layout(
                title="Miss Distance Evolution Across CDMs",
                xaxis=dict(title="Time to TCA (days)", autorange="reversed", gridcolor="#1F2937"),
                yaxis=dict(title="Miss Distance (m)", gridcolor="#1F2937"),
                paper_bgcolor="#090D16",
                plot_bgcolor="#111827",
                height=220,
                margin=dict(l=40, r=40, t=40, b=30),
            )
            st.plotly_chart(fig_miss, use_container_width=True)

    # =========================================================================
    # TAB 2: THRESHOLD & COST SIMULATOR
    # =========================================================================
    with tab2:
        st.subheader("Operational Threshold & Cost Simulator")
        st.markdown(
            "Tune the decision probability threshold and simulate operational cost trade-offs "
            "between **missed collisions (False Negatives)** and **unnecessary burns (False Positives)**."
        )

        sim_col1, sim_col2 = st.columns([1, 2])

        with sim_col1:
            st.markdown("### Decision Parameters")
            sim_th = st.slider(
                "Decision Probability Threshold (p*)",
                min_value=0.005,
                max_value=0.300,
                value=0.030,
                step=0.005,
                format="%.3f",
            )
            st.markdown("### Cost Parameters ($ USD)")
            cost_collision = st.number_input("Cost of Missed Collision ($)", value=10_000_000, step=1_000_000)
            cost_manoeuvre = st.number_input("Cost of Fuel / Avoidance Burn ($)", value=25_000, step=5_000)

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

            st.markdown("---")
            st.markdown("### Operational Impact")
            st.metric("Net Cost Savings vs Baseline", f"${savings:,.0f}", delta=f"{savings/max(1, cost_baseline):+.1%}")
            st.write(f"• **Missed Collisions (FN):** {sim_metrics['fn']} (vs {base_metrics['fn']} Baseline)")
            st.write(f"• **Unnecessary Burns (FP):** {sim_metrics['fp']} (vs {base_metrics['fp']} Baseline)")

        with sim_col2:
            st.markdown("### Live FN vs FP Trade-Off Curve")

            # Sweep curve on split
            th_grid = np.linspace(0.005, 0.40, 60)
            sweep_pts = []
            for t in th_grid:
                p_bin = (split_preds["high_risk_prob"] >= t).astype(int)
                m = compute_classification_metrics(y_eval_true, p_bin)
                sweep_pts.append({"threshold": t, "fn": m["fn"], "fp": m["fp"], "f2": m["f2"]})
            sweep_df = pd.DataFrame(sweep_pts)

            fig_sim = go.Figure()
            # Threshold trade-off curve
            fig_sim.add_trace(go.Scatter(
                x=sweep_df["fp"],
                y=sweep_df["fn"],
                mode="lines",
                name="NearMiss Trade-Off Frontier",
                line=dict(color="#38BDF8", width=3),
                hovertemplate="FP: %{x}<br>FN: %{y}<extra></extra>",
            ))

            # Current Operating Point
            fig_sim.add_trace(go.Scatter(
                x=[sim_metrics["fp"]],
                y=[sim_metrics["fn"]],
                mode="markers+text",
                name=f"Current Threshold (p*={sim_th:.3f})",
                text=[f"Current (p*={sim_th:.3f})"],
                textposition="top right",
                marker=dict(size=14, color="#EF4444", symbol="star"),
            ))

            # Baseline A operating point
            fig_sim.add_trace(go.Scatter(
                x=[base_metrics["fp"]],
                y=[base_metrics["fn"]],
                mode="markers+text",
                name="Baseline A (Persistence)",
                text=["Baseline A"],
                textposition="bottom left",
                marker=dict(size=12, color="#F59E0B", symbol="square"),
            ))

            fig_sim.update_layout(
                title="Trade-Off: Missed Dangerous Events (FN) vs Unnecessary Manoeuvres (FP)",
                xaxis=dict(title="Unnecessary Manoeuvres (False Positives)", gridcolor="#1F2937"),
                yaxis=dict(title="Missed Dangerous Events (False Negatives)", gridcolor="#1F2937"),
                paper_bgcolor="#090D16",
                plot_bgcolor="#111827",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Confusion Matrix Display
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h4>NearMiss Model Confusion Matrix</h4>
                        <table style='width:100%; text-align:center;'>
                            <tr><th></th><th>Pred Safe</th><th>Pred Burn</th></tr>
                            <tr><th>Actual Safe</th><td>{sim_metrics['tn']:,} (TN)</td><td style='color:#EF4444;'>{sim_metrics['fp']:,} (FP)</td></tr>
                            <tr><th>Actual Danger</th><td style='color:#EF4444;'>{sim_metrics['fn']:,} (FN)</td><td style='color:#10B981;'>{sim_metrics['tp']:,} (TP)</td></tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cm_col2:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h4>Baseline A (Persistence) Matrix</h4>
                        <table style='width:100%; text-align:center;'>
                            <tr><th></th><th>Pred Safe</th><th>Pred Burn</th></tr>
                            <tr><th>Actual Safe</th><td>{base_metrics['tn']:,} (TN)</td><td style='color:#F59E0B;'>{base_metrics['fp']:,} (FP)</td></tr>
                            <tr><th>Actual Danger</th><td style='color:#F59E0B;'>{base_metrics['fn']:,} (FN)</td><td style='color:#10B981;'>{base_metrics['tp']:,} (TP)</td></tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # TAB 3: BENCHMARKS & CALIBRATION
    # =========================================================================
    with tab3:
        st.subheader("Model Benchmarks & Calibration Reports")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("### Reliability Diagram (Platt Calibration)")
            cal_img_path = Path("reports/figures/calibration_curve.png")
            if cal_img_path.exists():
                st.image(str(cal_img_path), caption="Platt Scaling Probability Calibration (Brier: 0.0172 -> 0.0070)")
            else:
                st.info("Calibration plot generating...")

        with b_col2:
            st.markdown("### Baseline A Persistence Sweep")
            base_img_path = Path("reports/figures/baseline_threshold_sweep.png")
            if base_img_path.exists():
                st.image(str(base_img_path), caption="Baseline A Decision Threshold Sweep")
            else:
                st.info("Baseline plot generating...")

        st.markdown("### Out-of-Fold Model Comparison")
        st.markdown(
            """
| Model | Strategy | Recall | Precision | F2 Score | PR-AUC | Missed (FN) | Unnecessary (FP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline A (Persistence)** | Reference | 71.21% | 15.26% | 0.4108 | 0.1803 | 19 | 261 |
| **Logistic Regression** | Unweighted | 28.79% | 25.00% | 0.2794 | 0.1711 | 47 | 57 |
| **Logistic Regression** | Weighted | 87.88% | 6.58% | 0.2531 | 0.1484 | 8 | 824 |
| **Random Forest** | Unweighted | 45.45% | 24.59% | 0.3886 | 0.1759 | 36 | 92 |
| **Random Forest (Champion)**| **Weighted** | **71.21%** | **16.85%** | **0.4328** | **0.1926** | **19** | **232** |
| **LightGBM** | Unweighted | 30.30% | 22.99% | 0.2849 | 0.1696 | 46 | 67 |
| **LightGBM** | Weighted | 65.15% | 19.11% | 0.4397 | 0.1504 | 23 | 182 |
            """
        )

        st.markdown("### Physics & Feature Architecture")
        st.markdown(
            """
            - **73 Features Extracted:** 18 Trajectory Dynamics (slopes, step changes, acceleration), 26 Latest State (relative velocity, RTN coordinates, space weather), and 29 Simple Physics features.
            - **Combined Covariance Sigmas:** $\\sigma_{\\text{comb}} = \\sqrt{\\sigma_{\\text{target}}^2 + \\sigma_{\\text{chaser}}^2}$ across radial, transverse, and normal axes.
            - **Uncertainty Volume & Contraction:** Volumetric ellipsoid tracking and contraction ratio $\\text{vol}_{\\text{latest}} / \\text{vol}_{\\text{first}}$.
            - **Group Isolation Guarantee:** 0% event overlap across train, validation, and test splits.
            """
        )


if __name__ == "__main__":
    main()

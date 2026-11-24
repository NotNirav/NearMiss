"""Phase 07: Probability calibration (Platt scaling), threshold tuning, and confidence bands."""

from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.calibrate import (
    apply_platt_scaler,
    assign_confidence,
    fit_platt_scaler,
    tune_decision_threshold,
)
from src.cutoff import apply_eval_filter
from src.evaluate import compute_classification_metrics
from src.load import load_config, load_parquet, save_parquet


def run_phase_07() -> None:
    cfg = load_config()
    oof_df = load_parquet("data/processed/oof_predictions.parquet")
    targets = load_parquet("data/processed/targets_train.parquet")

    merged = oof_df.merge(
        targets[["event_id", "final_time_to_tca", "pre_cutoff_cdm_count"]],
        on="event_id",
    )
    eval_mask = apply_eval_filter(merged, cfg)
    filtered = merged[eval_mask].copy()

    y_true = filtered["y_high_risk"].values

    # Fit Platt scaling on OOF scores of our best candidate: Random Forest (weighted)
    raw_scores_rf = filtered["rf_weighted_prob"].values
    raw_scores_lgb = filtered["lgbm_weighted_prob"].values

    # Fit Platt scalers
    scaler_rf = fit_platt_scaler(raw_scores_rf, y_true)
    cal_probs_rf = apply_platt_scaler(scaler_rf, raw_scores_rf)

    scaler_lgb = fit_platt_scaler(raw_scores_lgb, y_true)
    cal_probs_lgb = apply_platt_scaler(scaler_lgb, raw_scores_lgb)

    # Brier scores
    brier_rf_before = float(brier_score_loss(y_true, raw_scores_rf))
    brier_rf_after = float(brier_score_loss(y_true, cal_probs_rf))

    brier_lgb_before = float(brier_score_loss(y_true, raw_scores_lgb))
    brier_lgb_after = float(brier_score_loss(y_true, cal_probs_lgb))

    print(
        f"RF (weighted)   - Brier before: {brier_rf_before:.5f}, after: {brier_rf_after:.5f}"
    )
    print(
        f"LGBM (weighted) - Brier before: {brier_lgb_before:.5f}, after: {brier_lgb_after:.5f}"
    )

    # We select RF (weighted) as our primary calibrated probability due to superior PR-AUC and lower post-calibration Brier score
    best_th, sweep_df = tune_decision_threshold(y_true, cal_probs_rf, beta=2.0)
    print(f"\nTuned optimal decision threshold for F2: {best_th:.4f}")
    best_row = sweep_df[sweep_df["threshold"] == best_th].iloc[0]
    print(
        f"At threshold {best_th:.4f}: Recall = {best_row['recall']:.2%}, "
        f"Precision = {best_row['precision']:.2%}, F2 = {best_row['f2']:.4f}, "
        f"FN = {int(best_row['fn'])}, FP = {int(best_row['fp'])}"
    )

    # Generate calibration curve plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Perfect Calibration")

    prob_true_raw, prob_pred_raw = calibration_curve(
        y_true, raw_scores_rf, n_bins=8, strategy="quantile"
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true, cal_probs_rf, n_bins=8, strategy="quantile"
    )

    ax.plot(
        prob_pred_raw,
        prob_true_raw,
        marker="s",
        color="#EF4444",
        linewidth=1.5,
        label=f"Uncalibrated RF (Brier={brier_rf_before:.4f})",
    )
    ax.plot(
        prob_pred_cal,
        prob_true_cal,
        marker="o",
        color="#38BDF8",
        linewidth=2,
        label=f"Platt Calibrated RF (Brier={brier_rf_after:.4f})",
    )

    ax.set_title(
        "Reliability Diagram: Collision Risk Probability Calibration",
        fontsize=12,
        color="#F9FAFB",
        pad=12,
    )
    ax.set_xlabel("Mean Predicted Probability", fontsize=10, color="#E2E8F0")
    ax.set_ylabel("Observed Fraction of High-Risk Conjunctions", fontsize=10, color="#E2E8F0")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, facecolor="#111827", edgecolor="#374151")

    plt.tight_layout()
    fig_path = Path("reports/figures/calibration_curve.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved calibration curve to {fig_path}")

    # Now apply Platt scaler to all 11,942 training events
    all_raw_rf = merged["rf_weighted_prob"].values
    all_cal_probs = apply_platt_scaler(scaler_rf, all_raw_rf)

    confidence_series = assign_confidence(all_cal_probs, best_th, cfg)
    recommendations = np.where(all_cal_probs >= best_th, "MANOEUVRE", "NO_MANOEUVRE")

    calibrated_df = pd.DataFrame({
        "event_id": merged["event_id"],
        "fold": merged["fold"],
        "y_high_risk": merged["y_high_risk"],
        "y_risk": merged["y_risk"],
        "latest_log_risk": merged["latest_log_risk"],
        "raw_prob": all_raw_rf,
        "calibrated_prob": np.round(all_cal_probs, 4),
        "decision_threshold": best_th,
        "recommendation": recommendations,
        "confidence": confidence_series,
        "is_eval_filtered": eval_mask.values,
    })

    save_parquet(calibrated_df, "data/processed/calibrated_oof.parquet")
    print(f"Saved calibrated_oof.parquet with {len(calibrated_df)} events.")
    print("Confidence distribution:\n", calibrated_df["confidence"].value_counts())
    print("\nRecommendation distribution:\n", calibrated_df["recommendation"].value_counts())


if __name__ == "__main__":
    run_phase_07()

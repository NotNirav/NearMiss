"""Evaluation metrics, official-style score, and threshold-sweep plots.

Rules:
- Section 12 Rule 4 (No magic numbers): read high_risk_log10 and eval_filter
  settings from config.yaml.
- Section 12 Rule 6 (Metrics): never report accuracy as a headline metric.
  Always report FN (missed dangerous events), FP (unnecessary manoeuvres),
  recall, precision, F2, PR-AUC, high-risk MSE, and official-style score on
  eval_filter events (plus unfiltered as a secondary table).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.cutoff import apply_eval_filter


def compute_regression_metrics(
    y_true_risk: np.ndarray | pd.Series,
    y_pred_risk: np.ndarray | pd.Series,
    high_risk_threshold: float,
) -> dict[str, float | int]:
    """Compute overall log-risk MSE and high-risk-only MSE (y_true >= threshold).

    Inputs:
        y_true_risk: True final log10 risk.
        y_pred_risk: Predicted final log10 risk.
        high_risk_threshold: Log10 threshold defining high-risk events (e.g. -6.0).
    Outputs:
        Dict with 'mse_overall', 'mse_high_risk', 'n_high_risk'.
    Rules:
        Section 12 Rule 6.
    """
    y_true = np.asarray(y_true_risk, dtype=float)
    y_pred = np.asarray(y_pred_risk, dtype=float)

    mse_overall = float(np.mean((y_true - y_pred) ** 2))

    high_risk_mask = y_true >= high_risk_threshold
    n_high = int(high_risk_mask.sum())
    if n_high > 0:
        mse_high = float(np.mean((y_true[high_risk_mask] - y_pred[high_risk_mask]) ** 2))
    else:
        mse_high = 0.0

    return {
        "mse_overall": mse_overall,
        "mse_high_risk": mse_high,
        "n_high_risk": n_high,
    }


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred_binary: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series | None = None,
) -> dict[str, float | int]:
    """Compute TP, FP, TN, FN, recall, precision, F2, PR-AUC, and Brier score.

    Inputs:
        y_true: Binary ground truth (0/1).
        y_pred_binary: Binary predictions (0/1).
        y_score: Optional continuous probability/risk score for PR-AUC & Brier.
    Outputs:
        Dict with tp, fp, tn, fn, recall, precision, f2, pr_auc, brier.
    Rules:
        Section 12 Rule 6 (never report accuracy as headline).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred_binary, dtype=int)

    tp = int(((y_t == 1) & (y_p == 1)).sum())
    fp = int(((y_t == 0) & (y_p == 1)).sum())
    tn = int(((y_t == 0) & (y_p == 0)).sum())
    fn = int(((y_t == 1) & (y_p == 0)).sum())

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    denom = 4.0 * precision + recall
    f2 = float((5.0 * precision * recall) / denom) if denom > 0 else 0.0

    pr_auc = 0.0
    brier = 0.0
    if y_score is not None:
        y_s = np.asarray(y_score, dtype=float)
        if len(np.unique(y_t)) > 1:
            try:
                pr_auc = float(average_precision_score(y_t, y_s))
            except Exception:
                pr_auc = 0.0
            # If scores are probabilities in [0, 1], compute Brier score
            if np.all(y_s >= 0.0) and np.all(y_s <= 1.0):
                brier = float(brier_score_loss(y_t, y_s))

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f2": f2,
        "pr_auc": pr_auc,
        "brier": brier,
    }


def compute_official_style_score(
    mse_high_risk: float,
    f2: float,
) -> float:
    """Compute Kelvins official-style score (high-risk MSE / F2; lower is better).

    Returns inf if f2 == 0.
    """
    if f2 <= 0.0 or np.isnan(f2):
        return float("inf")
    return float(mse_high_risk / f2)


def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    thresholds: np.ndarray | list[float],
) -> pd.DataFrame:
    """Sweep decision thresholds and record FN, FP, recall, precision, and F2."""
    y_t = np.asarray(y_true, dtype=int)
    sc = np.asarray(scores, dtype=float)

    rows = []
    for th in thresholds:
        pred_bin = (sc >= th).astype(int)
        m = compute_classification_metrics(y_t, pred_bin, sc)
        rows.append({
            "threshold": float(th),
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "recall": m["recall"],
            "precision": m["precision"],
            "f2": m["f2"],
        })

    return pd.DataFrame(rows)


def evaluate_predictions(
    targets_df: pd.DataFrame,
    pred_risk: np.ndarray | pd.Series,
    pred_binary: np.ndarray | pd.Series,
    pred_score: np.ndarray | pd.Series | None,
    cfg: dict[str, Any],
    eval_mask: pd.Series | None = None,
) -> dict[str, dict[str, float | int]]:
    """Run full regression + classification evaluation on filtered and unfiltered sets.

    Outputs:
        Dict with keys 'filtered' and 'unfiltered', each holding combined metrics.
    Rules:
        Section 12 Rule 4, Rule 6.
    """
    if eval_mask is None:
        eval_mask = apply_eval_filter(targets_df, cfg)

    high_risk_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))

    results: dict[str, dict[str, float | int]] = {}
    for subset_name, mask in [("filtered", eval_mask), ("unfiltered", pd.Series(True, index=targets_df.index))]:
        sub_targets = targets_df[mask].reset_index(drop=True)
        sub_pred_risk = np.asarray(pred_risk)[mask]
        sub_pred_bin = np.asarray(pred_binary)[mask]
        sub_score = np.asarray(pred_score)[mask] if pred_score is not None else None

        reg_m = compute_regression_metrics(
            sub_targets["y_risk"], sub_pred_risk, high_risk_thresh
        )
        cls_m = compute_classification_metrics(
            sub_targets["y_high_risk"], sub_pred_bin, sub_score
        )
        score = compute_official_style_score(reg_m["mse_high_risk"], cls_m["f2"])

        combined = {**reg_m, **cls_m, "official_score": score, "total_events": len(sub_targets)}
        results[subset_name] = combined

    return results


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Missed Dangerous Events (FN) vs Unnecessary Manoeuvres (FP)",
) -> Path:
    """Save FN-vs-FP threshold sweep plot to out_path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        sweep_df["fp"],
        sweep_df["fn"],
        color="#38BDF8",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Threshold Sweep",
    )

    # Highlight threshold points
    for _, row in sweep_df.iloc[:: max(1, len(sweep_df) // 8)].iterrows():
        ax.annotate(
            f"th={row['threshold']:.2f}",
            (row["fp"], row["fn"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#94A3B8",
        )

    ax.set_title(title, fontsize=12, color="#F9FAFB", pad=12)
    ax.set_xlabel("Unnecessary Manoeuvres (False Positives)", fontsize=10, color="#E2E8F0")
    ax.set_ylabel("Missed Dangerous Events (False Negatives)", fontsize=10, color="#E2E8F0")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, facecolor="#111827", edgecolor="#374151")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out.resolve()

"""Probability calibration (Platt scaling), threshold tuning, and confidence bands.

Rules:
- Section 12 Rule 3 (Test set quarantine): fit calibrator and tune decision
  threshold on out-of-fold predictions ONLY; never on test data.
- Section 12 Rule 4 (No magic numbers): read confidence margins from config.yaml.
- Section 12 Rule 5 (Fit on training data only): do not use CalibratedClassifierCV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_platt_scaler(
    oof_scores: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
) -> LogisticRegression:
    """Fit Platt scaling (1D logistic regression) on out-of-fold model scores.

    Inputs:
        oof_scores: Uncalibrated OOF probabilities or log-odds scores.
        y_true: Binary ground truth labels (0/1).
    Outputs:
        Fitted LogisticRegression calibrator.
    Rules:
        Section 12 Rule 3, Rule 5 (do not use CalibratedClassifierCV).
    """
    scores = np.asarray(oof_scores, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(scores, y)
    return calibrator


def apply_platt_scaler(
    calibrator: LogisticRegression,
    scores: np.ndarray | pd.Series,
) -> np.ndarray:
    """Transform raw model scores into calibrated probabilities in [0, 1]."""
    sc = np.asarray(scores, dtype=float).reshape(-1, 1)
    probs = calibrator.predict_proba(sc)[:, 1]
    return np.clip(probs, 0.0, 1.0)


def tune_decision_threshold(
    y_true: np.ndarray | pd.Series,
    probs: np.ndarray | pd.Series,
    beta: float = 2.0,
) -> tuple[float, pd.DataFrame]:
    """Select probability threshold maximizing F_beta (default F2) on OOF data.

    Inputs:
        y_true: Binary ground truth labels.
        probs: Calibrated OOF probabilities.
        beta: F-beta weight (2.0 for F2).
    Outputs:
        best_threshold: Optimal probability threshold.
        sweep_df: DataFrame of thresholds with recall, precision, f2, fn, fp.
    Rules:
        Section 12 Rule 3 (OOF only).
    """
    y_t = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    best_th = 0.35
    best_fbeta = -1.0
    best_fn = 999999

    b2 = beta ** 2
    for th in thresholds:
        pred = (p >= th).astype(int)
        tp = int(((y_t == 1) & (pred == 1)).sum())
        fp = int(((y_t == 0) & (pred == 1)).sum())
        fn = int(((y_t == 1) & (pred == 0)).sum())
        tn = int(((y_t == 0) & (pred == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        denom = (b2 * prec) + recall
        f_beta = float(((1.0 + b2) * prec * recall) / denom) if denom > 0 else 0.0

        rows.append({
            "threshold": float(th),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": recall,
            "precision": prec,
            "f2": f_beta,
        })

        # Maximize F_beta; break ties with lower false negatives (missed collisions)
        if f_beta > best_fbeta or (abs(f_beta - best_fbeta) < 1e-6 and fn < best_fn):
            best_fbeta = f_beta
            best_fn = fn
            best_th = float(th)

    sweep_df = pd.DataFrame(rows)
    return best_th, sweep_df


def assign_confidence(
    probs: np.ndarray | pd.Series,
    decision_prob: float,
    cfg: dict[str, Any],
) -> pd.Series:
    """Assign 'HIGH', 'MEDIUM', or 'LOW' confidence from |p - decision_prob| margins.

    Inputs:
        probs: Calibrated probabilities.
        decision_prob: Active decision threshold.
        cfg: Config dict with thresholds.confidence_margin_high / medium.
    Outputs:
        Series of strings ('HIGH', 'MEDIUM', 'LOW').
    Rules:
        Section 12 Rule 4 (No magic numbers).
    """
    thresholds = cfg.get("thresholds", {})
    margin_high = float(thresholds.get("confidence_margin_high", 0.25))
    margin_med = float(thresholds.get("confidence_margin_medium", 0.10))

    p = np.asarray(probs, dtype=float)
    margin = np.abs(p - decision_prob)

    confidence = np.full(len(p), "LOW", dtype=object)
    confidence[margin >= margin_med] = "MEDIUM"
    confidence[margin >= margin_high] = "HIGH"

    idx = probs.index if isinstance(probs, pd.Series) else None
    return pd.Series(confidence, index=idx, name="confidence")

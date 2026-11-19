"""Tests for probability calibration, threshold tuning, and confidence bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibrate import (
    apply_platt_scaler,
    assign_confidence,
    fit_platt_scaler,
    tune_decision_threshold,
)
from src.load import load_config


def test_platt_scaling() -> None:
    rng = np.random.default_rng(42)
    # Raw uncalibrated scores with poor calibration
    raw_scores = rng.uniform(0.0, 1.0, size=200)
    y_true = (raw_scores + rng.normal(0, 0.2, size=200) > 0.6).astype(int)

    scaler = fit_platt_scaler(raw_scores, y_true)
    cal_probs = apply_platt_scaler(scaler, raw_scores)

    assert len(cal_probs) == 200
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()


def test_threshold_tuning() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.05, 0.01])

    best_th, sweep_df = tune_decision_threshold(y_true, probs, beta=2.0)
    assert 0.3 < best_th <= 0.7
    # At this optimal threshold, recall should be 1.0 (all 3 caught) and precision 1.0
    best_row = sweep_df[sweep_df["threshold"] == best_th].iloc[0]
    assert best_row["fn"] == 0


def test_confidence_assignment() -> None:
    cfg = load_config()
    decision_th = 0.35
    # High margins: >= 0.25 -> p <= 0.10 or p >= 0.60
    # Med margins: 0.10 to 0.25 -> 0.10 < p <= 0.25 or 0.45 <= p < 0.60
    # Low margins: < 0.10 -> 0.25 < p < 0.45
    probs = pd.Series([0.05, 0.20, 0.36, 0.50, 0.80])
    conf = assign_confidence(probs, decision_th, cfg)

    assert list(conf) == ["HIGH", "MEDIUM", "LOW", "MEDIUM", "HIGH"]

"""Tests for evaluation metrics and Baseline A."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compute_classification_metrics,
    compute_official_style_score,
    compute_regression_metrics,
    threshold_sweep,
)
from src.models import predict_baseline_a


def test_classification_metrics_computation() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])

    m = compute_classification_metrics(y_true, y_pred)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["tn"] == 2
    assert m["fn"] == 1
    assert m["precision"] == 1.0
    assert round(m["recall"], 4) == round(2 / 3, 4)
    expected_f2 = (5.0 * 1.0 * (2 / 3)) / (4.0 * 1.0 + (2 / 3))
    assert round(m["f2"], 4) == round(expected_f2, 4)


def test_regression_and_official_score() -> None:
    y_true_risk = np.array([-20.0, -5.0, -4.0, -30.0])
    y_pred_risk = np.array([-20.0, -6.0, -5.0, -30.0])
    high_thresh = -6.0

    reg = compute_regression_metrics(y_true_risk, y_pred_risk, high_thresh)
    assert reg["n_high_risk"] == 2
    # High risk errors: (-5 - -6)^2 = 1, (-4 - -5)^2 = 1 -> MSE = 1.0
    assert reg["mse_high_risk"] == 1.0

    score = compute_official_style_score(reg["mse_high_risk"], f2=0.5)
    assert score == 2.0

    # f2 == 0 must return inf
    assert compute_official_style_score(1.0, f2=0.0) == float("inf")


def test_baseline_a_persistence() -> None:
    vis_cdms = pd.DataFrame({
        "event_id": [1, 1, 2, 2],
        "time_to_tca": [4.0, 2.5, 3.5, 2.1],
        "risk": [-10.0, -5.5, -20.0, -15.0],
    })
    preds = predict_baseline_a(vis_cdms, high_risk_threshold=-6.0)
    assert len(preds) == 2
    # Event 1: latest visible risk at tca=2.5 is -5.5 >= -6.0 -> MANOEUVRE
    ev1 = preds[preds["event_id"] == 1].iloc[0]
    assert ev1["pred_final_log_risk"] == -5.5
    assert ev1["baseline_recommendation"] == "MANOEUVRE"

    # Event 2: latest visible risk at tca=2.1 is -15.0 < -6.0 -> NO_MANOEUVRE
    ev2 = preds[preds["event_id"] == 2].iloc[0]
    assert ev2["pred_final_log_risk"] == -15.0
    assert ev2["baseline_recommendation"] == "NO_MANOEUVRE"

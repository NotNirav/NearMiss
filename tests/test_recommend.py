"""Tests for production recommendation export and dashboard parquet files."""

from __future__ import annotations

import pandas as pd
import pytest

from src.load import load_parquet
from src.recommend import CDM_HISTORY_COLUMNS, PREDICTIONS_COLUMNS


def test_production_dashboard_files() -> None:
    pred_path = "data/processed/predictions.parquet"
    hist_path = "data/processed/cdm_history.parquet"

    preds = load_parquet(pred_path)
    hist = load_parquet(hist_path)

    # 1. Exact Column schema check
    assert list(preds.columns) == PREDICTIONS_COLUMNS
    assert list(hist.columns) == CDM_HISTORY_COLUMNS

    # 2. Row count expectations
    assert len(preds) == 14109
    assert len(hist) > 50000

    # 3. Splits check
    splits = set(preds["split"].unique())
    assert splits == {"train", "val", "test"}

    # 4. Values and non-null checks
    assert not preds["high_risk_prob"].isna().any()
    assert preds["high_risk_prob"].between(0.0, 1.0).all()
    assert set(preds["recommendation"].unique()).issubset({"MANOEUVRE", "NO_MANOEUVRE"})
    assert set(preds["confidence"].unique()).issubset({"HIGH", "MEDIUM", "LOW"})
    assert set(preds["baseline_recommendation"].unique()).issubset({"MANOEUVRE", "NO_MANOEUVRE"})
    assert preds["actual_high_risk"].isin([0, 1]).all()

    # 5. History checks
    assert not hist["time_to_tca"].isna().any()
    assert not hist["log_risk"].isna().any()
    assert hist["visible"].isin([True, False]).all()

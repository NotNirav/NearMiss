from __future__ import annotations

import pandas as pd
import pytest

from app.streamlit_app import _slice_history_for_split, load_app_data
from src.load import load_config, load_parquet


def test_app_data_loading():
    """Verify that load_app_data returns valid config and dataframes."""
    cfg, preds_df, hist_df = load_app_data()
    assert isinstance(cfg, dict)
    assert isinstance(preds_df, pd.DataFrame)
    assert isinstance(hist_df, pd.DataFrame)

    assert len(preds_df) > 0
    assert len(hist_df) > 0

    # Ensure required columns for app rendering exist (Section 6 specifications)
    required_pred_cols = [
        "event_id",
        "split",
        "pred_final_log_risk",
        "high_risk_prob",
        "confidence",
        "recommendation",
        "baseline_recommendation",
        "actual_high_risk",
    ]
    for col in required_pred_cols:
        assert col in preds_df.columns, f"Missing required column in predictions: {col}"

    required_hist_cols = [
        "event_id",
        "time_to_tca",
        "log_risk",
        "miss_distance",
        "visible",
    ]
    for col in required_hist_cols:
        assert col in hist_df.columns, f"Missing required column in cdm_history: {col}"


def test_app_split_filtering():
    """Verify that both val and test splits exist and have non-empty history."""
    _, preds_df, hist_df = load_app_data()

    val_preds = preds_df[preds_df["split"] == "val"]
    test_preds = preds_df[preds_df["split"] == "test"]

    assert len(val_preds) > 0, "Validation split should not be empty"
    assert len(test_preds) > 0, "Test split should not be empty"

    # Verify event_ids match between predictions and history
    sample_val_event = val_preds["event_id"].iloc[0]
    val_hist = hist_df[hist_df["event_id"] == sample_val_event]
    assert len(val_hist) > 0, f"History should exist for event {sample_val_event}"


def test_history_split_isolation():
    """Verify val and test CDM histories do not merge on colliding event_ids."""
    _, preds_df, hist_df = load_app_data()

    val_preds = preds_df[preds_df["split"] == "val"]
    test_preds = preds_df[preds_df["split"] == "test"]
    shared_ids = set(val_preds["event_id"]).intersection(set(test_preds["event_id"]))
    if not shared_ids:
        return

    eid = sorted(shared_ids)[0]
    val_hist = _slice_history_for_split(hist_df, "val")
    test_hist = _slice_history_for_split(hist_df, "test")

    expected_val_vis = int(val_preds[val_preds["event_id"] == eid]["cdm_count_pre_cutoff"].iloc[0])
    expected_test_vis = int(test_preds[test_preds["event_id"] == eid]["cdm_count_pre_cutoff"].iloc[0])

    actual_val_vis = int(val_hist[(val_hist["event_id"] == eid) & (val_hist["visible"])].shape[0])
    actual_test_vis = int(test_hist[(test_hist["event_id"] == eid) & (test_hist["visible"])].shape[0])

    assert actual_val_vis == expected_val_vis
    assert actual_test_vis == expected_test_vis

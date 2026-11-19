"""Leakage and group-split tests (implemented in Phase 2 and Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cutoff import apply_cutoff_and_targets, apply_eval_filter
from src.load import load_config
from src.split import assert_zero_overlap, create_splits


def test_no_cutoff_leakage() -> None:
    """Changing a CDM at time_to_tca < cutoff must leave visible CDMs 100% unchanged."""
    cfg = load_config()
    cutoff_days = float(cfg["data"]["cutoff_days"])
    high_risk_thresh = float(cfg["thresholds"]["high_risk_log10"])

    # Create synthetic event with CDMs across pre-cutoff and post-cutoff periods
    cdms = pd.DataFrame({
        "event_id": [101, 101, 101, 101, 101],
        "time_to_tca": [5.0, 3.2, 2.1, 1.8, 0.4],
        "risk": [-20.0, -18.0, -15.0, -10.0, -4.0],
        "miss_distance": [5000.0, 4000.0, 3500.0, 1000.0, 200.0],
        "relative_speed": [12000.0, 12000.0, 12000.0, 12000.0, 12000.0],
    })

    vis_original, targets_orig, stats_orig = apply_cutoff_and_targets(
        cdms, cutoff_days, high_risk_thresh
    )

    # All visible CDMs must strictly have time_to_tca >= cutoff_days
    assert (vis_original["time_to_tca"] >= cutoff_days).all()
    assert len(vis_original) == 3

    # Mutate a CDM in the blind zone (time_to_tca < cutoff_days, e.g. at 1.8 or 0.4)
    cdms_mutated_blind_zone = cdms.copy()
    cdms_mutated_blind_zone.loc[cdms_mutated_blind_zone["time_to_tca"] == 1.8, "risk"] = 0.0
    cdms_mutated_blind_zone.loc[cdms_mutated_blind_zone["time_to_tca"] == 1.8, "miss_distance"] = 10.0

    vis_mutated, targets_mut, stats_mut = apply_cutoff_and_targets(
        cdms_mutated_blind_zone, cutoff_days, high_risk_thresh
    )

    # Visible inputs MUST be completely identical despite mutation in blind zone
    pd.testing.assert_frame_equal(vis_original, vis_mutated)

    # Mutate a CDM in the visible zone (time_to_tca >= cutoff_days, e.g. at 3.2)
    cdms_mutated_vis = cdms.copy()
    cdms_mutated_vis.loc[cdms_mutated_vis["time_to_tca"] == 3.2, "risk"] = -5.0
    vis_mut_vis, _, _ = apply_cutoff_and_targets(
        cdms_mutated_vis, cutoff_days, high_risk_thresh
    )

    # Visible inputs MUST change
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(vis_original, vis_mut_vis)


def test_eval_filter_logic() -> None:
    """Verify eval_filter mask correctly identifies events meeting Kelvins criteria."""
    cfg = load_config()
    targets_sample = pd.DataFrame({
        "event_id": [1, 2, 3, 4],
        "pre_cutoff_cdm_count": [2, 0, 3, 1],
        "final_time_to_tca": [0.5, 0.4, 1.8, 0.99],
        "y_risk": [-5.0, -10.0, -6.5, -4.0],
    })

    mask = apply_eval_filter(targets_sample, cfg)
    # Event 1: pre=2 >= 1, final=0.5 <= 1.0 -> True
    # Event 2: pre=0 < 1 -> False
    # Event 3: final=1.8 > 1.0 -> False
    # Event 4: pre=1 >= 1, final=0.99 <= 1.0 -> True
    assert list(mask) == [True, False, False, True]


def test_zero_event_id_overlap() -> None:
    """No event_id may appear in more than one split or fold, with balanced stratification."""
    cfg = load_config()
    rng = np.random.default_rng(42)

    # Create synthetic dataset with 200 events and 5% high risk
    n_events = 200
    y_high = (rng.uniform(0, 1, size=n_events) < 0.05).astype(int)
    # Ensure at least some positives
    y_high[0] = 1
    y_high[10] = 1

    targets_dummy = pd.DataFrame({
        "event_id": np.arange(1000, 1000 + n_events),
        "y_high_risk": y_high,
    })

    folds_df = create_splits(targets_dummy, cfg)

    # Assert zero overlap passes
    assert_zero_overlap(folds_df)

    # Check that each event appears exactly once
    assert len(folds_df) == n_events
    assert folds_df["event_id"].nunique() == n_events

    # Artificially inject an overlap between train and test
    corrupted_df = folds_df.copy()
    corrupted_df.loc[corrupted_df["split"] == "test", "event_id"] = corrupted_df[
        corrupted_df["split"] == "train"
    ]["event_id"].iloc[0]

    with pytest.raises(AssertionError):
        assert_zero_overlap(corrupted_df)

    # Artificially inject an overlap between fold 0 and fold 1
    corrupted_folds = folds_df.copy()
    f1_idx = corrupted_folds[corrupted_folds["fold"] == 1].index[0]
    corrupted_folds.loc[f1_idx, "fold"] = 0
    # Add a duplicate row with same event_id in fold 1
    dup_row = corrupted_folds.loc[[f1_idx]].copy()
    dup_row["fold"] = 1
    corrupted_folds = pd.concat([corrupted_folds, dup_row], ignore_index=True)

    with pytest.raises(AssertionError):
        assert_zero_overlap(corrupted_folds)

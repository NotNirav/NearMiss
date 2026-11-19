"""Tests for feature engineering and cutoff leakage invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cutoff import apply_cutoff_and_targets
from src.features import build_all_features, build_base_features, build_physics_features
from src.load import load_config


@pytest.fixture
def sample_visible_cdms() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 15
    eids = [101] * 5 + [102] * 4 + [103] * 6
    t_tca = [
        6.5, 5.0, 4.0, 3.0, 2.1,
        5.2, 4.1, 3.2, 2.05,
        6.0, 5.1, 4.5, 3.8, 2.9, 2.01,
    ]
    risks = rng.uniform(-25.0, -5.0, size=n_rows)
    miss_dists = rng.uniform(500.0, 15000.0, size=n_rows)
    rel_speed = [14000.0] * n_rows

    df = pd.DataFrame({
        "event_id": eids,
        "time_to_tca": t_tca,
        "risk": risks,
        "miss_distance": miss_dists,
        "relative_speed": rel_speed,
        "relative_position_r": rng.uniform(-2000, 2000, size=n_rows),
        "relative_position_t": rng.uniform(-5000, 5000, size=n_rows),
        "relative_position_n": rng.uniform(-1000, 1000, size=n_rows),
        "relative_velocity_r": rng.uniform(-10, 10, size=n_rows),
        "relative_velocity_t": rng.uniform(-14000, 14000, size=n_rows),
        "relative_velocity_n": rng.uniform(-20, 20, size=n_rows),
        "t_sigma_r": rng.uniform(10, 100, size=n_rows),
        "c_sigma_r": rng.uniform(20, 150, size=n_rows),
        "t_sigma_t": rng.uniform(50, 400, size=n_rows),
        "c_sigma_t": rng.uniform(80, 500, size=n_rows),
        "t_sigma_n": rng.uniform(15, 120, size=n_rows),
        "c_sigma_n": rng.uniform(25, 180, size=n_rows),
        "t_position_covariance_det": rng.uniform(1e6, 1e9, size=n_rows),
        "c_position_covariance_det": rng.uniform(1e6, 1e9, size=n_rows),
        "mission_id": [1] * 5 + [2] * 4 + [3] * 6,
        "c_object_type": ["DEBRIS"] * 5 + ["PAYLOAD"] * 4 + ["DEBRIS"] * 6,
        "F10": [90.0] * n_rows,
        "F3M": [85.0] * n_rows,
        "SSN": [40.0] * n_rows,
        "AP": [12.0] * n_rows,
        "mahalanobis_distance": rng.uniform(1.0, 10.0, size=n_rows),
        "max_risk_estimate": risks + 0.5,
        "max_risk_scaling": [1.0] * n_rows,
    })
    return df


def test_build_features_schema_and_values(sample_visible_cdms: pd.DataFrame) -> None:
    feats = build_all_features(sample_visible_cdms, include_physics=True)

    # 3 unique events
    assert len(feats) == 3
    assert feats["event_id"].nunique() == 3

    # Must contain base and physics features
    assert "latest_log_risk" in feats.columns
    assert "risk_slope_per_day" in feats.columns
    assert "latest_comb_sigma_3d" in feats.columns
    assert "latest_norm_miss_distance" in feats.columns
    assert "volume_contraction_ratio" in feats.columns

    # No NaN or Infinite values
    assert not feats.isna().any().any()
    assert not np.isinf(feats.select_dtypes(include=np.number).values).any()

    # Verify column count >= 40
    assert feats.shape[1] >= 40


def test_feature_leakage_invariance() -> None:
    """Mutating post-cutoff CDMs must leave all extracted features exactly identical."""
    cfg = load_config()
    cutoff_days = float(cfg["data"]["cutoff_days"])
    high_thresh = float(cfg["thresholds"]["high_risk_log10"])

    # Synthetic event with 3 pre-cutoff CDMs and 2 post-cutoff CDMs
    cdms = pd.DataFrame({
        "event_id": [201] * 5,
        "time_to_tca": [4.5, 3.2, 2.1, 1.5, 0.2],
        "risk": [-22.0, -18.0, -14.0, -8.0, -3.0],
        "miss_distance": [8000.0, 6000.0, 4500.0, 1200.0, 150.0],
        "relative_speed": [12500.0] * 5,
        "relative_position_r": [100.0] * 5,
        "relative_position_t": [500.0] * 5,
        "relative_position_n": [50.0] * 5,
        "relative_velocity_r": [2.0] * 5,
        "relative_velocity_t": [12500.0] * 5,
        "relative_velocity_n": [-1.0] * 5,
        "t_sigma_r": [20.0] * 5, "c_sigma_r": [30.0] * 5,
        "t_sigma_t": [100.0] * 5, "c_sigma_t": [150.0] * 5,
        "t_sigma_n": [25.0] * 5, "c_sigma_n": [35.0] * 5,
        "t_position_covariance_det": [1e6] * 5, "c_position_covariance_det": [1e6] * 5,
        "mission_id": [1] * 5, "c_object_type": ["DEBRIS"] * 5,
        "F10": [80.0] * 5, "F3M": [80.0] * 5, "SSN": [30.0] * 5, "AP": [10.0] * 5,
        "mahalanobis_distance": [3.0] * 5,
        "max_risk_estimate": [-21.0, -17.0, -13.0, -7.0, -2.5],
        "max_risk_scaling": [1.0] * 5,
    })

    # Visible slice 1
    vis1, _, _ = apply_cutoff_and_targets(cdms, cutoff_days, high_thresh)
    feats1 = build_all_features(vis1)

    # Mutate post-cutoff CDMs (time_to_tca < 2.0)
    cdms_mutated = cdms.copy()
    cdms_mutated.loc[cdms_mutated["time_to_tca"] < cutoff_days, "risk"] = 0.0
    cdms_mutated.loc[cdms_mutated["time_to_tca"] < cutoff_days, "miss_distance"] = 1.0

    vis2, _, _ = apply_cutoff_and_targets(cdms_mutated, cutoff_days, high_thresh)
    feats2 = build_all_features(vis2)

    pd.testing.assert_frame_equal(feats1, feats2)

"""Event-level feature engineering from visible CDMs only.

Rules:
- Section 12 Rule 1 (Cutoff isolation): takes only visible_cdms from cutoff.py.
- Section 12 Rule 4 (No magic numbers): no hardcoded cutoff or threshold constants.
- Section 12 Rule 5 (Fit on training data only): no cross-event scaling/imputation here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_log1p(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    # Clip negative values to zero before log1p
    arr = np.where(np.isnan(arr) | (arr < 0.0), 0.0, arr)
    return np.log1p(arr)


def build_base_features(visible_cdms: pd.DataFrame) -> pd.DataFrame:
    """Build core latest-state, trajectory, and geometry/uncertainty features per event_id.

    Inputs:
        visible_cdms: Filtered CDM DataFrame (time_to_tca >= cutoff_days) sorted
            by event_id ascending, time_to_tca descending.
    Outputs:
        DataFrame indexed by event_id with latest-state, trajectory, and geometry features.
    Rules:
        Section 12 Rule 1, Rule 5.
    """
    df = visible_cdms.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    ).copy()
    risk_col = "risk" if "risk" in df.columns else "true_risk"

    grouped = df.groupby("event_id")

    # 1. Trajectory aggregates
    agg_df = grouped.agg(
        cdm_count=(risk_col, "count"),
        risk_min=(risk_col, "min"),
        risk_max=(risk_col, "max"),
        risk_mean=(risk_col, "mean"),
        risk_std=(risk_col, "std"),
        miss_dist_min=("miss_distance", "min"),
        miss_dist_max=("miss_distance", "max"),
        miss_dist_mean=("miss_distance", "mean"),
        miss_dist_std=("miss_distance", "std"),
    )
    agg_df["risk_std"] = agg_df["risk_std"].fillna(0.0)
    agg_df["miss_dist_std"] = agg_df["miss_dist_std"].fillna(0.0)

    # 2. First visible CDM (largest time_to_tca)
    first_cdm = grouped.first()
    agg_df["risk_first"] = first_cdm[risk_col]
    agg_df["miss_dist_first"] = first_cdm["miss_distance"]
    agg_df["time_to_tca_first"] = first_cdm["time_to_tca"]

    # 3. Latest visible CDM (smallest time_to_tca)
    last_cdm = grouped.last()
    agg_df["latest_log_risk"] = last_cdm[risk_col]
    agg_df["latest_time_to_tca"] = last_cdm["time_to_tca"]
    agg_df["latest_miss_distance"] = last_cdm["miss_distance"]
    agg_df["latest_log_miss_distance"] = _safe_log1p(last_cdm["miss_distance"])
    agg_df["latest_relative_speed"] = last_cdm["relative_speed"]

    # Trajectory changes
    agg_df["risk_total_change"] = agg_df["latest_log_risk"] - agg_df["risk_first"]
    agg_df["miss_dist_total_change"] = agg_df["latest_miss_distance"] - agg_df["miss_dist_first"]

    # Step changes (difference between latest and second-to-latest CDM)
    # Using shift inside group
    df["prev_risk"] = grouped[risk_col].shift(1)
    df["prev_miss"] = grouped["miss_distance"].shift(1)
    last_step = df.groupby("event_id").last()
    agg_df["risk_last_step_change"] = (last_step[risk_col] - last_step["prev_risk"]).fillna(0.0)
    agg_df["miss_dist_last_step_change"] = (last_step["miss_distance"] - last_step["prev_miss"]).fillna(0.0)

    # Risk slope over time_to_tca (Cov(t, r) / Var(t))
    # Note: t is decreasing, so slope with respect to elapsed time
    df["t_diff"] = df["time_to_tca"] - grouped["time_to_tca"].transform("mean")
    df["r_diff"] = df[risk_col] - grouped[risk_col].transform("mean")
    df["cov_tr"] = df["t_diff"] * df["r_diff"]
    df["var_t"] = df["t_diff"] ** 2

    slope_df = df.groupby("event_id")[["cov_tr", "var_t"]].sum()
    var_zero = slope_df["var_t"] <= 1e-12
    slope_series = slope_df["cov_tr"] / np.where(var_zero, 1.0, slope_df["var_t"])
    agg_df["risk_slope_per_day"] = np.where(var_zero, 0.0, slope_series)

    # Acceleration over last 3 CDMs
    # For events with >= 3 CDMs: finite second difference
    df["cdm_rev_order"] = grouped.cumcount(ascending=False)
    # CDMs 0, 1, 2 from the end
    last_3 = df[df["cdm_rev_order"] < 3].sort_values(["event_id", "cdm_rev_order"])
    piv = last_3.pivot(index="event_id", columns="cdm_rev_order", values=risk_col)
    # If 0, 1, 2 exist: (r0 - r1) - (r1 - r2) = r0 - 2*r1 + r2
    if 0 in piv.columns and 1 in piv.columns and 2 in piv.columns:
        accel = piv[0] - 2 * piv[1] + piv[2]
        agg_df["risk_accel_last_3"] = accel.fillna(0.0)
    else:
        agg_df["risk_accel_last_3"] = 0.0

    # Risk estimates & scaling
    if "max_risk_estimate" in last_cdm.columns:
        agg_df["latest_max_risk_estimate"] = last_cdm["max_risk_estimate"].fillna(agg_df["latest_log_risk"])
    if "max_risk_scaling" in last_cdm.columns:
        agg_df["latest_max_risk_scaling"] = last_cdm["max_risk_scaling"].fillna(1.0)

    # Object info & space weather
    if "mission_id" in last_cdm.columns:
        agg_df["mission_id"] = last_cdm["mission_id"].fillna(-1).astype(int)

    if "c_object_type" in last_cdm.columns:
        # One-hot encode common object types
        obj_types = ["DEBRIS", "PAYLOAD", "ROCKET BODY"]
        for ot in obj_types:
            agg_df[f"is_obj_{ot.lower().replace(' ', '_')}"] = (
                last_cdm["c_object_type"].str.upper() == ot
            ).astype(int)

    for sw in ["F10", "F3M", "SSN", "AP"]:
        if sw in last_cdm.columns:
            agg_df[f"latest_{sw}"] = last_cdm[sw].fillna(0.0)

    # RTN components
    for axis in ["r", "t", "n"]:
        pos_col = f"relative_position_{axis}"
        vel_col = f"relative_velocity_{axis}"
        if pos_col in last_cdm.columns:
            agg_df[f"latest_pos_{axis}"] = last_cdm[pos_col]
        if vel_col in last_cdm.columns:
            agg_df[f"latest_vel_{axis}"] = last_cdm[vel_col]

    # Covariance determinants & sigmas
    for prefix in ["t_", "c_"]:
        det_col = f"{prefix}position_covariance_det"
        if det_col in last_cdm.columns:
            agg_df[f"latest_{det_col}"] = _safe_log1p(last_cdm[det_col])
        for axis in ["r", "t", "n"]:
            sig_col = f"{prefix}sigma_{axis}"
            if sig_col in last_cdm.columns:
                agg_df[f"latest_{sig_col}"] = last_cdm[sig_col].fillna(0.0)
                agg_df[f"latest_log_{sig_col}"] = _safe_log1p(last_cdm[sig_col])

    # Clean infinities and NaN
    agg_df = agg_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return agg_df.reset_index()


def build_physics_features(visible_cdms: pd.DataFrame) -> pd.DataFrame:
    """Build simple physics features (combined sigmas, normalised miss distance,
    uncertainty volume, and volume contraction) per event_id.

    Inputs:
        visible_cdms: Filtered CDM DataFrame (time_to_tca >= cutoff_days).
    Outputs:
        DataFrame indexed by event_id with physics-pack features.
    Rules:
        Section 12 Rule 1, Rule 7 (CREDITS.md attribution for competition ideas).
    """
    df = visible_cdms.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    ).copy()

    # Calculate combined sigma per axis for every row
    # combined_sigma = sqrt(t_sigma^2 + c_sigma^2)
    sig_r2 = df["t_sigma_r"].fillna(0.0) ** 2 + df["c_sigma_r"].fillna(0.0) ** 2
    sig_t2 = df["t_sigma_t"].fillna(0.0) ** 2 + df["c_sigma_t"].fillna(0.0) ** 2
    sig_n2 = df["t_sigma_n"].fillna(0.0) ** 2 + df["c_sigma_n"].fillna(0.0) ** 2

    df["comb_sigma_r"] = np.sqrt(np.maximum(sig_r2, 1e-10))
    df["comb_sigma_t"] = np.sqrt(np.maximum(sig_t2, 1e-10))
    df["comb_sigma_n"] = np.sqrt(np.maximum(sig_n2, 1e-10))

    # 3D combined sigma magnitude
    df["comb_sigma_3d"] = np.sqrt(sig_r2 + sig_t2 + sig_n2)

    # Normalised miss distance: sqrt(sum((rel_pos / comb_sigma)^2))
    norm_sq = (
        (df["relative_position_r"].fillna(0.0) / df["comb_sigma_r"]) ** 2
        + (df["relative_position_t"].fillna(0.0) / df["comb_sigma_t"]) ** 2
        + (df["relative_position_n"].fillna(0.0) / df["comb_sigma_n"]) ** 2
    )
    df["normalised_miss_distance"] = np.sqrt(norm_sq)

    # Uncertainty volume (ellipsoid volume proportional to product of principal semi-axes)
    df["uncertainty_volume"] = df["comb_sigma_r"] * df["comb_sigma_t"] * df["comb_sigma_n"]

    grouped = df.groupby("event_id")
    first_cdm = grouped.first()
    last_cdm = grouped.last()

    physics_df = pd.DataFrame(index=last_cdm.index)
    physics_df["latest_comb_sigma_r"] = last_cdm["comb_sigma_r"]
    physics_df["latest_comb_sigma_t"] = last_cdm["comb_sigma_t"]
    physics_df["latest_comb_sigma_n"] = last_cdm["comb_sigma_n"]
    physics_df["latest_comb_sigma_3d"] = last_cdm["comb_sigma_3d"]

    physics_df["latest_log_comb_sigma_r"] = _safe_log1p(last_cdm["comb_sigma_r"])
    physics_df["latest_log_comb_sigma_t"] = _safe_log1p(last_cdm["comb_sigma_t"])
    physics_df["latest_log_comb_sigma_n"] = _safe_log1p(last_cdm["comb_sigma_n"])
    physics_df["latest_log_comb_sigma_3d"] = _safe_log1p(last_cdm["comb_sigma_3d"])

    physics_df["latest_norm_miss_distance"] = last_cdm["normalised_miss_distance"]
    physics_df["min_norm_miss_distance"] = grouped["normalised_miss_distance"].min()

    # Relative velocity and position magnitudes
    rel_speed_sq = (
        last_cdm["relative_velocity_r"] ** 2
        + last_cdm["relative_velocity_t"] ** 2
        + last_cdm["relative_velocity_n"] ** 2
    )
    rel_pos_sq = (
        last_cdm["relative_position_r"] ** 2
        + last_cdm["relative_position_t"] ** 2
        + last_cdm["relative_position_n"] ** 2
    )
    physics_df["latest_rel_speed_mag"] = np.sqrt(np.maximum(rel_speed_sq, 0.0))
    physics_df["latest_rel_pos_mag"] = np.sqrt(np.maximum(rel_pos_sq, 0.0))
    physics_df["latest_log_rel_pos_mag"] = _safe_log1p(physics_df["latest_rel_pos_mag"])

    # Uncertainty volume and volume contraction
    physics_df["latest_uncertainty_volume"] = last_cdm["uncertainty_volume"]
    physics_df["latest_log_uncertainty_volume"] = _safe_log1p(last_cdm["uncertainty_volume"])
    physics_df["first_log_uncertainty_volume"] = _safe_log1p(first_cdm["uncertainty_volume"])

    first_vol = first_cdm["uncertainty_volume"].fillna(1.0).values
    latest_vol = last_cdm["uncertainty_volume"].fillna(1.0).values
    physics_df["volume_contraction_ratio"] = np.where(
        first_vol > 0, latest_vol / np.maximum(first_vol, 1e-6), 1.0
    )
    physics_df["log_volume_contraction"] = np.log10(
        np.maximum(physics_df["volume_contraction_ratio"], 1e-12)
    )

    if "mahalanobis_distance" in last_cdm.columns:
        physics_df["latest_mahalanobis_distance"] = last_cdm["mahalanobis_distance"].fillna(0.0)

    # Clean infinities and NaN
    physics_df = physics_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return physics_df.reset_index()


def build_all_features(
    visible_cdms: pd.DataFrame,
    include_physics: bool = True,
) -> pd.DataFrame:
    """Combine base features and optional physics features into one event-level table.

    Inputs:
        visible_cdms: Filtered CDM DataFrame (time_to_tca >= cutoff_days).
        include_physics: Whether to join build_physics_features output.
    Outputs:
        Feature matrix DataFrame indexed by event_id.
    Rules:
        Section 12 Rule 1, Rule 5.
    """
    base_feats = build_base_features(visible_cdms)
    if not include_physics:
        return base_feats

    physics_feats = build_physics_features(visible_cdms)
    all_feats = base_feats.merge(physics_feats, on="event_id", how="left")
    all_feats = all_feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return all_feats

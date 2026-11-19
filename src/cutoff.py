"""Leak-proof 2-day cutoff filter and event-level target builder.

Rules:
- Section 12 Rule 1 (Cutoff isolation): all downstream feature, model, and
  evaluation code receives data through this module. Never use CDMs with
  time_to_tca < cutoff_days as inputs.
- Section 12 Rule 4 (No magic numbers): read cutoff_days, thresholds, and
  eval_filter parameters from config.yaml.
- Section 3 (Training set rule & Evaluation filter): drop and log events with
  no pre-cutoff CDM; provide eval_filter helper for validation/test reporting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_cutoff_and_targets(
    cdms_df: pd.DataFrame,
    cutoff_days: float,
    high_risk_threshold: float,
    operational_threshold: float = -4.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Filter CDMs to time_to_tca >= cutoff_days and build event-level targets.

    Inputs:
        cdms_df: Raw CDM DataFrame (from src.load.load_raw_cdms).
        cutoff_days: Minimum time_to_tca (days) allowed for visible input CDMs.
        high_risk_threshold: Primary log10 risk threshold (e.g. -6.0).
        operational_threshold: Secondary log10 risk threshold (e.g. -4.0).
    Outputs:
        visible_cdms: CDMs with time_to_tca >= cutoff_days, sorted by
            event_id ascending and time_to_tca descending.
        targets_df: One row per valid event_id with columns:
            event_id, y_risk, y_high_risk, y_operational_risk,
            final_time_to_tca, pre_cutoff_cdm_count, total_cdm_count.
        stats: Counts dictionary (e.g. total_events, kept_events, dropped_no_pre_cutoff).
    Rules:
        Section 12 Rule 1 (Cutoff isolation), Rule 4 (No magic numbers).
    """
    if "event_id" not in cdms_df.columns or "time_to_tca" not in cdms_df.columns:
        raise ValueError("cdms_df must contain 'event_id' and 'time_to_tca'")

    # Ensure clean sorting by event_id ascending and time_to_tca descending
    sorted_df = cdms_df.sort_values(
        by=["event_id", "time_to_tca"], ascending=[True, False]
    ).copy()

    total_events = int(sorted_df["event_id"].nunique())

    # Build event-level ground truth from final CDM (smallest time_to_tca per event)
    # Since sorted by time_to_tca descending, tail(1) per group is the final CDM
    final_cdms = sorted_df.groupby("event_id", as_index=False).last()

    # Pre-cutoff counts
    pre_cutoff_mask = sorted_df["time_to_tca"] >= cutoff_days
    pre_cutoff_counts = (
        sorted_df[pre_cutoff_mask]
        .groupby("event_id")
        .size()
        .rename("pre_cutoff_cdm_count")
    )
    total_counts = (
        sorted_df.groupby("event_id").size().rename("total_cdm_count")
    )

    targets = pd.DataFrame({"event_id": final_cdms["event_id"]})
    risk_col = "risk" if "risk" in final_cdms.columns else "true_risk"
    targets["y_risk"] = final_cdms[risk_col].values
    targets["y_high_risk"] = (targets["y_risk"] >= high_risk_threshold).astype(int)
    targets["y_operational_risk"] = (
        targets["y_risk"] >= operational_threshold
    ).astype(int)
    targets["final_time_to_tca"] = final_cdms["time_to_tca"].values

    targets = targets.merge(total_counts, on="event_id", how="left")
    targets = targets.merge(pre_cutoff_counts, on="event_id", how="left")
    targets["pre_cutoff_cdm_count"] = targets["pre_cutoff_cdm_count"].fillna(0).astype(int)

    # Filter out events with no visible pre-cutoff CDMs
    valid_mask = targets["pre_cutoff_cdm_count"] > 0
    kept_events_ids = set(targets.loc[valid_mask, "event_id"])
    targets_clean = targets[valid_mask].reset_index(drop=True)

    # Filter visible CDMs strictly to time_to_tca >= cutoff_days for kept events
    visible_cdms = sorted_df[
        pre_cutoff_mask & sorted_df["event_id"].isin(kept_events_ids)
    ].reset_index(drop=True)

    stats = {
        "total_events": total_events,
        "kept_events": len(targets_clean),
        "dropped_no_pre_cutoff": total_events - len(targets_clean),
        "total_cdms": len(sorted_df),
        "visible_cdms": len(visible_cdms),
    }

    return visible_cdms, targets_clean, stats


def apply_eval_filter(
    targets_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    """Return boolean mask for events passing Kelvins evaluation filter.

    Inputs:
        targets_df: Event-level target table from apply_cutoff_and_targets.
        cfg: Loaded config dictionary containing eval_filter.min_pre_cutoff_cdms
            and eval_filter.max_final_time_to_tca.
    Outputs:
        Boolean Series indexed like targets_df (True = included in primary eval).
    Rules:
        Section 12 Rule 4, Rule 6.
    """
    eval_cfg = cfg.get("eval_filter", {})
    min_pre = int(eval_cfg.get("min_pre_cutoff_cdms", 1))
    max_final_tca = float(eval_cfg.get("max_final_time_to_tca", 1.0))

    mask = (
        (targets_df["pre_cutoff_cdm_count"] >= min_pre)
        & (targets_df["final_time_to_tca"] <= max_final_tca)
    )
    return mask


def build_multi_cutoff_dataset(
    cdms_df: pd.DataFrame,
    cutoffs: list[float],
    high_risk_threshold: float,
    operational_threshold: float = -4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optional Phase 6 augmentation: stack visible CDMs and targets across cutoffs.

    Inputs:
        cdms_df: Raw CDM DataFrame.
        cutoffs: List of cutoff_days values (e.g. [2.0, 3.0, 4.0]).
        high_risk_threshold: Primary log10 risk threshold.
        operational_threshold: Secondary log10 risk threshold.
    Outputs:
        visible_cdms_aug: Visible CDMs with column 'cutoff_days'.
        targets_aug: Targets with column 'cutoff_days'.
    Rules:
        Section 12 Rule 1 (Cutoff isolation), Rule 2 (keep event_id in one fold).
    """
    vis_list = []
    tgt_list = []

    for c in cutoffs:
        vis, tgt, _ = apply_cutoff_and_targets(
            cdms_df=cdms_df,
            cutoff_days=c,
            high_risk_threshold=high_risk_threshold,
            operational_threshold=operational_threshold,
        )
        vis = vis.copy()
        tgt = tgt.copy()
        vis["cutoff_days"] = c
        tgt["cutoff_days"] = c
        vis_list.append(vis)
        tgt_list.append(tgt)

    visible_cdms_aug = pd.concat(vis_list, ignore_index=True)
    targets_aug = pd.concat(tgt_list, ignore_index=True)

    return visible_cdms_aug, targets_aug

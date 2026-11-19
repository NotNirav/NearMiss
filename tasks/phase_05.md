# Task: Phase 05 — Feature Engineering (5 to 7 h)

## Relevant Plan Section
> **Phase 5: Feature engineering (5 to 7 h)**
> All features use only visible CDMs, through `cutoff.py`.
> - **Latest state:** `latest_log_risk`, `latest_max_risk_estimate`, `latest_max_risk_scaling`, `latest_miss_distance`, `latest_relative_speed`, `latest_time_to_tca`, `mission_id`, `object_type`, space-weather values.
> - **Trajectory:** CDM count, first, min, max, mean and standard deviation of log risk, total change (latest minus first), last-step change, slope per day (linear fit against `time_to_tca`), and acceleration over the last 3 CDMs.
> - **Geometry and uncertainty:** miss-distance change and slope, the primary and secondary position sigmas, and missing-value flags.
> - **Simple physics pack** (own implementation of ideas common in the competition write-ups):
>   - combined sigma per axis = sqrt(primary sigma squared + secondary sigma squared) for radial, transverse and normal axes;
>   - normalised miss distance = sqrt of the sum over axes of (relative position / combined sigma) squared;
>   - uncertainty volume = product of the three combined sigmas, and contraction = latest volume divided by first volume.
> - Keep risk on the log scale and log-transform heavy-tailed positive columns.
> - **Done when:** `features_base` and `features_physics` tables exist, are documented, and pass the leak test.

## Files to Touch
- `src/features.py` (implement `build_base_features`, `build_physics_features`, `build_all_features`)
- `tests/test_no_leakage.py` (extend `test_no_cutoff_leakage` to assert every extracted feature from `build_all_features` is unchanged when a CDM at `time_to_tca = 1.99` is mutated)
- `reports/assumptions.md` (document exact feature list, ~40–80 features, and any log-transforms/missing flags)
- `CREDITS.md` (note any read competition write-up/talk for physics features)

## Inputs & Outputs
- **Inputs:** `visible_cdms` from `src/cutoff.py` (using verified column names from Phase 1 `reports/assumptions.md`)
- **Outputs:**
  - `data/processed/features_base.parquet`
  - `data/processed/features_physics.parquet`
  - Passing end-to-end leak test in `tests/test_no_leakage.py`
  - `reports/status/phase_5.md`

## Instructions
1. Read `reports/assumptions.md` to use the exact verified column names from Phase 1.
2. Implement `build_base_features(visible_cdms)` in `src/features.py` (latest state, trajectory stats, slope/day, accel over last 3 CDMs, geometry/uncertainty, missing-value flags, log-transforms for heavy-tailed positive columns). Handle 1-CDM and 2-CDM events cleanly without NaN crashes where possible (e.g., slope=0 or NaN + missing flag).
3. Implement `build_physics_features(visible_cdms)` (combined sigma per RTN axis, normalised miss distance, uncertainty volume, volume contraction = latest / first).
4. Implement `build_all_features(visible_cdms, include_physics=True)`.
5. Save `data/processed/features_base.parquet` and `data/processed/features_physics.parquet`.
6. Update `tests/test_no_leakage.py` so `test_no_cutoff_leakage` runs `apply_cutoff_and_targets` + `build_all_features` before and after mutating a `time_to_tca = 1.99` CDM (`risk = 0.0`, etc.) and asserts `pd.testing.assert_frame_equal` holds.

## Acceptance Criteria
- `features_base.parquet` and `features_physics.parquet` exist (combined ~40–80 documented features).
- `test_no_cutoff_leakage` passes on `build_all_features`.
- No cross-event scaling or imputation inside `src/features.py` (Rule 5).
- No `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_5.md` (artifacts, feature counts, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

# Task: Phase 02 — Leak-Proof Cutoff and Target (3 to 4 h) [CRITICAL]

## Relevant Plan Section
> **Phase 2: Leak-proof cutoff and target (3 to 4 h) [critical]**
> - Implement `src/cutoff.py`:
>   1. Group CDMs by `event_id`, sorted by `time_to_tca` descending.
>   2. Build the target from the last CDM (smallest `time_to_tca`): `y_risk` and `y_high_risk`.
>   3. Keep as inputs only CDMs with `time_to_tca` at or above the cutoff. Support a list of cutoffs for optional augmentation.
>   4. Drop and log events with no pre-cutoff CDM.
> - `tests/test_no_leakage.py`: change one CDM at `time_to_tca = 1.99` (for example set its risk to 0.0) and assert that every extracted feature is unchanged.
> - **Done when:** the leak test passes.

## Files to Touch
- `src/cutoff.py` (implement `apply_cutoff_and_targets`, `apply_eval_filter`, `build_multi_cutoff_dataset`)
- `tests/test_no_leakage.py` (implement `test_no_cutoff_leakage`)

## Inputs & Outputs
- **Inputs:** Raw CDM DataFrame via `src.load.load_raw_cdms`, `config.yaml`
- **Outputs:**
  - Working `src/cutoff.py`
  - Passing `test_no_cutoff_leakage` in `tests/test_no_leakage.py`
  - Optional cached `data/processed/targets_train.parquet`
  - `reports/status/phase_2.md`

## Instructions
1. Implement `apply_cutoff_and_targets` in `src/cutoff.py`:
   - Group by `event_id`, sort by `time_to_tca` descending (largest `time_to_tca` first, smallest last).
   - Extract `y_risk` from the last CDM (smallest `time_to_tca`), `y_high_risk = int(y_risk >= high_risk_threshold)`, `y_operational_risk = int(y_risk >= operational_threshold)`, `final_time_to_tca`, `pre_cutoff_cdm_count`, and `total_cdm_count`.
   - Filter input CDMs strictly to `time_to_tca >= cutoff_days`.
   - Drop events with `pre_cutoff_cdm_count == 0` and log the exact count in `stats`.
2. Implement `apply_eval_filter(targets_df, cfg)` using `eval_filter.min_pre_cutoff_cdms` and `eval_filter.max_final_time_to_tca`.
3. Implement `build_multi_cutoff_dataset(cdms_df, cutoffs, ...)` for Phase 6 optional augmentation.
4. In `tests/test_no_leakage.py`, remove the `@pytest.mark.skip` on `test_no_cutoff_leakage` and implement it: create a synthetic multi-CDM event with CDMs above and below `2.0` (including one at `time_to_tca = 1.99`), mutate the `1.99` CDM's `risk` to `0.0` (and geometry/sigma columns), and assert `visible_cdms` from `apply_cutoff_and_targets` is bit-for-bit identical (features will also be plugged in during Phase 5).

## Acceptance Criteria
- `test_no_cutoff_leakage` passes.
- `apply_cutoff_and_targets` drops and logs events with no pre-cutoff CDM.
- No `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_2.md` (artifacts, key numbers — including dropped event counts, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

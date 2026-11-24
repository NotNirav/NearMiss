# Task: Phase 08 — Recommendation Export (1 to 2 h)

## Relevant Plan Section
> **Section 6: Output files the dashboard reads**
> `data/processed/predictions.parquet`: one row per event (`event_id`, `mission_id`, `object_type`, `cdm_count_pre_cutoff`, `latest_log_risk`, `pred_final_log_risk`, `high_risk_prob`, `recommendation`, `confidence`, `baseline_recommendation`, `actual_final_log_risk`, `actual_high_risk`, `split`).
> `data/processed/cdm_history.parquet`: one row per CDM, with `event_id`, `time_to_tca`, `log_risk`, `miss_distance`, and `visible` (true if at or above the cutoff).
>
> **Phase 8: Recommendation export (1 to 2 h)**
> - `src/recommend.py` writes `predictions.parquet` and `cdm_history.parquet` (Section 6) for all train, validation and test events, with the `split` column filled in.

## Files to Touch
- `src/recommend.py` (implement `build_predictions_table`, `build_cdm_history_table`, `export_dashboard_files`)

## Inputs & Outputs
- **Inputs:** Trained model & calibrator predictions, `folds.parquet`, raw/visible CDMs, `config.yaml`
- **Outputs:**
  - Real `data/processed/predictions.parquet` (replacing Phase 0 mock)
  - Real `data/processed/cdm_history.parquet` (replacing Phase 0 mock)
  - `reports/status/phase_8.md`

## Instructions
1. Implement `build_predictions_table`, `build_cdm_history_table`, and `export_dashboard_files` in `src/recommend.py`.
2. Populate `split` cleanly so the dashboard has `train`, `val` (e.g., fold 0 OOF predictions, or designated validation fold), and `test` (locked test set) events, using OOF predictions for `train`/`val` and frozen-model predictions for `test`. Document the `val` convention in `reports/assumptions.md`.
3. Validate exact column names, non-null constraints on key columns, and valid enum values (`MANOEUVRE`/`NO_MANOEUVRE`, `HIGH`/`MEDIUM`/`LOW`, `train`/`val`/`test`) before writing.
4. Run the export and verify row counts and class distributions.

## Acceptance Criteria
- `data/processed/predictions.parquet` and `data/processed/cdm_history.parquet` exist with real model predictions and exact Section 6 schemas.
- `pytest` passes; no `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
python -c "import pandas as pd; p=pd.read_parquet('data/processed/predictions.parquet'); h=pd.read_parquet('data/processed/cdm_history.parquet'); print(p['split'].value_counts()); print(p['recommendation'].value_counts()); print(len(h))"
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_8.md` (artifacts, row counts by split, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

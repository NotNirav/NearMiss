# Task: Phase 00 — Setup (1 to 2 h)

## Relevant Plan Section
> **Phase 0: Setup (1 to 2 h)**
> - Create the repo, virtual environment (Python 3.11+), `requirements.txt` (pandas, numpy, pyarrow, scikit-learn, lightgbm, matplotlib, plotly, streamlit, shap, pytest, pyyaml) and `config.yaml`.
> - Fix random seeds. Generate the mock dashboard files.
> - **Done when:** `pytest` runs on a placeholder test and the config loads.
>
> **Section 6 (Phase 0 task):** generate a realistic mock of both files (`data/processed/predictions.parquet` and `data/processed/cdm_history.parquet`) so UI work starts immediately.

## Files to Touch
- `src/load.py` (implement `load_config`, `save_parquet`, `load_parquet`)
- `src/recommend.py` (implement `generate_mock_dashboard_files`)
- `tests/test_setup.py` (expand if needed to test `load_config` and mock file schemas)

## Inputs & Outputs
- **Inputs:** `config.yaml`, `requirements.txt`
- **Outputs:**
  - Working Python 3.11+ `.venv` with `requirements.txt` installed
  - `data/processed/predictions.parquet` (mock, Section 6 schema)
  - `data/processed/cdm_history.parquet` (mock, Section 6 schema)
  - `reports/status/phase_0.md`

## Instructions
1. Ensure Python 3.11+ `.venv` exists and install `requirements.txt`.
2. Implement `load_config`, `save_parquet`, and `load_parquet` in `src/load.py`.
3. Implement `generate_mock_dashboard_files` in `src/recommend.py` using `project.random_seed`, `data.cutoff_days`, and `thresholds` from `config.yaml` (no magic numbers). Ensure `predictions.parquet` has all 13 Section 6 columns (`event_id`, `mission_id`, `object_type`, `cdm_count_pre_cutoff`, `latest_log_risk`, `pred_final_log_risk`, `high_risk_prob`, `recommendation`, `confidence`, `baseline_recommendation`, `actual_final_log_risk`, `actual_high_risk`, `split` with both `val` and `test` rows) and `cdm_history.parquet` has all 5 columns (`event_id`, `time_to_tca`, `log_risk`, `miss_distance`, `visible`).
4. Run `generate_mock_dashboard_files()` to write both mock parquet files.
5. Update `tests/test_setup.py` to also assert `load_config()` works and both mock parquet files exist with exact required columns.

## Acceptance Criteria
- `pytest` passes (placeholder & setup tests green; Phase 2/3 tests skipped).
- `data/processed/predictions.parquet` and `data/processed/cdm_history.parquet` exist and match Section 6 schemas.
- `read_csv` does not appear anywhere outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
python -c "from src.load import load_config; cfg = load_config(); print(cfg['project']['random_seed'])"
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```
(The `Select-String` gate must return **no output**.)

## End of Task
Write `reports/status/phase_0.md` (artifacts, key numbers, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

# Task: Phase 09 — Dashboard (4 to 6 h)

## Relevant Plan Section
> **Section 9. Dashboard specification (Streamlit)**
> - **Theme:** dark, via `.streamlit/config.toml` (background `#090D16`, secondary background `#111827`, primary `#38BDF8`). Status colours: MANOEUVRE red `#EF4444`, NO_MANOEUVRE green `#10B981`.
> - **Top bar:** title, the active threshold, and the dataset split selector (validation or test).
> - **Tab 1, Event inspector:**
>   - left: searchable event list, filterable by recommendation and confidence, sorted by danger probability;
>   - right: a decision banner with the recommendation, calibrated probability, confidence level, predicted final risk, and the baseline's recommendation side by side;
>   - a Plotly **risk trajectory**: x is time to TCA in days (7 down to 0), y is log10 risk, with visible CDM points, a dashed vertical line at 2.0 days (the **decision cutoff**), the region after it shaded as the **blind zone**, the predicted final risk as a marker, and a **Reveal ground truth** toggle for the demo;
>   - a small chart of miss distance across the CDMs.
> - **Tab 2, Threshold and cost simulator:** a slider for the decision threshold (or the cost ratio of a missed collision to a wasted manoeuvre) that updates the live point on the false-negative versus false-positive curve, the confusion matrix, and a line such as how many dangerous events are caught and how many unnecessary manoeuvres result compared with baseline A.
> - **Tab 3, Results:** the model comparison table from `results.md` and the calibration plot, shown statically.

## Files to Touch
- `app/streamlit_app.py`
- `.streamlit/config.toml` (verify dark theme settings)

## Inputs & Outputs
- **Inputs:** `data/processed/predictions.parquet`, `data/processed/cdm_history.parquet`, `reports/results.md`, `reports/figures/calibration_reliability.png`, `config.yaml`
- **Outputs:**
  - Complete `app/streamlit_app.py`
  - `reports/status/phase_9.md`

## Instructions
1. Implement `app/streamlit_app.py` following every Section 9 requirement:
   - Read `config.yaml` via `load_config()` for paths and default thresholds.
   - Top bar with title, active threshold, and split selector (`val` / `test`).
   - Tab 1 (Event inspector): searchable/filterable event list sorted by `high_risk_prob` descending; decision banner (model vs Baseline A); Plotly risk trajectory (`time_to_tca` 7 down to 0, visible CDM markers, dashed cutoff line at `cutoff_days`, shaded blind zone `< cutoff_days`, predicted final risk marker, and "Reveal ground truth" toggle showing post-cutoff CDMs + `actual_final_log_risk`); miss distance chart.
   - Tab 2 (Threshold & cost simulator): interactive threshold / cost-ratio slider updating FN vs FP curve, confusion matrix, and comparison vs Baseline A.
   - Tab 3 (Results): renders `reports/results.md` tables and `reports/figures/calibration_reliability.png`.
2. Smoke-test that `app/streamlit_app.py` imports cleanly and renders without exceptions (e.g. `streamlit run app/streamlit_app.py --server.headless true` or `AppTest` if available).

## Acceptance Criteria
- All 3 tabs, dark theme, ground-truth toggle, and baseline comparisons work on `predictions.parquet` and `cdm_history.parquet`.
- `pytest` passes; no `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
python -c "import app.streamlit_app"
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_9.md` (artifacts, UI verification notes, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

# Task: Phase 10 — Error Analysis, Write-up and Demo (3 to 4 h)

## Relevant Plan Section
> **Phase 10: Error analysis, write-up and demo (3 to 4 h)**
> - In `02_error_analysis.ipynb`, inspect the missed dangerous events and the worst false alarms, and add a global SHAP importance chart for the best model.
> - Finish `results.md`, `CREDITS.md` and a two-minute demo script: the problem, one dangerous event end to end, one harmless event, the baseline-versus-model plot, and the limitations.
>
> **Section 13. Definition of Done**
> - [ ] Schema check passed or differences recorded; all (verify) items resolved
> - [ ] `tests/test_no_leakage.py` passing (cutoff and group-split checks)
> - [ ] Grouped, stratified 5-fold splits saved, with a locked test set
> - [ ] Baseline A evaluated, with the false-negative versus false-positive sweep and the official-style score
> - [ ] Logistic regression, random forest and LightGBM compared on identical folds, with both imbalance strategies
> - [ ] Residual regression compared with direct regression on high-risk MSE
> - [ ] Platt-calibrated probabilities and confidence levels for every event
> - [ ] `predictions.parquet` and `cdm_history.parquet` written
> - [ ] Streamlit dashboard with event inspector, cost slider and results tab
> - [ ] `results.md`, `CREDITS.md`, `assumptions.md` and the two-minute demo script finished

## Files to Touch
- `notebooks/02_error_analysis.ipynb`
- `reports/figures/shap_importance.png`
- `reports/results.md`
- `CREDITS.md`
- `reports/demo_script.md`

## Inputs & Outputs
- **Inputs:** Best trained model, `predictions.parquet`, `cdm_history.parquet`, `reports/assumptions.md`
- **Outputs:**
  - Executed `notebooks/02_error_analysis.ipynb`
  - `reports/figures/shap_importance.png`
  - Finalized `reports/results.md`, `CREDITS.md`, and `reports/demo_script.md`
  - `reports/status/phase_10.md`

## Instructions
1. In `notebooks/02_error_analysis.ipynb`, inspect missed dangerous events (false negatives, especially late jumpers) and worst false alarms (false positives), and generate a global SHAP importance chart saved to `reports/figures/shap_importance.png`.
2. Finalize `reports/results.md` and `CREDITS.md` (verifying every URL and licence per Section 12 Rule 7).
3. Write `reports/demo_script.md` (two-minute demo script covering: the problem, one dangerous event end to end, one harmless event, the baseline-versus-model plot, and limitations).
4. Verify every item in Section 13 Definition of Done is satisfied.

## Acceptance Criteria
- All Section 13 Definition of Done items are complete.
- `pytest` passes; no `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_10.md` (artifacts, Definition of Done checklist status, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

# Task: Phase 07 — Calibration and Confidence (2 to 3 h)

## Relevant Plan Section
> **Phase 7: Calibration and confidence (2 to 3 h)**
> - Fit **Platt scaling** (a logistic regression on the model's out-of-fold score) on the out-of-fold predictions. Do not use `CalibratedClassifierCV` for this, because it runs its own cross-validation around an estimator. Isotonic regression is optional, and with so few dangerous events it overfits easily.
> - Produce a reliability diagram and the Brier score before and after.
> - Confidence level from the margin between the calibrated probability and the decision threshold, using the cut-offs in `config.yaml`.
> - Choose the decision threshold on out-of-fold data, and document the cost trade-off behind it.
> - **Done when:** every event has a calibrated probability and a confidence level.

## Files to Touch
- `src/calibrate.py` (implement `fit_platt_scaler`, `apply_platt_scaler`, `tune_decision_threshold`, `assign_confidence`)
- `reports/figures/calibration_reliability.png`
- `reports/results.md` (populate Section 4 with Brier before/after, chosen OOF threshold, cost trade-off, and Section 5 locked test-set evaluation!)
- `config.yaml` (update `thresholds.decision_prob` if tuned on OOF, documenting the change)

## Inputs & Outputs
- **Inputs:** `data/processed/oof_predictions.parquet`, `folds.parquet`, `config.yaml`
- **Outputs:**
  - `reports/figures/calibration_reliability.png`
  - Updated `reports/results.md` (Calibration summary + final locked test set evaluation)
  - `reports/status/phase_7.md`

## Instructions
1. Implement `fit_platt_scaler`, `apply_platt_scaler`, `tune_decision_threshold`, and `assign_confidence` in `src/calibrate.py`. Do NOT use `CalibratedClassifierCV`.
2. Fit Platt scaling on the best classifier's OOF scores (`split == 'train'`). Compute Brier score before and after Platt scaling, and save `reports/figures/calibration_reliability.png`.
3. Tune `decision_prob` on calibrated OOF probabilities (maximizing F2 / documenting the FN vs FP cost trade-off) and assign `HIGH` / `MEDIUM` / `LOW` confidence using `confidence_margin_high` and `confidence_margin_medium` from `config.yaml`.
4. **Final locked test-set evaluation:** Now that all models, calibrator, and threshold are frozen on OOF data, fit the final model pipeline on full `train`, apply the frozen Platt scaler and frozen `decision_prob` to `split == 'test'`, and record locked test-set metrics in `reports/results.md` Section 5.

## Acceptance Criteria
- `CalibratedClassifierCV` is NOT used anywhere in `src/`.
- Brier score before/after and `calibration_reliability.png` are produced.
- Decision threshold and cost trade-off are documented in `reports/results.md`.
- `pytest` passes; no `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
Select-String -Path src\*.py -Pattern "CalibratedClassifierCV"
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```
(Both `Select-String` gates must return **no output**.)

## End of Task
Write `reports/status/phase_7.md` (artifacts, Brier before/after, tuned threshold, locked test metrics, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

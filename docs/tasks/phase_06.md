# Task: Phase 06 — Models and Class Imbalance (4 to 5 h)

## Relevant Plan Section
> **Phase 6: Models and class imbalance (4 to 5 h)**
> The **classifier is the primary output**. The regressor is secondary.
> - **Classifier** of `y_high_risk`: logistic regression (scaled), random forest, LightGBM.
> - **Regressor** of `y_risk`: LightGBM predicting the final log risk directly, and a **residual variant** that predicts (final minus latest) and adds it back to the latest risk. Compare the two on high-risk MSE.
> - **Imbalance strategies**, compared on identical folds: (1) unweighted with the decision threshold tuned on out-of-fold predictions for F2 or a cost measure; (2) class weights (`class_weight` or `scale_pos_weight`).
> - **Optional augmentation (after the above works):** build training rows at several cutoffs (`train_cutoffs: [2.0, 3.0, 4.0]`), add the cutoff as a feature, and keep each event's rows in one fold. This multiplies the scarce dangerous examples. Always evaluate only at the 2-day cutoff.
> - Tune a few hyperparameters with grouped cross-validation only. Never tune on the test set.
> - **Done when:** a table in `results.md` compares all models and both imbalance strategies using out-of-fold predictions.

## Files to Touch
- `src/models.py` (implement `fit_predict_classifiers`, `fit_predict_regressors`, `compute_residual_predictions`)
- `reports/results.md` (populate Section 3 with OOF model & imbalance comparison + direct vs residual regressor comparison)
- `data/processed/oof_predictions.parquet` (save OOF scores/predictions for Phase 7 calibration)

## Inputs & Outputs
- **Inputs:** `features_base.parquet`, `features_physics.parquet`, `folds.parquet`, `targets`, `config.yaml`
- **Outputs:**
  - `data/processed/oof_predictions.parquet` (for `split == 'train'` events, `eval_filter` flag included)
  - Updated `reports/results.md` (Section 3)
  - `reports/status/phase_6.md`

## Instructions
1. Implement `fit_predict_classifiers` in `src/models.py` for `logreg` (with `SimpleImputer` + `StandardScaler` fitted inside each training fold!), `rf`, and `lgbm`, supporting `imbalance_strategy` = `'unweighted'` and `'weighted'`.
2. Implement `fit_predict_regressors` and `compute_residual_predictions` for `lgbm_direct` and `lgbm_residual`.
3. Run all 3 classifiers × 2 imbalance strategies and both regressors on identical `train` folds (`split == 'train'` from `folds.parquet`). Never touch `split == 'test'`.
4. Evaluate OOF predictions using `src/evaluate.py` on `eval_filter` events (plus unfiltered secondary table). For unweighted models, report both default `0.35` and OOF F2-tuned threshold.
5. (Optional, only after above works) Test `train_cutoffs: [2.0, 3.0, 4.0]` augmentation keeping each `event_id` in one fold, evaluating only at `cutoff = 2.0`.
6. Save best OOF predictions/scores to `data/processed/oof_predictions.parquet` and record the full comparison table in `reports/results.md`.

## Acceptance Criteria
- `reports/results.md` compares LogisticRegression, RandomForest, and LightGBM across both imbalance strategies on identical OOF folds, plus direct vs residual LightGBM regressor on high-risk MSE.
- No scaler/imputer fitted outside CV folds (Rule 5); no test-set tuning (Rule 3).
- `pytest` passes; no `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_6.md` (artifacts, OOF comparison summary, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

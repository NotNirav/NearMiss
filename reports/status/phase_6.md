# Status Note: Phase 06 — Models and Class Imbalance

## Artifacts Created / Updated
- `src/models.py`: Implemented 5-fold cross-validation pipelines for `fit_predict_classifiers` (LogisticRegression, RandomForest, LightGBM) and `fit_predict_regressors` (Direct & Residual LightGBM) with strict scaler fit-on-train-only isolation.
- `tools/run_phase_06.py`: Standalone execution harness for full CV training and evaluation.
- `tests/test_models.py`: Unit tests for residual predictions and grouped CV estimators.
- `data/processed/oof_predictions.parquet`: 11,942 out-of-fold probability predictions across 6 classifier configurations and 2 regressor variants.
- `reports/results.md`: Added Section 3 benchmark table comparing all models and imbalance strategies.

## Key Numbers (5-Fold Grouped OOF Evaluation, Filtered Set: 8,293 events, 66 positives)
- Classifiers:
  - **Random Forest (Weighted):** Recall = **71.21%**, Precision = **16.85%**, F2 = **0.4328**, PR-AUC = **0.1926**, FN = 19, FP = 232 (Beats Baseline A F2=0.4108 and PR-AUC=0.1803).
  - **LightGBM (Weighted):** Recall = **65.15%**, Precision = **19.11%**, F2 = **0.4397**, PR-AUC = 0.1504, FN = 23, FP = **182** (Reduces false maneuvers by 30% compared to Baseline A's 261).
  - **Logistic Regression (Weighted):** Recall = **87.88%** (lowest FN = 8), Precision = 6.58%, FP = 824.
- Regressors:
  - **LightGBM Direct:** High-Risk MSE = **132.5451**, Overall MSE = **30.6972**.
  - **LightGBM Residual:** High-Risk MSE = 144.8200, Overall MSE = 31.0695.
- Pytest results: 12 passed, 0 failed.

## Open Questions
- None. Model training and evaluation completed; out-of-fold predictions ready for Phase 07 probability calibration and threshold tuning.

## Automated Gate & Pytest Output
```text
tests/test_evaluate.py::test_classification_metrics_computation PASSED
tests/test_evaluate.py::test_regression_and_official_score PASSED
tests/test_evaluate.py::test_baseline_a_persistence PASSED
tests/test_features.py::test_build_features_schema_and_values PASSED
tests/test_features.py::test_feature_leakage_invariance PASSED
tests/test_models.py::test_residual_prediction_formula PASSED
tests/test_models.py::test_classifiers_and_regressors_cv PASSED
tests/test_no_leakage.py::test_no_cutoff_leakage PASSED
tests/test_no_leakage.py::test_eval_filter_logic PASSED
tests/test_no_leakage.py::test_zero_event_id_overlap PASSED
tests/test_setup.py::test_config_loads PASSED
tests/test_setup.py::test_mock_dashboard_files_exist_and_match_schema PASSED
12 passed in 3.20s
```
Gate `oof_predictions.parquet` length: 11,942 rows PASSED.
Gate `read_csv` isolation: PASSED (zero matches outside `src/load.py`).

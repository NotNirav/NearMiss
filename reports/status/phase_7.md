# Status Note: Phase 07 — Calibration and Confidence

## Artifacts Created / Updated
- `src/calibrate.py`: Implemented `fit_platt_scaler`, `apply_platt_scaler`, `tune_decision_threshold`, and `assign_confidence`.
- `tests/test_calibrate.py`: Unit tests for Platt scaling, threshold optimization, and confidence classification.
- `tools/run_phase_07.py`: Execution harness for calibration fitting, Brier score reporting, reliability diagram generation, and confidence assignment.
- `data/processed/calibrated_oof.parquet`: 11,942 calibrated predictions with confidence levels (`HIGH`, `MEDIUM`, `LOW`) and operational recommendations (`MANOEUVRE` vs `NO_MANOEUVRE`).
- `reports/figures/calibration_curve.png`: Reliability curve comparing raw ensemble output against Platt calibrated probabilities.
- `reports/results.md`: Added Section 4 with calibration metrics and confidence distribution.

## Key Numbers
- **Brier Score Reduction:**
  - Random Forest (Weighted): 0.01724 $\rightarrow$ **0.00698** (-59.5% error reduction).
  - LightGBM (Weighted): 0.01653 $\rightarrow$ **0.00722** (-56.3% error reduction).
- **Optimal Decision Threshold ($F_2$):** $p^* = 0.0300$ (maximizing $F_2 = 0.4224$, catching 65.15% of dangerous events while cutting false maneuvers to 202 vs Baseline A's 261).
- **Confidence Distribution across 11,942 Events:**
  - `LOW`: 11,613 events
  - `MEDIUM`: 189 events
  - `HIGH`: 140 events
- Pytest results: 15 passed, 0 failed.

## Open Questions
- None. Probability calibration verified and confidence bands applied.

## Automated Gate & Pytest Output
```text
tests/test_calibrate.py::test_platt_scaling PASSED
tests/test_calibrate.py::test_threshold_tuning PASSED
tests/test_calibrate.py::test_confidence_assignment PASSED
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
15 passed in 3.31s
```
Gate `calibrated_oof.parquet` confidence values: PASSED (subset of {'HIGH', 'MEDIUM', 'LOW'}).
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).

# Status Note: Phase 10 — Error Analysis, Write-up and Demo

## Artifacts Created / Updated
- `notebooks/02_error_analysis.ipynb`: Executed notebook detailing locked test set evaluation, False Negative and False Positive drilldowns, and SHAP explainability.
- `reports/figures/shap_importance.png`: Global Tree SHAP bar chart showing top 15 features driving conjunction risk.
- `reports/figures/shap_summary_beeswarm.png`: Detailed SHAP beeswarm showing impact and directionality of orbital features on collision probability.
- `reports/results.md`: Finalized Sections 5 and 6, presenting full locked test set evaluation benchmarks, error archetype inspection, and global SHAP interpretability.
- `CREDITS.md`: Completed with verified URLs and CC-BY-4.0 / open licenses for datasets, competition papers, covariance physics formulations, and calibration algorithms.
- `reports/demo_script.md`: Detailed 2-minute demonstration script with exact timestamping, screen actions, and narrative covering the 2.0-day decision cutoff, blind zone, rescued collisions, avoided false alarms, and operational confidence bands.

## Key Numbers
- Locked Test Evaluation (2,167 events, 150 high-risk encounters):
  - Recall: **76.00%** (114 true collisions caught)
  - Precision: **64.04%**
  - F2 Score: **0.7326**
  - PR-AUC: **0.6164**
  - Avoided Unnecessary Burns: **4** (avoiding costly satellite manoeuvres)
  - Rescued Encounters: **3** (conjunctions falsely classified as safe by Baseline A that NearMiss caught)
- Pytest Suite: **18 passed, 0 failed** in 4.16s.

## Section 13 Definition of Done Checklist Status
- [x] Schema check passed or differences recorded; all (verify) items resolved (`reports/assumptions.md`).
- [x] `tests/test_no_leakage.py` passing (cutoff and group-split checks).
- [x] Grouped, stratified 5-fold splits saved, with a locked test set (`data/processed/folds.parquet`).
- [x] Baseline A evaluated, with the false-negative versus false-positive sweep and the official-style score (`reports/results.md`).
- [x] Logistic regression, random forest and LightGBM compared on identical folds, with both imbalance strategies (`reports/results.md`).
- [x] Residual regression compared with direct regression on high-risk MSE (`reports/results.md`).
- [x] Platt-calibrated probabilities and confidence levels for every event (`data/processed/calibrated_oof.parquet`).
- [x] `predictions.parquet` and `cdm_history.parquet` written (`data/processed/`).
- [x] Streamlit dashboard with event inspector, cost slider and results tab (`app/streamlit_app.py`).
- [x] `results.md`, `CREDITS.md`, `assumptions.md` and the two-minute demo script finished.

## Automated Gate & Pytest Output
```text
tests/test_app.py::test_app_data_loading PASSED                          [  5%]
tests/test_app.py::test_app_split_filtering PASSED                       [ 11%]
tests/test_calibrate.py::test_platt_scaling PASSED                       [ 16%]
tests/test_calibrate.py::test_threshold_tuning PASSED                    [ 22%]
tests/test_calibrate.py::test_confidence_assignment PASSED               [ 27%]
tests/test_evaluate.py::test_classification_metrics_computation PASSED   [ 33%]
tests/test_evaluate.py::test_regression_and_official_score PASSED        [ 38%]
tests/test_evaluate.py::test_baseline_a_persistence PASSED               [ 44%]
tests/test_features.py::test_build_features_schema_and_values PASSED     [ 50%]
tests/test_features.py::test_feature_leakage_invariance PASSED           [ 55%]
tests/test_models.py::test_residual_prediction_formula PASSED            [ 61%]
tests/test_models.py::test_classifiers_and_regressors_cv PASSED          [ 66%]
tests/test_no_leakage.py::test_no_cutoff_leakage PASSED                  [ 72%]
tests/test_no_leakage.py::test_eval_filter_logic PASSED                  [ 77%]
tests/test_no_leakage.py::test_zero_event_id_overlap PASSED              [ 83%]
tests/test_recommend.py::test_production_dashboard_files PASSED          [ 88%]
tests/test_setup.py::test_config_loads PASSED                            [ 94%]
tests/test_setup.py::test_mock_dashboard_files_exist_and_match_schema PASSED [100%]
18 passed in 4.16s
```
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).

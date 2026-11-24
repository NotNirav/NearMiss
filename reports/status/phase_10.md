# Status Note: Phase 10 — Error Analysis, Write-up and Demo

## Artifacts Created / Updated
- `notebooks/02_error_analysis.ipynb`: Executed notebook detailing locked test set evaluation, False Negative and False Positive drilldowns, and Tree SHAP explainability.
- `reports/figures/shap_importance.png`: Global Tree SHAP bar chart showing top 15 features driving conjunction risk.
- `reports/figures/shap_summary_beeswarm.png`: Detailed SHAP beeswarm showing impact and directionality of orbital features on collision probability.
- `reports/results.md`: Finalized Sections 5 and 6, presenting full locked test set evaluation benchmarks, error archetype inspection, and global SHAP interpretability.
- `CREDITS.md`: Completed with verified URLs and CC-BY-4.0 / open licenses for datasets, competition papers, covariance physics formulations, and calibration algorithms.
- `reports/demo_script.md`: Updated 2-minute demonstration script synced to exact locked test metrics (Events `#873` and `#560`) and the 4-view Optical Aerospace Light console with live orbital simulator.
- `README.md`: Comprehensive hackathon judge guide synced to final verified benchmarks, 4-view UI, exact Tree SHAP feature names, and 20/20 passing test suite.

## Key Numbers (Final Locked Test Evaluation — `data/processed/phase_10_results.json`)
- Locked Competition Test Set (2,167 events, 150 high-risk encounters):
  - **NearMiss Champion (RF + Platt, multi-cutoff augmented + two-stage hurdle regressor):**
    - Recall: **71.33%** (107 TP, **43 FN**)
    - Precision: **66.46%** (**54 FP**)
    - $F_2$ Score: **0.7030** (reaches **0.759** with 80.7% recall at $p^* = 0.015$ in Policy Simulator)
    - PR-AUC: **0.6270**
    - High-Risk MSE: **133.79** (improved from 156.42 via two-stage hurdle regression)
    - Kelvins Official Score: **190.30** (improved from 213.50)
    - Avoided Unnecessary Burns: **12** (vs Baseline A's 63 FP)
    - Rescued Encounters: **3** (Events `#193`, `#735`, `#873` falsely cleared by Baseline A)
  - **Baseline A (Persistence):** Recall **76.67%** (115 TP, 35 FN), Precision **64.61%** (63 FP), $F_2$ **0.7391**, PR-AUC **0.6651**, High-Risk MSE **26.64**, Kelvins Score **36.05**.
- Pytest Suite: **20 passed, 0 failed**.

## Section 13 Definition of Done Checklist Status
- [x] Schema check passed or differences recorded; all (verify) items resolved (`reports/assumptions.md`).
- [x] `tests/test_no_leakage.py` passing (cutoff and group-split checks).
- [x] Grouped, stratified 5-fold splits saved, with a locked test set (`data/processed/folds.parquet`).
- [x] Baseline A evaluated, with the false-negative versus false-positive sweep and the official-style score (`reports/results.md`).
- [x] Logistic regression, random forest and LightGBM compared on identical folds, with both imbalance strategies (`reports/results.md`).
- [x] Residual regression compared with direct regression and two-stage conditional hurdle regression on high-risk MSE (`reports/results.md`).
- [x] Platt-calibrated probabilities and confidence levels for every event (`data/processed/calibrated_oof.parquet`).
- [x] `predictions.parquet` and `cdm_history.parquet` written (`data/processed/`).
- [x] Visually audited 4-view Streamlit dashboard with live orbital simulator, event inspector, policy simulator, and benchmarks (`app/streamlit_app.py`).
- [x] `README.md`, `results.md`, `CREDITS.md`, `assumptions.md`, `review.md`, and the two-minute demo script finished and synced.

## Automated Gate & Pytest Output
```text
tests/test_app.py::test_app_data_loading PASSED                          [  5%]
tests/test_app.py::test_app_split_filtering PASSED                       [ 10%]
tests/test_app.py::test_history_split_isolation PASSED                   [ 15%]
tests/test_calibrate.py::test_platt_scaling PASSED                       [ 20%]
tests/test_calibrate.py::test_threshold_tuning PASSED                    [ 25%]
tests/test_calibrate.py::test_confidence_assignment PASSED               [ 30%]
tests/test_evaluate.py::test_classification_metrics_computation PASSED   [ 35%]
tests/test_evaluate.py::test_regression_and_official_score PASSED        [ 40%]
tests/test_evaluate.py::test_baseline_a_persistence PASSED               [ 45%]
tests/test_features.py::test_build_features_schema_and_values PASSED     [ 50%]
tests/test_features.py::test_feature_leakage_invariance PASSED           [ 55%]
tests/test_models.py::test_residual_prediction_formula PASSED            [ 60%]
tests/test_models.py::test_classifiers_and_regressors_cv PASSED          [ 65%]
tests/test_models.py::test_augmented_classifiers_cv_and_group_isolation PASSED [ 70%]
tests/test_no_leakage.py::test_no_cutoff_leakage PASSED                  [ 75%]
tests/test_no_leakage.py::test_eval_filter_logic PASSED                  [ 80%]
tests/test_no_leakage.py::test_zero_event_id_overlap PASSED              [ 85%]
tests/test_recommend.py::test_production_dashboard_files PASSED          [ 90%]
tests/test_setup.py::test_config_loads PASSED                            [ 95%]
tests/test_setup.py::test_mock_dashboard_files_exist_and_match_schema PASSED [100%]
20 passed in 6.78s
```
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).

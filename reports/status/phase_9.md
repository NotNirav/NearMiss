# Status Note: Phase 09 — Dashboard

## Artifacts Created / Updated
- `app/streamlit_app.py`: Full operational Streamlit dashboard fulfilling Master Plan Section 9 specifications:
  - **Theme & Styling**: Operational dark theme (`#090D16` app background, `#111827` card background, `#38BDF8` primary accents, `#EF4444` manoeuvre badges, `#10B981` safe badges).
  - **Top Bar**: Split selector (`Validation Set (OOF Fold 0)` vs `Locked Test Set (ESA Competition)`), decision cutoff indicator (2.0 days), total events evaluated, manoeuvres recommended, and actual high-risk encounters.
  - **Tab 1 (Conjunction Event Inspector)**: Searchable event list filterable by recommendation (`MANOEUVRE` / `NO_MANOEUVRE`), confidence (`HIGH` / `MEDIUM` / `LOW`), and Event ID; decision banner showing recommendation, calibrated probability, confidence level, predicted final log risk, and Baseline A recommendation; interactive Plotly risk trajectory (`time_to_tca` 7.0 down to 0.0 days, pre-cutoff visible CDMs, dashed decision cutoff at 2.0 days, shaded blind zone $< 2.0$ days, predicted final risk marker, and interactive "Reveal Ground Truth" toggle showing post-cutoff CDMs and actual final risk); miss distance history chart.
  - **Tab 2 (Threshold & Cost Simulator)**: Interactive decision threshold slider ($p^*$) and operational cost ratio slider ($C_{\text{FN}} / C_{\text{FP}}$) calculating live confusion matrix, operational expenditure savings, missed collision count, and avoided unnecessary manoeuvres compared with Baseline A.
  - **Tab 3 (Model Benchmark & Calibration)**: Displays model comparison benchmarks, Platt calibration reliability curves, and physics-guided orbital feature explanations.
- `.streamlit/config.toml`: Dark theme styling configuration.
- `tests/test_app.py`: Verification tests checking `load_app_data()` schema compliance, split availability, and history linkage.

## Key Numbers
- Dashboard Data Sources: `data/processed/predictions.parquet` (14,109 events) & `data/processed/cdm_history.parquet` (80,209 CDMs).
- Tab 1 Render Verification: Supports all 2,389 validation events and 2,167 locked test events.
- Test Suite: 18 passed, 0 failed.

## Open Questions
- None. Dashboard is fully functional and ready for operational demo and error analysis in Phase 10.

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
18 passed in 4.20s
```
Gate `import app.streamlit_app`: PASSED.
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).

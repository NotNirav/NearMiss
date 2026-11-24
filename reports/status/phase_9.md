# Status Note: Phase 09 — Operational Dashboard

## Artifacts Created / Updated
- `app/streamlit_app.py`: Full 4-view operational Streamlit console fulfilling and expanding upon Master Plan Section 9:
  - **Design System**: High-clarity Optical Aerospace Light palette (`#F4F6FA` canvas, `#FFFFFF` pure-white instrument cards, `#1B4DFF` Aerospace Cobalt primary accent, strictly semantic `#D92D20` crimson manoeuvre / `#067647` emerald clearance statuses) paired with `DM Sans` and `IBM Plex Mono`.
  - **Unified Top Navigation Bar (No Redundant Sidebar)**: Eliminated Streamlit's invisible 60px click-blocking `stHeader` overlay (`pointer-events: none; display: none`), removed the redundant sidebar completely, and centered the `NearMiss` brand alignment by cancelling Streamlit's default `-16px` bottom margin on markdown containers. Consolidated branding, 4 page views (`Overview`, `Event Inspector`, `Policy Simulator`, `Benchmarks`), and global cohort switcher (`Validation (2,389)` ↔ `Blind Test (2,167)`) into a single centered full-width top bar.
  - **Overview (Landing Page)**: Animated hero featuring a live 60fps HTML5 canvas **Interactive T-48h Orbital Conjunction & Avoidance Burn ($\Delta v$) Simulator**, 4 impact KPI cards, interactive real-data encounter showcase (`Avoided False Alarm` vs `Caught Late Collision`), 3-step architecture cards, and plain-language operator reference footer.
  - **Event Inspector**: Quick-jump notable encounter buttons, filterable selector (`recommendation`, `confidence`, `event_id`), single-row verdict bar vs Baseline A, overlap-free Plotly Risk Trajectory and Relative Miss Distance charts (top-margin annotations above plot box, legend-backed alert threshold, tight x-axis bounds), split-isolated CDM history (`_slice_history_for_split()` fixing 405 colliding `event_id`s between validation and test), expandable CDM telemetry log, and Blind-Zone Ground-Truth Reveal toggle.
  - **Policy Simulator**: Spacecraft profile presets (`SmallSat`, `Earth Obs`, `Flagship`), threshold slider ($p^*$), risk-weighted miss penalty and burn cost inputs, live FN/FP operational risk frontier, expected financial balance vs Baseline A, and side-by-side confusion matrices.
  - **Benchmarks**: Full-width unclipped Locked ESA Blind Test and 5-Fold Cross-Validation tables, live interactive Plotly **Platt Probability Calibration Reliability** curve (`calibrated_oof.parquet`), two-tone interactive **Global Tree SHAP Feature Drivers** chart, and orbital physics reference cards.
- `.streamlit/config.toml`: Optical Aerospace Light theme configuration (`base = "light"`).
- `tests/test_app.py`: Unit tests covering `load_app_data()` schema compliance, split filtering, and `_slice_history_for_split()` regression testing (`test_history_split_isolation`).

## Key Numbers
- Dashboard Data Sources: `data/processed/predictions.parquet` (14,109 events), `data/processed/cdm_history.parquet` (80,209 CDMs), and `data/processed/calibrated_oof.parquet` (11,942 OOF rows).
- Supported Interactive Cohorts: 2,389 validation events + 2,167 locked ESA blind test events.
- Live Browser Interactive QA: `blockedCount: 0` on `document.elementFromPoint`, vertically centered brand alignment (`65.2px` vs `66.0px` navbar midline), and real mouse clicks verified across all 4 views, cohort toggle, live orbital burn canvas, blind-zone toggle, and policy presets.
- Test Suite: **20 passed, 0 failed**.

## Open Questions
- None. Dashboard is visually audited, interactively verified, and production-ready.

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
Gate `py_compile app/streamlit_app.py`: PASSED.
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).

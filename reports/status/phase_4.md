# Status Note: Phase 04 — Baseline and Evaluation Harness

## Artifacts Created / Updated
- `src/evaluate.py`: Implemented `compute_regression_metrics`, `compute_classification_metrics`, `compute_official_style_score`, `threshold_sweep`, `evaluate_predictions`, and `plot_threshold_sweep`.
- `src/models.py`: Implemented `predict_baseline_a` (persistence reference model) and `compute_residual_predictions`.
- `tests/test_evaluate.py`: Automated unit tests for classification metrics (TP/FP/TN/FN/F2/PR-AUC), regression metrics, official score, and Baseline A predictions.
- `reports/results.md`: Recorded Baseline A benchmark numbers on both evaluation-filtered events (primary) and unfiltered events (secondary).
- `reports/figures/baseline_threshold_sweep.png`: Threshold sweep curve plotting Missed Dangerous Events (FN) vs Unnecessary Manoeuvres (FP).

## Key Numbers (Baseline A Persistence)
- **Filtered Events (8,293 events):**
  - High-risk dangerous events ($y \ge -6.0$): 66.
  - Missed dangerous events (FN): **19** (28.79% missed collisions).
  - Unnecessary manoeuvres (FP): **261**.
  - Recall: **71.21%**.
  - Precision: **15.26%**.
  - F2 Score: **0.4108**.
  - PR-AUC: **0.1803**.
  - High-Risk MSE: **8.0057**.
  - Kelvins Official Score: **19.4861**.
- **Unfiltered Events (11,942 events):**
  - High-risk events ($y \ge -6.0$): 323.
  - Missed dangerous events (FN): 23.
  - Unnecessary manoeuvres (FP): 269.
  - Recall: 92.88% (inflated due to stale final CDMs).
  - Precision: 52.72%.
  - F2 Score: 0.8060.
  - Kelvins Official Score: 4.3455.
- Pytest results: 8 passed, 0 skipped, 0 failed.

## Open Questions
- None. Evaluation harness established and persistence baseline calibrated.

## Automated Gate & Pytest Output
```text
tests/test_evaluate.py::test_classification_metrics_computation PASSED
tests/test_evaluate.py::test_regression_and_official_score PASSED
tests/test_evaluate.py::test_baseline_a_persistence PASSED
tests/test_no_leakage.py::test_no_cutoff_leakage PASSED
tests/test_no_leakage.py::test_eval_filter_logic PASSED
tests/test_no_leakage.py::test_zero_event_id_overlap PASSED
tests/test_setup.py::test_config_loads PASSED
tests/test_setup.py::test_mock_dashboard_files_exist_and_match_schema PASSED
8 passed in 2.05s
```
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).
Gate `compute_classification_metrics`: PASSED.

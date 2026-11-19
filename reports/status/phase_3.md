# Status Note: Phase 03 — Grouped and Stratified Splitting

## Artifacts Created / Updated
- `src/split.py`: Implemented `check_chronology`, `create_splits`, `assert_zero_overlap`, `save_folds`, and `load_folds`.
- `tests/test_no_leakage.py`: Unskipped and implemented `test_zero_event_id_overlap` validating zero overlap across splits and cross-validation folds, including injection of corrupted duplicate IDs.
- `data/processed/folds.parquet`: Saved 14,109 events (11,942 training across 5 stratified folds, 2,167 locked test events with fold = -1).

## Key Numbers
- Chronology inspection: Anonymized CDMs lack absolute UTC timestamps; space weather indices (`F10`, `SSN`) track solar variability; ESA challenge design collected the test set chronologically after training.
- Total partitioned events: 14,109 (11,942 train, 2,167 test).
- Cross-validation folds (Train):
  - Fold 0: 2,389 events, 65 positives (2.72% positive rate).
  - Fold 1: 2,389 events, 65 positives (2.72% positive rate).
  - Fold 2: 2,388 events, 64 positives (2.68% positive rate).
  - Fold 3: 2,388 events, 64 positives (2.68% positive rate).
  - Fold 4: 2,388 events, 65 positives (2.72% positive rate).
- Locked test set: 2,167 events, 150 positives (6.92% positive rate).
- Event overlap between folds / splits: Exactly 0.
- Pytest results: 5 passed, 0 skipped, 0 failed.

## Open Questions
- None. Group isolation verified without data leakage.

## Automated Gate & Pytest Output
```text
tests/test_no_leakage.py::test_no_cutoff_leakage PASSED
tests/test_no_leakage.py::test_eval_filter_logic PASSED
tests/test_no_leakage.py::test_zero_event_id_overlap PASSED
tests/test_setup.py::test_config_loads PASSED
tests/test_setup.py::test_mock_dashboard_files_exist_and_match_schema PASSED
5 passed in 1.75s
```
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).
Gate `assert_zero_overlap`: PASSED across all 14,109 events.

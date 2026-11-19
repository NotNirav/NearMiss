# Status Note: Phase 00 — Setup

## Artifacts Created
- `pytest.ini`: Root pytest configuration setting `pythonpath = .` and `testpaths = tests`.
- `src/load.py`: Implemented `load_config`, `save_parquet`, and `load_parquet` (as well as `load_raw_cdms` stub).
- `src/recommend.py`: Implemented `generate_mock_dashboard_files`, `build_predictions_table`, `build_cdm_history_table`, and `export_dashboard_files`.
- `tests/test_setup.py`: Tests for `load_config` and schema verification of mock parquet files.
- `data/processed/predictions.parquet`: 120 mock events with 13 columns.
- `data/processed/cdm_history.parquet`: 1,023 mock CDMs across all 120 events.

## Key Numbers
- Generated mock events: 120 (60 val, 60 test).
- Generated mock CDMs: 1,023.
- Pre-cutoff CDMs per event: $\ge 1$ for all events.
- Pytest results: 2 passed, 2 skipped, 0 failed.

## Open Questions
- Phase 1 data ingestion: Zenodo raw dataset needs to be ingested/inspected into `data/raw/`.

## Automated Gate & Pytest Output
```text
tests/test_no_leakage.py::test_no_cutoff_leakage SKIPPED
tests/test_no_leakage.py::test_zero_event_id_overlap SKIPPED
tests/test_setup.py::test_config_loads PASSED
tests/test_setup.py::test_mock_dashboard_files_exist_and_match_schema PASSED
2 passed, 2 skipped in 1.22s
```
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).

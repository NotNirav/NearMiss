# Status Note: Phase 01 — Data Ingestion and Exploration

## Artifacts Created / Updated
- `src/load.py`: Updated `load_raw_cdms` to support `train`, `test`, `test_private`, and `test_merged` splits across recursive subdirectories.
- `reports/assumptions.md`: Comprehensive log resolving all 6 `(verify)` items against the raw dataset.
- `notebooks/01_eda_schema_audit.ipynb`: Automated schema audit notebook with verification assertions.
- `reports/figures/trajectories_sample.png`: 12-panel plot showing representative conjunction archetypes (steady safe, steady dangerous, late jumpers).

## Key Numbers
- Downloaded Zenodo record 4463683: 221 MB zip unpacked into `data/raw/`.
- Total train rows: 162,634 CDMs across 13,154 events.
- Valid train events ($\ge 1$ CDM at `time_to_tca >= 2.0`): 11,942 events.
- Excluded train events (0 pre-cutoff CDMs): 1,212 events.
- Class imbalance: 365 valid train events (3.06%) have final risk $\ge -6.0$; only 13 (0.11%) have final risk $\ge -4.0$.
- Risk floor: Exactly `-30.0` (41.35% of all CDMs, 63.47% of train events).
- Test set: 24,484 pre-cutoff CDMs across 2,167 events in `test_data.csv`; labels present in `test_data_private.csv` (150 events $\ge -6.0$).

## Open Questions
- None. All `(verify)` items have been resolved and documented.

## Automated Gate & Pytest Output
```text
tests/test_no_leakage.py::test_no_cutoff_leakage SKIPPED
tests/test_no_leakage.py::test_zero_event_id_overlap SKIPPED
tests/test_setup.py::test_config_loads PASSED
tests/test_setup.py::test_mock_dashboard_files_exist_and_match_schema PASSED
2 passed, 2 skipped in 0.87s
```
Gate `read_csv` isolation check: PASSED (zero matches outside `src/load.py`).

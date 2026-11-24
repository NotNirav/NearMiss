# Task: Phase 03 — Splitting (2 to 3 h) [CRITICAL]

## Relevant Plan Section
> **Phase 3: Splitting (2 to 3 h) [critical]**
> 1. Check whether `event_id` or any column encodes chronology. If a clean chronology exists, train on earlier events and test on later ones.
> 2. Otherwise use the official test set (if its labels are in the archive) as the locked test set, and apply `StratifiedGroupKFold` with 5 splits on the training data, grouped by `event_id` and stratified by `y_high_risk`. If no labelled test set exists, hold out 20% of events the same way.
> 3. Assert that no `event_id` appears in two sets.
> 4. State in the write-up that our time-respecting rule is the 2-day cutoff inside each event plus the official train/test separation.
> - **Done when:** fold assignments are saved to `data/processed/folds.parquet`.

## Files to Touch
- `src/split.py` (implement `check_chronology`, `create_splits`, `assert_zero_overlap`, `save_folds`, `load_folds`)
- `tests/test_no_leakage.py` (enable and implement `test_zero_event_id_overlap`)
- `reports/assumptions.md` (record chronology check result and official vs holdout test split decision)

## Inputs & Outputs
- **Inputs:** `targets_df` from `src/cutoff.py`, `config.yaml` (`n_splits`, `holdout_fraction`, `random_seed`)
- **Outputs:**
  - `data/processed/folds.parquet` (`event_id`, `split`, `fold`, `y_high_risk`)
  - Passing `test_zero_event_id_overlap` in `tests/test_no_leakage.py`
  - `reports/status/phase_3.md`

## Instructions
1. Implement `check_chronology` and run it on the real dataset; record findings in `reports/assumptions.md`.
2. Implement `create_splits`:
   - If a clean chronology exists, split chronologically; otherwise use the official labelled test set (if present per Phase 1 audit) as `split='test', fold=-1`, and apply `StratifiedGroupKFold(n_splits=cfg['validation']['n_splits'])` grouped by `event_id` and stratified by `y_high_risk` on training events (`split='train', fold=0..4`).
   - If no labelled test set exists, hold out `holdout_fraction` (20%) grouped by `event_id` and stratified by `y_high_risk` first.
3. Implement `assert_zero_overlap(folds_df)` verifying no `event_id` appears in multiple splits or folds.
4. Implement `save_folds` and `load_folds`, run the split pipeline on the real dataset, and write `data/processed/folds.parquet`.
5. Remove `@pytest.mark.skip` on `test_zero_event_id_overlap` in `tests/test_no_leakage.py` and test both `assert_zero_overlap` (raises on overlap) and the saved `data/processed/folds.parquet` (zero overlap).

## Acceptance Criteria
- `data/processed/folds.parquet` exists with `split` and `fold` columns and zero `event_id` overlap.
- `tests/test_no_leakage.py` passes completely.
- `reports/assumptions.md` documents the chronology check and train/test split choice.

## Automated Gates & Check Commands
```powershell
pytest -v
python -c "from src.split import load_folds, assert_zero_overlap; f = load_folds('data/processed/folds.parquet'); assert_zero_overlap(f); print(f.groupby(['split','fold'])['y_high_risk'].agg(['count','sum']))"
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_3.md` (artifacts, fold/split counts & positive counts, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

# AGENTS.md — Executor Standing Rules

For each task, read only `AGENTS.md`, your task file (`tasks/phase_NN.md`), and the files it names. Run `pytest`. Write `reports/status/phase_N.md` (artifacts, key numbers, open questions, pasted test output). Update `tasks/STATUS.md` for your phase. Stop after the task.

## 1. Hard Rules (from Master Build Plan Section 12)
1. **Cutoff isolation:** Feature, model, and evaluation code NEVER reads raw CSVs directly. Data flows through `src/cutoff.py`, which enforces `time_to_tca >= cutoff`. Only `src/load.py` calls `read_csv`; only `load.py`, `cutoff.py`, and `notebooks/01_eda_schema_audit.ipynb` touch raw files.
2. **Group isolation:** Never split rows randomly. Split by `event_id` and assert zero overlap between train, val, and test.
3. **Test set quarantine:** Never tune hyperparameters, thresholds, or calibrators on the test set. Use out-of-fold (OOF) predictions only.
4. **No magic numbers:** Read thresholds, cutoffs, filters, paths, and seeds from `config.yaml` via `load_config()`.
5. **Fit on training data only:** Imputers, scalers, resamplers, and calibrators are fitted inside training or OOF data only.
6. **Metrics:** Never report accuracy as a headline metric. Always report missed dangerous events (FN), unnecessary manoeuvres (FP), recall, precision, F2, PR-AUC, and high-risk MSE on `eval_filter` events (plus unfiltered as a secondary table).
7. **Attribution:** Do not copy code from other repositories. Reimplement ideas in our own code and list every paper/repo in `CREDITS.md` with URL and licence.
8. **Assumptions:** Record every assumption or verified finding about column names, units, floor values, or thresholds in `reports/assumptions.md`. Never guess `(verify)` items — check the real data or page.
9. **After each phase:** Run `pytest` and all task-file gates, then write `reports/status/phase_N.md` before stopping.
10. **Order:** Follow Phases 0 to 10 strictly. Do not start anything from Section 8's post-v1 list before the Definition of Done is met.

## 2. Repo Conventions
- Python 3.11+ in a local virtual environment (`.venv`).
- All `src/` modules use type hints (`from __future__ import annotations`) and keep function signatures defined in the stubs.
- `data/raw/` is read-only; derived tables go in `data/processed/` as `.parquet`.
- `reports/figures/` stores generated plots (`.png` / `.html`).
- Status notes go in `reports/status/phase_N.md` (e.g., `phase_0.md` .. `phase_10.md`).
- Do not modify `tasks/phase_*.md` or `GOAL.md`.

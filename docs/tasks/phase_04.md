# Task: Phase 04 — Baseline and Evaluation Harness (3 to 4 h)

## Relevant Plan Section
> **Phase 4: Baseline and evaluation harness (3 to 4 h)**
> - **Baseline A (persistence):** predicted final risk equals the latest visible risk. Recommend MANOEUVRE if it is at or above the high-risk threshold.
> - `src/evaluate.py` computes, always on the evaluation-filtered events:
>   - **Regression:** MSE on log risk overall, and **MSE on high-risk events only**. Overall MSE is misleading because many events sit at the floor.
>   - **Classification:** confusion matrix, recall, precision, **F2**, **PR-AUC**, Brier score.
>   - **Official-style score:** I believe Kelvins scored a submission as high-risk MSE divided by F2, lower being better **(verify on the Scoring page)**. Reporting it lets us compare with the paper's baseline and the leaderboard.
>   - **Headline plot:** a threshold sweep with missed dangerous events (false negatives) against unnecessary manoeuvres (false positives). For models, sweep the probability threshold from 0.01 to 0.99. Baseline A has no probability, so sweep its risk threshold instead.
> - Also report the same table on unfiltered events as a secondary result.
> - **Done when:** baseline numbers and the first sweep plot are in `reports/results.md`.

## Files to Touch
- `src/models.py` (implement `predict_baseline_a`)
- `src/evaluate.py` (implement all evaluation, sweep, and plotting functions)
- `reports/assumptions.md` (record verified official-style score formula from Kelvins Scoring page)
- `reports/results.md` (populate Sections 1 & 2 with Baseline A filtered & unfiltered tables)
- `reports/figures/baseline_threshold_sweep.png`

## Inputs & Outputs
- **Inputs:** `visible_cdms`, `targets_df`, `folds.parquet` (evaluate on `split == 'train'` OOF/validation only — keep `test` locked!), `config.yaml`
- **Outputs:**
  - Working `src/evaluate.py` and `predict_baseline_a` in `src/models.py`
  - `reports/figures/baseline_threshold_sweep.png`
  - `reports/results.md` (Baseline A filtered + unfiltered tables)
  - `reports/status/phase_4.md`

## Instructions
1. **Verify** the Kelvins official-style score formula (`high_risk_MSE / F2` or any variant) on the Kelvins Scoring page / paper (`arxiv.org/abs/2008.03069`) and record it in `reports/assumptions.md`.
2. Implement `predict_baseline_a` in `src/models.py` using `thresholds.high_risk_log10` from `config.yaml`.
3. Implement `compute_regression_metrics`, `compute_classification_metrics`, `compute_official_style_score` (handle `f2 == 0` -> `inf`), `threshold_sweep`, `evaluate_predictions`, and `plot_threshold_sweep` in `src/evaluate.py`. Never report accuracy as a headline metric.
4. Evaluate Baseline A on `train` events (both `eval_filter` and unfiltered) — do NOT tune or evaluate on locked `test` set. Sweep Baseline A's risk threshold (e.g. `-10.0` to `-2.0`) and save `reports/figures/baseline_threshold_sweep.png`.
5. Write Baseline A filtered and unfiltered tables into `reports/results.md`.

## Acceptance Criteria
- `reports/results.md` contains Baseline A's FN, FP, recall, precision, F2, PR-AUC, high-risk MSE, overall MSE, and official-style score for both filtered and unfiltered events.
- `reports/figures/baseline_threshold_sweep.png` exists.
- `reports/assumptions.md` records the verified official-style score formula.
- `pytest` passes; no `read_csv` outside `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_4.md` (artifacts, Baseline A key numbers, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

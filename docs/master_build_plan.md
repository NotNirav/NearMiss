# SW5 Satellite Collision Risk Predictor: Master Build Plan (v3, Final)

StarForge Hackathon, Dark Side Software Track, Challenge SW-5 · Prototype v1 · Plan dated 18 Nov 2026

## 0. How to use this document

This is the single source of truth for the human team and for AI coding agents. It is self-contained.

- **Sections 1 to 3:** the problem, what the challenge requires, and the glossary with data rules.
- **Sections 4 to 6:** data sources, repository layout and configuration, and the output-file contract.
- **Section 7:** the build plan (Phases 0 to 10) with effort estimates. Total core effort is roughly 30 to 45 person-hours.
- **Section 8:** which prior-art ideas are in v1 and which wait for later.
- **Section 9:** dashboard specification.
- **Sections 10 to 13:** team split, risks, strict rules for AI agents, and the Definition of Done.

This is a **prototype**: the goal is a working, honestly evaluated first version that shows we are serious, built on the best known ideas. Items marked **(verify)** come from memory of public documentation and must be confirmed against the dataset before anyone relies on them.

## 1. Problem framing

Satellites and debris move at several kilometres per second. Operators receive **Conjunction Data Messages (CDMs)** warning of close approaches (**events**). New CDMs for the same event arrive every few hours as tracking improves, and the estimated collision risk changes with each one.

A thruster burn (**manoeuvre**) costs fuel, shortens mission life and disrupts operations, so operators must decide **at least 2 days before the time of closest approach (TCA)**, using incomplete early data.

**Objective.** Using **only** CDMs received at least 2 days before TCA for one `event_id`:

1. Predict the **final collision risk**, defined as the `risk` of the last CDM before TCA.
2. Output a decision, **MANOEUVRE** or **NO\_MANOEUVRE**, with a calibrated **probability** and a **confidence level** (HIGH, MEDIUM, LOW).
3. Show each event's risk history, the 2-day cutoff, the prediction, and the comparison with a baseline in a simple dashboard.

Common misunderstanding: this is not a form where you type circumstances and get yes/no. The input is the message history of one event.

## 2. Challenge deliverables checklist

| SW-5 requirement | Our deliverable | How we verify it |
| --- | --- | --- |
| Features from each event's message history | `src/features.py` | Leak test passes; about 40 to 80 documented features |
| Time-respecting split and class imbalance handled | `src/cutoff.py`, `src/split.py`, class weights, threshold tuning | Zero `event_id` overlap; nothing with `time_to_tca` below 2.0 days used |
| Comparison with a simple baseline: missed dangerous events versus unnecessary manoeuvres | `src/models.py`, `src/evaluate.py` | Threshold-sweep plot (false negatives versus false positives), F2, PR-AUC, official-style score |
| Recommendation and risk history per event | `app/streamlit_app.py` | Event inspector, risk chart with cutoff, confidence level, cost slider |

## 3. Glossary and data rules

| Term | Definition and project rule |
| --- | --- |
| **CDM** | One row of the dataset: one conjunction warning issued at a specific `time_to_tca`. |
| **Event (`event_id`)** | One close approach: a time-ordered sequence of CDMs. |
| **TCA** | Time of closest approach (`time_to_tca = 0`). |
| **`time_to_tca`** | Time remaining until TCA, in days **(verify unit and range, roughly 0 to 7)**. |
| **`risk`** | Estimated collision probability on a log10 scale: -6 means 1 in 1,000,000 **(verify)**. Many harmless events sit at a very low floor value, around -30 **(verify)**. |
| **Target** | `y_risk`: the `risk` of the event's last CDM (smallest `time_to_tca`). `y_high_risk`: 1 if `y_risk` is at or above the primary high-risk threshold (default -6.0), else 0. Also report evaluation metrics at the secondary operational threshold (-4.0). |
| **2-day cutoff** | Only CDMs with `time_to_tca` at or above 2.0 are visible to features and models. |
| **Training set rule** | Train on events with at least 1 CDM at or above the cutoff. Events with none cannot be predicted 2 days ahead: log the count and exclude them. |
| **Evaluation filter** | Validation and test results use the same filter the Kelvins test set used: at least 1 pre-cutoff CDM and the final CDM within about 1 day of TCA **(verify on the Kelvins data page)**. Without this, events with a stale final CDM make the persistence baseline look unrealistically good. Both numbers go in the config, and we also report unfiltered results as a secondary table. |
| **Class imbalance** | Dangerous events are rare. Accuracy is never a headline metric. |
| **Data leakage** | Letting a model see information it would not have 2 days ahead. The biggest way to fool ourselves. |

## 4. Data sources and schema check

### 4.1 Sources

- **Dataset (Zenodo open release, about 221 MB):** https://zenodo.org/records/4463683. Anonymised ESA CDMs from 2015 to 2019, 103 columns, with the training data, test data and the labels that were private during the competition.
- **Challenge pages (column definitions, rules, scoring):** https://kelvins.esa.int/collision-avoidance-challenge/
- **Paper:** Spacecraft Collision Avoidance Challenge: Design and Results of a Machine Learning Competition, https://arxiv.org/abs/2008.03069 (also in Astrodynamics: https://link.springer.com/article/10.1007/s42064-021-0101-5)
- **ESA workshop with top teams' talks:** https://indico.esa.int/event/385

### 4.2 Schema check (Phase 1, a notebook cell with asserts)

In the EDA notebook, assert and print:

1. Which files exist, and whether the test labels are included.
2. The range and unit of `time_to_tca`.
3. That the columns we use exist. Names to confirm **(verify)**: risk (`risk`, `max_risk_estimate`, `max_risk_scaling`), geometry (`miss_distance`, `relative_speed`, relative position and velocity in radial/transverse/normal axes), uncertainty (`t_sigma_*` and `c_sigma_*`), object info (`mission_id`, `c_object_type`, object size), and space weather (probably `F10`, `F3M`, `SSN`, `AP`).
4. The value of the risk floor, and how many events sit at it.

Write anything that differs from this plan into `reports/assumptions.md`.

## 5. Repository structure and configuration

```text
sw5-collision-risk/
  README.md
  CREDITS.md                 # every borrowed idea or snippet: link + licence
  config.yaml                # thresholds, cutoff, filters, seeds, paths
  requirements.txt
  .streamlit/
    config.toml              # dark theme settings (Section 9)
  data/
    raw/                     # Zenodo files, never edited
    processed/               # event tables, folds, predictions, history
  notebooks/
    01_eda_schema_audit.ipynb
    02_error_analysis.ipynb
  src/
    __init__.py
    load.py                  # read raw data (only module calling read_csv)
    cutoff.py                # leak-proof 2-day filter and target builder
    split.py                 # grouped and stratified splits
    features.py              # core + simple physics features
    models.py                # baseline, logistic regression, random forest, LightGBM
    calibrate.py             # probability calibration and confidence bands
    evaluate.py              # metrics, score, threshold sweep
    recommend.py             # writes the dashboard files
  app/
    streamlit_app.py
  tests/
    test_setup.py            # config and schema smoke tests
    test_no_leakage.py       # cutoff and group-split assertions
  reports/
    figures/
    status/                  # phase_N.md status notes
    assumptions.md
    results.md
    demo_script.md
```

```yaml
# config.yaml
project:
  random_seed: 42

data:
  raw_dir: data/raw
  processed_dir: data/processed
  cutoff_days: 2.0
  train_cutoffs: [2.0]            # optional augmentation, see Phase 6: [2.0, 3.0, 4.0]

eval_filter:                      # applied to validation and test results
  min_pre_cutoff_cdms: 1
  max_final_time_to_tca: 1.0      # verify against Kelvins data page

thresholds:
  high_risk_log10: -6.0           # primary
  operational_risk_log10: -4.0    # secondary, report as well
  decision_prob: 0.35             # placeholder; tuned on out-of-fold F2 / cost
  confidence_margin_high: 0.25    # |p - decision_prob| >= 0.25 -> HIGH
  confidence_margin_medium: 0.10  # 0.10 to 0.25 -> MEDIUM, below 0.10 -> LOW

validation:
  n_splits: 5
  holdout_fraction: 0.20          # only if no official labelled test set
```

## 6. Output files the dashboard reads

`src/recommend.py` writes two files, so the dashboard can be built in parallel against mock data from the first day.

**`data/processed/predictions.parquet`**: one row per event.

| Column | Meaning |
| --- | --- |
| `event_id`, `mission_id`, `object_type` | Identifiers |
| `cdm_count_pre_cutoff` | Visible messages at or above the cutoff |
| `latest_log_risk` | Last visible risk, which is also the baseline A prediction |
| `pred_final_log_risk` | Model's predicted final log10 risk |
| `high_risk_prob` | Calibrated probability of a high-risk final outcome |
| `recommendation` | MANOEUVRE or NO\_MANOEUVRE |
| `confidence` | HIGH, MEDIUM or LOW |
| `baseline_recommendation` | What baseline A would say |
| `actual_final_log_risk`, `actual_high_risk` | Ground truth, for the demo reveal toggle |
| `split` | train, val or test |

**`data/processed/cdm_history.parquet`**: one row per CDM, with `event_id`, `time_to_tca`, `log_risk`, `miss_distance`, and `visible` (true if at or above the cutoff).

**Phase 0 task:** generate a realistic mock of both files so UI work starts immediately.

## 7. Build plan

Effort is rough hours for one person. Phases 2, 3 and 4 decide whether the results can be trusted.

### Phase 0: Setup (1 to 2 h)

- Create the repo, virtual environment (Python 3.11+), `requirements.txt` (pandas, numpy, pyarrow, scikit-learn, lightgbm, matplotlib, plotly, streamlit, shap, pytest, pyyaml) and `config.yaml`.
- Fix random seeds. Generate the mock dashboard files.
- **Done when:** `pytest` runs on a placeholder test and the config loads.

### Phase 1: Data ingestion and exploration (3 to 4 h)

- Download the Zenodo files into `data/raw/`. Run the schema check from Section 4.2.
- In `01_eda_schema_audit.ipynb`, report: total events, events with at least 1 pre-cutoff CDM, CDMs per event, how many events are above -6 and above -4 (the imbalance), and how many sit at the risk floor.
- Plot about 12 events: steady safe, steady high-risk, and **late jumpers** (safe before the cutoff, dangerous after it).
- Write a one-page data dictionary of the columns we use.
- **Done when:** the schema check passes or every difference is in `assumptions.md`, and all (verify) items are resolved.

### Phase 2: Leak-proof cutoff and target (3 to 4 h) \[critical\]

- Implement `src/cutoff.py`:
  1. Group CDMs by `event_id`, sorted by `time_to_tca` descending.
  2. Build the target from the last CDM (smallest `time_to_tca`): `y_risk` and `y_high_risk`.
  3. Keep as inputs only CDMs with `time_to_tca` at or above the cutoff. Support a list of cutoffs for optional augmentation.
  4. Drop and log events with no pre-cutoff CDM.
- `tests/test_no_leakage.py`: change one CDM at `time_to_tca = 1.99` (for example set its risk to 0.0) and assert that every extracted feature is unchanged.
- **Done when:** the leak test passes.

### Phase 3: Splitting (2 to 3 h) \[critical\]

1. Check whether `event_id` or any column encodes chronology. If a clean chronology exists, train on earlier events and test on later ones.
2. Otherwise use the official test set (if its labels are in the archive) as the locked test set, and apply `StratifiedGroupKFold` with 5 splits on the training data, grouped by `event_id` and stratified by `y_high_risk`. If no labelled test set exists, hold out 20% of events the same way.
3. Assert that no `event_id` appears in two sets.
4. State in the write-up that our time-respecting rule is the 2-day cutoff inside each event plus the official train/test separation.

- **Done when:** fold assignments are saved to `data/processed/folds.parquet`.

### Phase 4: Baseline and evaluation harness (3 to 4 h)

- **Baseline A (persistence):** predicted final risk equals the latest visible risk. Recommend MANOEUVRE if it is at or above the high-risk threshold.
- `src/evaluate.py` computes, always on the evaluation-filtered events:
  - **Regression:** MSE on log risk overall, and **MSE on high-risk events only**. Overall MSE is misleading because many events sit at the floor.
  - **Classification:** confusion matrix, recall, precision, **F2**, **PR-AUC**, Brier score.
  - **Official-style score:** I believe Kelvins scored a submission as high-risk MSE divided by F2, lower being better **(verify on the Scoring page)**; return `inf` if `F2 == 0`. Reporting it lets us compare with the paper's baseline and the leaderboard.
  - **Headline plot:** a threshold sweep with missed dangerous events (false negatives) against unnecessary manoeuvres (false positives). For models, sweep the probability threshold from 0.01 to 0.99. Baseline A has no probability, so sweep its risk threshold instead.
- Also report the same table on unfiltered events as a secondary result.
- **Done when:** baseline numbers and the first sweep plot are in `reports/results.md`.

### Phase 5: Feature engineering (5 to 7 h)

All features use only visible CDMs, through `cutoff.py`.

- **Latest state:** `latest_log_risk`, `latest_max_risk_estimate`, `latest_max_risk_scaling`, `latest_miss_distance`, `latest_relative_speed`, `latest_time_to_tca`, `mission_id`, `object_type`, space-weather values.
- **Trajectory:** CDM count, first, min, max, mean and standard deviation of log risk, total change (latest minus first), last-step change, slope per day (linear fit against `time_to_tca`), and acceleration over the last 3 CDMs.
- **Geometry and uncertainty:** miss-distance change and slope, the primary and secondary position sigmas, and missing-value flags.
- **Simple physics pack** (own implementation of ideas common in the competition write-ups):
  - combined sigma per axis = sqrt(primary sigma squared + secondary sigma squared) for radial, transverse and normal axes;
  - normalised miss distance = sqrt of the sum over axes of (relative position / combined sigma) squared;
  - uncertainty volume = product of the three combined sigmas, and contraction = latest volume divided by first volume.
- Keep risk on the log scale and log-transform heavy-tailed positive columns.
- **Done when:** `features_base` and `features_physics` tables exist, are documented, and pass the leak test.

### Phase 6: Models and class imbalance (4 to 5 h)

The **classifier is the primary output**. The regressor is secondary.

- **Classifier** of `y_high_risk`: logistic regression (scaled), random forest, LightGBM.
- **Regressor** of `y_risk`: LightGBM predicting the final log risk directly, and a **residual variant** that predicts (final minus latest) and adds it back to the latest risk. Compare the two on high-risk MSE.
- **Imbalance strategies**, compared on identical folds: (1) unweighted with the decision threshold tuned on out-of-fold predictions for F2 or a cost measure; (2) class weights (`class_weight` or `scale_pos_weight`).
- **Optional augmentation (after the above works):** build training rows at several cutoffs (`train_cutoffs: [2.0, 3.0, 4.0]`), add the cutoff as a feature, and keep each event's rows in one fold. This multiplies the scarce dangerous examples. Always evaluate only at the 2-day cutoff.
- Tune a few hyperparameters with grouped cross-validation only. Never tune on the test set.
- **Done when:** a table in `results.md` compares all models and both imbalance strategies using out-of-fold predictions.

### Phase 7: Calibration and confidence (2 to 3 h)

- Fit **Platt scaling** (a logistic regression on the model's out-of-fold score) on the out-of-fold predictions. Do not use `CalibratedClassifierCV` for this, because it runs its own cross-validation around an estimator. Isotonic regression is optional, and with so few dangerous events it overfits easily.
- Produce a reliability diagram and the Brier score before and after.
- Confidence level from the margin between the calibrated probability and the decision threshold, using the cut-offs in `config.yaml`.
- Choose the decision threshold on out-of-fold data, and document the cost trade-off behind it.
- **Done when:** every event has a calibrated probability and a confidence level.

### Phase 8: Recommendation export (1 to 2 h)

- `src/recommend.py` writes `predictions.parquet` and `cdm_history.parquet` (Section 6) for all train, validation (e.g., fold 0 OOF predictions) and test events, with the `split` column filled in (`train`, `val`, `test`).

### Phase 9: Dashboard (4 to 6 h)

- Build the Streamlit app described in Section 9.

### Phase 10: Error analysis, write-up and demo (3 to 4 h)

- In `02_error_analysis.ipynb`, inspect the missed dangerous events and the worst false alarms, and add a global SHAP importance chart for the best model.
- Finish `results.md`, `CREDITS.md` and a two-minute demo script: the problem, one dangerous event end to end, one harmless event, the baseline-versus-model plot, and the limitations.

## 8. Prior-art ideas

**In v1** (cheap, and implemented in our own code, with the source noted in `CREDITS.md`):

1. **Persistence baseline and residual learning:** the last-CDM risk is the standard reference to beat. Predicting the change from it instead of the raw value is our own refinement, so credit it as such.
2. **Simple uncertainty-geometry features:** the physics pack in Phase 5.
3. **Multi-cutoff training rows:** our own idea for the scarce positives, optional in Phase 6.

**After v1, only if time remains:** a GRU or 1D-CNN sequence model, a rule-plus-model cascade, prediction intervals from quantile models, conformal prediction for confidence levels, per-event SHAP explanations in plain language, and a React/FastAPI front end.

Before writing any idea into `CREDITS.md` as coming from a paper or talk, read that source and confirm it.

## 9. Dashboard specification (Streamlit)

- **Theme:** dark, via `.streamlit/config.toml` at repo root (run `streamlit run app/streamlit_app.py` from repo root; background `#090D16`, secondary background `#111827`, primary `#38BDF8`). Status colours: MANOEUVRE red `#EF4444`, NO\_MANOEUVRE green `#10B981`.
- **Top bar:** title, the active threshold, and the dataset split selector (validation or test).
- **Tab 1, Event inspector:**
  - left: searchable event list, filterable by recommendation and confidence, sorted by danger probability;
  - right: a decision banner with the recommendation, calibrated probability, confidence level, predicted final risk, and the baseline's recommendation side by side;
  - a Plotly **risk trajectory**: x is time to TCA in days (7 down to 0), y is log10 risk, with visible CDM points, a dashed vertical line at 2.0 days (the **decision cutoff**), the region after it shaded as the **blind zone**, the predicted final risk as a marker, and a **Reveal ground truth** toggle for the demo;
  - a small chart of miss distance across the CDMs.
- **Tab 2, Threshold and cost simulator:** a slider for the decision threshold (or the cost ratio of a missed collision to a wasted manoeuvre) that updates the live point on the false-negative versus false-positive curve, the confusion matrix, and a line such as how many dangerous events are caught and how many unnecessary manoeuvres result compared with baseline A.
- **Tab 3, Results:** the model comparison table from `results.md` and the calibration plot, shown statically.

## 10. Team split

| Role | Owns | Delivers |
| --- | --- | --- |
| Data and cutoff lead | Phases 0 to 3 | `cutoff.py`, `split.py`, `folds.parquet`, leak test |
| Features | Phase 5 and the ideas in Section 8 | the two feature tables |
| Models and evaluation | Phases 4, 6, 7, 8 | `results.md`, `predictions.parquet`, `cdm_history.parquet` |
| Dashboard and demo | Phases 9 and 10 (demo part) | Streamlit app built on the Phase 0 mock files, demo script |

With fewer people, merge roles in this order: data with features, models with evaluation. The event-level table from Phase 2 and the two files in Section 6 are the interfaces between roles.

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Leakage from CDMs below 2.0 days | All data passes through `cutoff.py`; `test_no_leakage.py` must stay green. |
| Same event in train and test | Grouped splits with an assertion. |
| Evaluation on events the official test would exclude | Apply `eval_filter`; report the unfiltered numbers separately. |
| Few dangerous events, noisy results | Stratified grouped folds, report raw false-negative and false-positive counts, lock the test set until the end. |
| Overall MSE hides failures on dangerous events | Report high-risk MSE, F2, PR-AUC and the official-style score. |
| Probabilities badly calibrated with few positives | Platt scaling by default; check the reliability diagram. |
| Many events sit at the risk floor | Make the classifier primary; treat the regressor as secondary. |
| Dashboard blocked by model training | Mock files from Phase 0. |
| Scope creep | Nothing in Section 8's later list starts before the Definition of Done is met. |

## 12. Rules for AI coding agents

Follow these on every task:

1. **Cutoff isolation:** feature, model and evaluation code never reads raw CSVs directly. Data flows from `src/load.py` through `src/cutoff.py`, which enforces `time_to_tca` at or above the cutoff. Only `src/load.py` calls `read_csv`; only `load.py`, `cutoff.py` and the EDA notebook touch raw data.
2. **Group isolation:** never split rows randomly. Split by `event_id` and assert zero overlap.
3. **Test set quarantine:** never tune hyperparameters, thresholds or calibrators on the test set. Use out-of-fold predictions.
4. **No magic numbers:** read thresholds, cutoffs, filters, paths and seeds from `config.yaml`.
5. **Fit on training data only:** imputers, scalers, resamplers and calibrators are fitted inside the training or out-of-fold data.
6. **Metrics:** never report accuracy as a headline. Always report missed dangerous events, unnecessary manoeuvres, recall, precision, F2, PR-AUC and high-risk MSE, on the evaluation-filtered events.
7. **Attribution:** do not copy code from other repositories. Reimplement ideas in our own code and list every paper or repository in `CREDITS.md` with its URL and licence.
8. **Assumptions:** record every assumption about column names, units or thresholds in `reports/assumptions.md`.
9. **After each phase:** run `pytest`, then write a short status note with artifacts created, key numbers and open questions before moving on.
10. **Order:** follow Phases 0 to 10. Do not start anything from the later list in Section 8 before the Definition of Done is met.

## 13. Definition of Done

- [ ] Schema check passed or differences recorded; all (verify) items resolved
- [ ] `tests/test_no_leakage.py` passing (cutoff and group-split checks)
- [ ] Grouped, stratified 5-fold splits saved, with a locked test set
- [ ] Baseline A evaluated, with the false-negative versus false-positive sweep and the official-style score
- [ ] Logistic regression, random forest and LightGBM compared on identical folds, with both imbalance strategies
- [ ] Residual regression compared with direct regression on high-risk MSE
- [ ] Platt-calibrated probabilities and confidence levels for every event
- [ ] `predictions.parquet` and `cdm_history.parquet` written
- [ ] Streamlit dashboard with event inspector, cost slider and results tab
- [ ] `results.md`, `CREDITS.md`, `assumptions.md` and the two-minute demo script finished

# Task: Phase 01 — Data Ingestion and Exploration (3 to 4 h)

## Relevant Plan Section
> **Section 4.1 Sources:**
> - Dataset (Zenodo open release, about 221 MB): https://zenodo.org/records/4463683. Anonymised ESA CDMs from 2015 to 2019, 103 columns, with the training data, test data and the labels that were private during the competition.
> - Challenge pages (column definitions, rules, scoring): https://kelvins.esa.int/collision-avoidance-challenge/
>
> **Section 4.2 Schema check:**
> In the EDA notebook, assert and print:
> 1. Which files exist, and whether the test labels are included.
> 2. The range and unit of `time_to_tca`.
> 3. That the columns we use exist. Names to confirm **(verify)**: risk (`risk`, `max_risk_estimate`, `max_risk_scaling`), geometry (`miss_distance`, `relative_speed`, relative position and velocity in radial/transverse/normal axes), uncertainty (`t_sigma_*` and `c_sigma_*`), object info (`mission_id`, `c_object_type`, object size), and space weather (probably `F10`, `F3M`, `SSN`, `AP`).
> 4. The value of the risk floor, and how many events sit at it.
> Write anything that differs from this plan into `reports/assumptions.md`.
>
> **Phase 1: Data ingestion and exploration (3 to 4 h)**
> - Download the Zenodo files into `data/raw/`. Run the schema check from Section 4.2.
> - In `01_eda_schema_audit.ipynb`, report: total events, events with at least 1 pre-cutoff CDM, CDMs per event, how many events are above -6 and above -4 (the imbalance), and how many sit at the risk floor.
> - Plot about 12 events: steady safe, steady high-risk, and **late jumpers** (safe before the cutoff, dangerous after it).
> - Write a one-page data dictionary of the columns we use.
> - **Done when:** the schema check passes or every difference is in `assumptions.md`, and all (verify) items are resolved.

## Files to Touch
- `src/load.py` (implement `load_raw_cdms`)
- `notebooks/01_eda_schema_audit.ipynb`
- `reports/assumptions.md`
- `reports/figures/` (save event trajectory plots)
- `config.yaml` (only if `eval_filter.max_final_time_to_tca` needs updating after verifying Kelvins page)

## Inputs & Outputs
- **Inputs:** Zenodo files in `data/raw/` (read-only)
- **Outputs:**
  - Executed `notebooks/01_eda_schema_audit.ipynb`
  - Updated `reports/assumptions.md` with **all** `(verify)` items resolved against real data (never guessed)
  - Trajectory plot(s) of ~12 events in `reports/figures/`
  - `reports/status/phase_1.md`

## Instructions
1. Verify Zenodo dataset files are in `data/raw/` (if missing, download from https://zenodo.org/records/4463683 without modifying raw files).
2. Implement `load_raw_cdms(raw_dir, split)` in `src/load.py`.
3. In `notebooks/01_eda_schema_audit.ipynb`, run assertions for all 4 checks in Section 4.2.
4. **Verify against real data (do not guess)** and record in `reports/assumptions.md`:
   - Exact filenames in `data/raw/` and whether test labels are included.
   - Unit and min/max range of `time_to_tca`.
   - Exact column names for risk, geometry (RTN pos/vel), uncertainty (`t_sigma_*`, `c_sigma_*`), object info (`mission_id`, `c_object_type`, object size), and space weather (`F10`, `F3M`, `SSN`, `AP`).
   - Exact `risk` scale (log10?) and exact risk floor value + count of events at the floor.
   - `eval_filter.max_final_time_to_tca` from Kelvins data page / dataset inspection.
5. Report total events, events with >= 1 pre-cutoff CDM, CDMs per event, events >= -6 and >= -4, and floor count. Plot ~12 example events (steady safe, steady high-risk, late jumpers). Include a one-page data dictionary in the notebook or `reports/assumptions.md`.

## Acceptance Criteria
- Every `(verify)` item in `reports/assumptions.md` is checked and populated with observed values.
- `src/load.py` loads train (and test if present) cleanly.
- `pytest` passes and `read_csv` appears only in `src/load.py`.

## Automated Gates & Check Commands
```powershell
pytest -v
Select-String -Path src\*.py,app\*.py,tests\*.py -Pattern "read_csv" | Where-Object { $_.Path -notmatch "src\\load\.py$" }
```

## End of Task
Write `reports/status/phase_1.md` (artifacts, key numbers, open questions, pasted test output), update `tasks/STATUS.md` to `DONE`, and stop.

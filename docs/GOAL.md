# GOAL.md — SW5 Satellite Collision Risk Predictor

## Problem Framing
Satellites and debris move at several kilometres per second. Operators receive **Conjunction Data Messages (CDMs)** warning of close approaches (**events**, identified by `event_id`). New CDMs for the same event arrive every few hours as tracking improves, updating the estimated collision risk.

A thruster burn (**manoeuvre**) costs fuel, shortens mission life, and disrupts operations, so operators must decide **at least 2 days before the time of closest approach (TCA)** using incomplete early data.

**Objective.** Using **only** CDMs received at or above 2.0 days before TCA (`time_to_tca >= 2.0`) for one `event_id`:
1. Predict the **final collision risk**, defined as the `risk` of the last CDM before TCA.
2. Output a decision, **MANOEUVRE** or **NO_MANOEUVRE**, with a calibrated **probability** and a **confidence level** (`HIGH`, `MEDIUM`, `LOW`).
3. Show each event's risk history, the 2-day cutoff, the prediction, and the comparison with a persistence baseline in a simple Streamlit dashboard.

## Deliverables Checklist
| SW-5 Requirement | Our Deliverable | How We Verify It |
| --- | --- | --- |
| Features from each event's message history | `src/features.py` | Leak test passes; ~40 to 80 documented features |
| Time-respecting split and class imbalance handled | `src/cutoff.py`, `src/split.py`, class weights, threshold tuning | Zero `event_id` overlap; nothing with `time_to_tca < 2.0` days used |
| Comparison with simple baseline: missed dangerous events vs unnecessary manoeuvres | `src/models.py`, `src/evaluate.py` | Threshold-sweep plot (FN vs FP), F2, PR-AUC, official-style score |
| Recommendation and risk history per event | `app/streamlit_app.py` | Event inspector, risk chart with cutoff, confidence level, cost slider |

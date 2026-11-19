# Assumptions & Verified Schema Log

All items marked `(verify)` in the Master Build Plan have been verified against the official ESA Collision Avoidance Challenge dataset (Zenodo record 4463683).

---

## 1. Verified `(verify)` Items

### 1.1 Dataset Files & Test Labels
- **Archive Contents:** Downloaded and extracted from Zenodo record 4463683 (`Collision Avoidance Challenge - Dataset.zip`, 221 MB).
- **Files Present in `data/raw/Collision Avoidance Challenge - Dataset/kelvins_competition_data/`:**
  - `train_data.csv`: 162,634 rows, 103 columns, 13,154 unique conjunction events.
  - `test_data.csv`: 24,484 rows, 103 columns, 2,167 unique conjunction events (all visible messages with `time_to_tca >= 2.0`).
  - `test_data_private.csv`: 2,167 rows, 103 columns, exactly 1 row per test event containing ground truth labels (`true_risk`) at the final CDM (`time_to_tca <= 0.9772`).
  - **Confirmation:** The test set labels ARE included in the open Zenodo archive, enabling locked, unbiased out-of-sample testing.

### 1.2 `time_to_tca` Unit & Range
- **Unit:** Days remaining until Time of Closest Approach (TCA).
- **Train Range:** `min = -0.1443` days (post-TCA reporting), `max = 6.9907` days (~7 days).
- **Test (Public) Range:** `min = 2.0002` days, `max = 6.9932` days (pre-cutoff only).
- **Test (Private Final) Range:** `min = -0.0670` days, `max = 0.9772` days.
- **Evaluation Filter:** `eval_filter.max_final_time_to_tca: 1.0` in `config.yaml` matches the ESA competition rule where all evaluated test events have final CDMs within 1.0 day of TCA.

### 1.3 `risk` Scale, Floor Value & Class Imbalance
- **Scale:** $\log_{10}(\text{probability of collision})$. For example, `-6.0` corresponds to $10^{-6}$ (1 in 1,000,000 risk).
- **Risk Floor Value:** Exactly `-30.0` (representing numerical zero probability).
- **Rows at Floor in Train:** 67,240 rows (41.35% of all CDMs).
- **Events with Final Risk at Floor:** 8,349 events (63.47% of train events).
- **Imbalance Statistics (Train Events):**
  - Total events: 13,154.
  - Events with $\ge 1$ pre-cutoff CDM (`time_to_tca >= 2.0`): 11,942.
  - Events with no pre-cutoff CDM (dropped per Section 3 rule): 1,212.
  - Final risk $\ge -6.0$ (Primary High-Risk threshold): 365 events (~3.06% of valid events).
  - Final risk $\ge -4.0$ (Operational threshold): 13 events (~0.11% of valid events).
- **Test Set Imbalance:** 150 events $\ge -6.0$ (6.92%) and 17 events $\ge -4.0$ (0.78%).

### 1.4 Column Names Verification
All 103 columns were inspected and verified:
- **Risk Columns:** `risk` (in train and test), `true_risk` (in test_data_private), `max_risk_estimate`, `max_risk_scaling`.
- **Conjunction Geometry:** `miss_distance` (meters), `relative_speed` (m/s).
- **Relative State (RTN coordinates):**
  - Position: `relative_position_r`, `relative_position_t`, `relative_position_n` (meters).
  - Velocity: `relative_velocity_r`, `relative_velocity_t`, `relative_velocity_n` (m/s).
- **Uncertainty & Covariance:**
  - Standard deviations: `t_sigma_r`, `c_sigma_r`, `t_sigma_t`, `c_sigma_t`, `t_sigma_n`, `c_sigma_n`, plus velocity sigmas (`*_rdot`, `*_tdot`, `*_ndot`).
  - Covariance determinants: `t_position_covariance_det`, `c_position_covariance_det`.
- **Target & Chaser Object Metadata:**
  - `mission_id` (integer identifier of the target satellite).
  - `c_object_type` (categorical: DEBRIS, PAYLOAD, ROCKET BODY, UNKNOWN).
  - Radar Cross Section: `t_rcs_estimate`, `c_rcs_estimate`.
  - Area-to-mass ratios: `t_cd_area_over_mass`, `c_cd_area_over_mass`.
- **Space Weather Indices:** `F10` (solar flux), `F3M` (3-month average solar flux), `SSN` (sunspot number), `AP` (geomagnetic index).

---

## 2. Official-Style Score Verification
- **Formula:** $\text{Score} = \frac{\text{MSE}_{\text{high-risk}}}{F_2}$
- **Definition:** High-risk MSE is computed on events where the true final risk is $\ge -6.0$. $F_2$ score weights recall twice as heavily as precision. If $F_2 = 0$, the score is defined as $+\infty$. Lower is better.

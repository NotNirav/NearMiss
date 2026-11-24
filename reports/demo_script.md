# NearMiss (SW-5) — 2-Minute Operational Demonstration Script

**Objective:** Demonstrate autonomous satellite collision risk prediction under strict 48-hour operational cutoff constraints using real European Space Agency (ESA) Conjunction Data Messages.

---

### [00:00 - 00:30] 1. The 48-Hour Blind Zone & Live Orbital Simulator
- **Visual:** Open **Overview** (`http://localhost:8501`). Point to the top navigation bar and the live **Interactive T-48h Orbital Conjunction** canvas on the right.
- **Screen Action:**
  - Click **Simulate Avoidance Burn ($\Delta v$)** on the live orbital canvas. Watch the target satellite fire a thruster burn prior to T-48h, smoothly deflecting away from the crimson debris cone (`185 m` → `4,820 m`, status flips from `CONJUNCTION ALERT` to `ORBIT CLEARED`).
- **Presenter:**
  > "Satellite operators receive hundreds of Conjunction Data Messages every week, yet planning and uplinking an avoidance burn takes 24 to 48 hours. Once an encounter crosses **T-48h**, teams enter the **Blind Zone** where late radar updates arrive too late to act.
  >
  > Industry standard **Baseline A (Persistence)** assumes the last pre-cutoff risk stays flat—missing **28.8% of dangerous collisions** while triggering hundreds of costly false-alarm burns. NearMiss uses 73 orbital covariance physics features strictly from early passes to decide before the cutoff."

---

### [00:30 - 01:00] 2. Rescuing a Late-Jumping Collision (Blind Test Event `#873`)
- **Visual:** In the top navigation bar, switch the cohort toggle to **Blind Test (2,167)**, then click **Event Inspector**. Click the quick-jump button **Caught Collision #873** (one of 3 blind-test collisions rescued from Baseline A).
- **Screen Action:**
  - Verdict bar displays: `Command Avoidance Burn`, Calibrated Risk Odds: `22.43%`, `Medium Confidence`, while **Baseline A** says `Stand Down` (`Last visible: -6.01`).
  - Point to the **Collision Risk Trajectory**: 15 pre-cutoff passes sit right below the 1-in-1M alert line (`-6.0`) up to the dashed **T-48h Cutoff**.
  - Toggle **Reveal post-cutoff outcome (Blind Zone)**: banner flips to **`True Positive — Hazard Caught`**, and the actual terminal risk jumps across the threshold to **`-5.89`**.
- **Presenter:**
  > "Look at Event 873 from the blind ESA competition test set. At T-48h, the latest radar pass sat at `-6.01`—just below 1 in a million—so Baseline Persistence cleared it as safe.
  >
  > NearMiss read the tightening covariance geometry across 15 passes and ordered an avoidance burn early. When we reveal the blind zone, risk crossed directly into the hazard zone at `-5.89`. NearMiss caught a collision that legacy persistence missed."

---

### [01:00 - 01:25] 3. Preventing a Costly False-Alarm Burn (Blind Test Event `#560`)
- **Visual:** Click the quick-jump button **Saved Fuel Burn #560** in **Event Inspector**.
- **Screen Action:**
  - Verdict bar displays: `Stand Down — Nominal` (`5.02%`) vs **Baseline A**: `Order Burn` (`Last visible: -3.73`).
  - Toggle **Reveal post-cutoff outcome (Blind Zone)**: **`True Negative — Safe Clearance`**—actual risk plunges from `-3.73` all the way down to **`-26.17`** (matching NearMiss's `-20.38` hurdle forecast).
- **Presenter:**
  > "On the flip side, Event 560 had an early risk spike of `-3.73`, forcing Baseline A to command a fuel-draining burn. NearMiss recognized transient uncertainty and stood down. Inside the blind zone, actual risk collapsed to `-26.17`—saving propellant and mission uptime. Across the blind test set, NearMiss avoided 12 unnecessary burns, and cut false alarms by 22.6% out-of-fold."

---

### [01:25 - 01:45] 4. Mission Economics & Operational Threshold Simulator
- **Visual:** Click **Policy Simulator** in the top navigation bar (with **Blind Test (2,167)** active).
- **Screen Action:**
  - Switch **Spacecraft Profile** from `SmallSat` to `Earth Obs` ($p^* = 0.015$).
  - Highlight the KPI strip: Recall rises to **80.7%** (`121 of 150` hazards caught, `-6` fewer missed hazards than Baseline A), **$F_2 = 0.759$** (beating Baseline `0.739`), and expected financial balance reaches **`USD +5,145,000 (+12.0%)`** on the live FN/FP frontier and confusion matrices.
- **Presenter:**
  > "Operators don't need a one-size-fits-all cutoff. Because NearMiss outputs calibrated probabilities, flight teams can pick their spacecraft profile or tune $p^*$ directly on the FN-versus-FP trade-off frontier—beating Baseline A on both recall and $F_2$ when tuned for high-value Earth observation assets."

---

### [01:45 - 02:00] 5. Calibration Reliability, SHAP Physics & Limitations
- **Visual:** Click **Benchmarks** in the top navigation bar and scroll to the interactive **Platt Probability Calibration Reliability** and **Global Feature Drivers (Tree SHAP)** charts.
- **Presenter:**
  > "Platt scaling cuts Brier calibration error by **59.5%**, bringing predicted odds right onto the diagonal, and Tree SHAP confirms the model relies on physical trajectory and covariance envelope features (`latest_log_risk`, `latest_max_risk_estimate`, `latest_uncertainty_volume`) rather than spurious artifacts.
  >
  > **Limitation & Safeguard:** Extreme late jumpers with sparse early radar passes remain fundamentally uncertain—which is why NearMiss attaches explicit **High / Medium / Low Confidence** tiers so Low-confidence events immediately trigger additional radar tracking."

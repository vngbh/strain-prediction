**Timing matching fails the prespecified criterion. The first-window slope elevation survives stronger controls, but its attribution to sensor-network conditioning does not.**

I inspected both probes and ran additional read-only calculations with `.venv/bin/python -I -B`.

**1. Timing verdict and F2**

- \(H_\mathrm{match}\) fails: 1/4 aggregate bins meets the criterion. However, \(H_\mathrm{mid}\) also fails literally: no single offset wins every bin. The supported description is “240–480 s source calibration generally performed better.”
- “Source 0 never wins” applies to the aggregate table. It wins individual direction/bin cells, including early calibration windows for 33°C A_to_B and 43°C A_to_B.
- The 0.25 “chance” reference is not a calibrated null distribution. Offsets are not exchangeable, windows overlap, and directions share specimens/models. Treat 0.28 descriptively.
- These are **calibration-position bins**, not test-time bins. Every RMSE evaluates the entire opposite session. “Later target windows are better” can misleadingly suggest improved prediction late in deployment.

The F2-artifact claim needs narrowing. With source offset 480 s, four directions have positive position–RMSE Spearman correlations and four negative. At 43°C, A_to_B remains approximately −0.998 and B_to_A +0.999 under **every** source offset. Those opposing transfer trends survive the convention change.

Defensible conclusion: **F2’s association depends partly on source preprocessing; its original observations remain valid for the frozen 0 s convention.** Neither matrix identifies timing as the cause.

**2. First-window finding**

For fixed 125 s windows starting at 0/250/500/750 s, I compared the first slope with the median of the three later slopes:

| Estimator | First slope higher |
|---|---:|
| Original four-level calibration | 16/16 electrode/session combinations |
| Bands fitted at their actual mean strain | 16/16 |
| Plain OLS, \(R_i\sim1+\epsilon\) | 16/16 |
| Loading-only bands, actual mean strain | 15/16 |
| Unloading-only bands, actual mean strain | 16/16 |
| Equal-weight fixed strain bins, both branches pooled | 15/16 |

The loading-only-content explanation fails: the first window contains approximately one complete cycle, with 50–53% loading and 47–49% unloading.

A stronger control also survives: I aligned complete cycles, fitted equal-weight strain bins over **0.20–0.45% strain**, and balanced loading/unloading equally. First-cycle slopes exceeded the median of cycles 3–7 in **16/16**, by **2.5–13.7%**. This weakens explanations based solely on moving band endpoints, low-strain nonlinearity, or branch imbalance.

However:

- **The claimed universally faster initial decline is incorrect.** Original four-level slopes satisfy it in 15/16. For 33°C experiment 4, electrode 1, the first drop is 320 Ω/% versus 349 Ω/% for the later average. Under the cycle-aligned control, accelerated initial decline holds in 13/16.
- Nominal band coordinates introduce estimator dependence: resistance averages are assigned nominal strain coordinates, including zero, rather than their actual mean strain.
- Constant \(B\) cannot bias a free-intercept slope; changing \(B(t)\) can:
  \[
  \hat S=S+\operatorname{Cov}(\epsilon,B(t))/\operatorname{Var}(\epsilon).
  \]
  Adding linear time to window OLS reduces first-higher agreement to 12/16 and accelerated decline to 4/16. This is sensitivity evidence, **not proof of baseline drift**: time can also absorb hysteresis, particularly within individual ramps.
- The full-label two-electrode affine oracle also retains a scale shift: its effective inverse gain along a fixed full-session resistance–strain direction is higher initially in 8/8 sessions, by 2.0–24.4%. It has no uniquely defined electrode sensitivity, so individual coefficients should not be interpreted mechanistically.

**3. Mechanism, literature, thesis wording**

Small temperature changes weaken an instantaneous temperature explanation. They do not exclude thermal-history-dependent annealing at nearly constant temperature.

Host-composite conditioning, changing local strain transfer, and reference-gauge/adhesive settling remain unresolved. Pure host softening would not explain changing resistance versus accurately measured identical local strain, but changing transfer between the embedded sensor and surface gauge could. Calling this “Mullins softening” specifically is unsupported. Both electrodes share these confounders; 16/16 is not 16 independent confirmations.

First-cycle sensitivity overshoot is already reported: Montazerian et al. describe a high first-cycle gauge factor followed by a sharp decrease in carbon-fiber/PDMS yarns. [2019 primary paper](https://doi.org/10.1016/j.compositesb.2019.01.090). Mu et al. explicitly quantify cycling sensitivity attenuation in CNT/silicone and carbon-black/silicone composites, with composition-dependent behavior. [2023 primary paper](https://doi.org/10.3390/polym15163379).

For thermally drawn fibers, Leber et al. report mechanical hysteresis and strain softening; this does not establish the identical electrical overshoot. [2023 primary paper](https://doi.org/10.1002/advs.202207573). The dataset’s source paper already reports branch-dependent slopes and low-strain nonlinearity. [Caglar et al.](https://doi.org/10.1002/admt.202000957).

Thesis-worthy as a **descriptive calibration finding**:

> “Across eight sessions from one embedded sensor/specimen, initial-cycle apparent resistance–reference-strain slopes exceeded later-cycle slopes under several estimators, including branch-balanced fitting over a common strain interval.”

“Conductive-network conditioning” is a compatible hypothesis. “After-rest recovery,” sensor-specific mechanism, explanation of history-feature gains, and generalization beyond this specimen remain unsupported.
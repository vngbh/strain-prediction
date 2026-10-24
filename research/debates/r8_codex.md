**Verdict: the numerical decision is correct; exact pre-registration compliance is incomplete.**

1. **Freeze and implementation.** Commit `303ccdb` (2026-10-20 21:13:48 +0900) added both files. `git diff 303ccdb -- research/probes/probe_preconditioning.py` is empty; the pre-registration and imported `src/strain` modules are also unchanged. Both result CSVs are untracked and absent from that commit. This supports the stated sequence, but Git cannot prove that no earlier execution occurred.

   The [script](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_preconditioning.py:40) correctly implements the 250-second prefix, specified peak/trough detection, three failure conditions, P0 without detection, full-window calibration, source/target rules, frozen LOTO fitting, and conjunctive decision rule. Failures raise exceptions without repair. I reran the existing leakage gate successfully, verified P0 with a detection trap, and matched all 800 calibration-content values against raw-session recomputation; `c1` spans 121.1–124.2 seconds.

   Two discrepancies prevent “exactly”:
   
   - The [registered leakage check](/Users/vngbh/Documents/Lab/strain-prediction/research/directions/2026-10-21-preregistration-preconditioning.md:99) requires identical **predictions**. The implemented gate checks boundaries, B/S parameters, and fusion weights only. Predictions follow from unchanged parameters and label-free feature construction, but that required executable comparison is absent.
   - P0d has no explicit implementation/output rows. Its prediction equivalence to P0 is stipulated by construction, so this omission does not change the decision, but reporting is incomplete.

2. **Independent CSV calculation.** Using a separate standard-library calculation on the [RMSE CSV](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/out/preconditioning_rmse.csv):

   | Comparator | Mean RMSE threshold | Mean directional improvement | Wins | Worst deterioration |
   |---|---:|---:|---:|---:|
   | P0 | 0.04598303 | 4.68457% | 5/8 | +29.99923% |
   | P2 | 0.04567481 | 4.84378% | 7/8 | +11.34270% |

   P1 mean RMSE is **0.04610249**. All three conditions fail against P0; the mean-RMSE and improvement conditions fail against P2. **“Decision criterion unmet” follows exactly.** Physics mean RMSE increases from 0.08814768 to 0.08858383; “did not improve” should explicitly refer to the mean.

3. **Wording.** The quoted numbers are accurate. “Is real” overstates what the descriptive analysis establishes. Existing estimator outputs show elevated apparent slopes in 16/16 electrode-session pairs under OLS and branch-balanced bins, but only 12/16 with a time term.

   Replace the final sentences with:

   > Mean physics-only RMSE increased slightly. Higher first-cycle apparent resistance–reference-strain slopes were observed under several estimators in this corpus. This P1 calibration convention did not satisfy the pre-registered adoption criterion; the analysis does not establish a conditioning mechanism or generalization beyond this specimen and loading program.
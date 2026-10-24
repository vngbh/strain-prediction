I reproduced all 813 window results in memory, matching the CSV within numerical precision, and reran F4. No repository files were changed.

1. **F1 supports window-position sensitivity, but “robust dynamic gain” overstates the comparison.**

   The counts are correct: seven folds improve at every sampled window; 33°C B→A improves at 98/109 windows, with a minimum improvement of −4.0%. No scheduled windows were skipped. [Results](</Users/vngbh/Documents/Lab/strain-prediction/research/probes/out/window_distribution.csv>)

   Adjacent windows overlap by 92%. That does not manufacture their pointwise advantages, but 813 windows cannot establish 813 independent successes. Thinning to starts 130 s apart preserves the pattern: seven folds improve throughout; the weak fold improves at 8/9 retained windows. Those still share calibration and test sessions.

   The weak fold exposes a denominator problem: its default HGB RMSE is **0.0383% strain**, versus a window-distribution median of **0.0505%**. Physics deteriorates from **0.0385%** to median **0.0659%**. Thus larger median percentage improvement coexists with worse absolute HGB prediction.

   Source sessions remain self-calibrated at their own prefixes. This legitimately tests the frozen model’s tolerance to target calibration changes under that training convention; it does not test matched source/target calibration shifts. There is no demonstrated inflation from this asymmetry, but its contribution remains unresolved. Also, HGB-versus-physics does not isolate the contribution of dynamics: that requires a static-HGB window sweep.

   Thesis wording: **“Under nominal-level calibration, the frozen dynamic HGB outperformed matched physics across all sampled calibration positions in seven directions and 98/109 positions in the eighth. These dependent sensitivity evaluations demonstrate a broadly persistent relative advantage within this corpus, without establishing improved absolute accuracy at later positions.”**

2. **F2 establishes associations, not an explanation or calibration-selection rule.**

   The correlations reproduce. However, the reported approximately ±0.43 correlations with time are pooled. Within folds, time–coverage correlations reach −0.995 and time–low-band-mean correlations reach +0.990. Pooling understates the confounding relevant to within-session interpretation.

   At 43°C, time–HGB-RMSE correlations reverse between directions: **−0.998 versus +0.999**. That contradicts a universal “later calibration transfers worse” rule. Coverage likewise associates with error in opposite directions across folds.

   Coverage also changes the nominal calibration reference levels because those levels depend on each window’s minimum and maximum. It is therefore entangled with the calibration construction itself.

   These reversals meet the direction document’s criterion for abandoning a general predictive content rule.

   Thesis wording: **“Later calibration windows were associated with higher HGB RMSE in six directions, lower RMSE in one, and little monotonic association in one. Strong within-session dependence among window position and content descriptors prevents causal attribution; no general calibration-selection rule was established.”**

3. **F3 is a legitimate constrained optimum, not evidence of an implementation bug.**

   All 813 windows select zero electrode-1 weight. I additionally checked all eight original prefixes: their *unconstrained* least-squares fusion optima range from approximately **−2.36 to −1.22**. Thus zero is the continuous constrained optimum at those prefixes, rather than merely a coarse-grid accident. [Selector](</Users/vngbh/Documents/Lab/strain-prediction/src/strain/calibration.py:46>)

   But selection minimizes error on the same four nominal calibration aggregates used for fitting. It does not establish that electrode 2 is optimal for opposite-session transfer.

   REPORT’s algorithm description is technically correct but omits the realized endpoint. The physics baseline is operationally electrode 2 alone. HGB still receives electrode 1 and its derived temporal features; claims that *fusion itself* contributes predictive value are unsupported.

   Thesis wording: **“The constrained calibration-point selector assigned zero weight to electrode 1 throughout the evaluated windows, making the fused physics estimate identical to electrode 2. Electrode 1 remained available separately to HGB through its normalized signal and temporal features.”**

4. **F4 rejects one elapsed-time representation, not the loading-program explanation.**

   All reported numbers reproduce; static-plus-time improves only **5/8** directions. The hook supplies elapsed time consistently during training and evaluation. [Probe](</Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_memory_vs_time.py:28>)

   Absolute time does not explicitly represent cycle phase, branch, or cycle count. Under the fixed HGB capacity, learning repeated cycles from absolute time can be harder than reading phase information already exposed by derivatives and rolling extrema. Failure of this representation cannot exclude program tracking.

   Thesis wording: **“Adding elapsed session time to static normalized inputs did not recover the dynamic model’s performance under the tested HGB configuration. This supports the usefulness of the temporal representation, while leaving loading-program tracking and sensor-state mechanisms unresolved.”**

The **single next analysis** should be the already-planned, source-derived cycle-template control: keep static normalized inputs and the same HGB, replace detailed history with explicit program phase, branch, and cycle count. Freeze its construction using source sessions; target phase tracking must use causal resistance observations, never test strain or centered peak detection.

Prespecify “matches” as mean RMSE within 5% of dynamic HGB and per-direction RMSE within 10% in at least seven directions. If met, **kill the claim that detailed history demonstrates transferable sensor memory**. Failure would reject that particular program control, without proving a sensor-memory mechanism.
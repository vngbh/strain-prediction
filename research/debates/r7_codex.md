Do not freeze the draft unchanged. No proposed analysis was run; no files were changed.

1. **P2 answers an operational comparison, not a causal mechanism.** P1 versus P2 compares discarding initial calibration data with pooling all available data during the same event. It also changes sample count, level locations, and branch weighting.

   Add: “P1–P2 estimates the effect of retaining only the later window under this calibration estimator; it does not isolate physical conditioning.”

   Add a duration-matched P0 control: calibrate on `[0,125]`, continue the event until P1 finishes, then deploy. With unchanged opposite-session evaluation, its predictions equal P0; no extra model run is needed. P1 versus this control holds calibration-window duration and event duration constant. Separating cycling from elapsed time would require a matched rest/no-load condition absent from this corpus. Keep P1t: P1 also changes source preprocessing and therefore the fitted HGB.

2. **c1 exists in all nine sessions, but the draft’s timing estimate is wrong.** Direct `find_peaks` checks gave first peaks at 62.2–62.4 s and these first following troughs:

   | Temperature | c1, sorted sessions (s) |
   |---|---|
   | 21°C, diagnostic only | 122.7 |
   | 33°C | 121.1, 124.2 |
   | 43°C | 122.8, 124.0 |
   | 52°C | 123.2, 124.1 |
   | 62°C | 122.8, 122.4 |

   Initial strain is 0.0011–0.0044%, followed by rising strain; none appears to start mid-cycle. State explicitly that only eight sessions enter evaluation.

   All four bands are populated in both P0 and P1, but coverage is not identical. P1 band counts range from 42 to 198, with changed branch balance. For 62°C B, minimum strain changes from 0.0022% to 0.0084%, while the lowest calibration point remains nominally zero. Add mandatory reporting of window extrema, level locations, band counts, and branch counts; do not call calibration content “matched.” Replace “~110–120 s” with the observed 121.1–124.2 s.

3. **Retain absolute RMSE, but add a conjunctive relative criterion.** In the [existing baseline table](/Users/vngbh/Documents/Lab/strain-prediction/results/tables/frozen_protocol_ablations_per_fold.csv), 43°C contributes 41.8% of summed HGB RMSE; A_to_B alone contributes 28.8%. That is substantial influence, not majority domination.

   Define, for comparator \(Q\),
   \[
   I_Q=\frac{100}{8}\sum_d\left(1-\frac{R_{P1,d}}{R_{Q,d}}\right).
   \]
   Require **both** existing 5% mean-RMSE reductions **and** \(I_{P0}\ge5\%\), \(I_{P2}\ge5\%\). Require wins in ≥6/8 directions against **each** comparator. Treat thresholds as practical conventions without power or significance claims; directions are dependent. Report worst relative deterioration too. Remove “as in the main result”: the repository’s primary metric is mean per-direction improvement over physics.

   Replace the failed-P2 interpretation with: “No demonstrated advantage from exclusion over pooling.” Failure to beat P2 does **not** establish that longer calibration caused the benefit. Define every other failed conjunction as “decision criterion unmet.”

4. **Freeze the selection limitation now.** Replace “recommend preconditioning” with “support this candidate calibration convention within the observed corpus.”

   Add: “This is a prespecified reanalysis following corpus-informed hypothesis generation and extensive calibration-window exploration. Freezing prevents further outcome-dependent changes; it does not create untouched confirmation or an unbiased generalization estimate.” Retain the parent note’s mechanism exclusions and permanent specimen/loading-program boundaries.

5. **Whole-session detection violates the declared label budget.** Prominence can require later samples ([SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html)). The [existing probe](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_preconditioning.py:21) searches 300 s, exceeding P1’s 246.1–249.2 s event.

   Concrete repair: fix P1/P2 event completion at **250 s**, detect peaks/troughs only within that prefix, retain P1’s `[c1,c1+125]`, and use P2 `[0,250]`. Predefine missing detections or `c1+125>250` as protocol failures. P0 must never invoke cycle detection. Before scoring, require mutations of post-event labels to leave selection, calibration, and predictions unchanged.
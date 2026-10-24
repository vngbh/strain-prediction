Yes: I accept the corrected model-specific wording, the explicitly post-hoc ρ=0 comparison, and withdrawal of both the “little independent information” claim and E6/fusion link.

**Do not freeze yet.** The present decision rule cannot distinguish physical delay from hysteresis.

Read-only checks: dataset1–4 contain eight sessions with finite required inputs, uniform 0.1 s spacing, and median absolute interior strain rates of 0.00879–0.00951 %/s. Syntax parses. No files changed; dataset0 was untouched; the probe was neither imported nor executed; no lag/collapse outcomes were computed.

1. **OLS and collapse are confounded.** In the [draft](/Users/vngbh/Documents/Lab/strain-prediction/research/directions/2026-10-21-preregistration-reference-lag.md:33),
   \[
   b\epsilon(t-\tau)\approx b\epsilon(t)-b\tau\dot\epsilon(t).
   \]
   Consequently, branch-dependent resistance offsets correlated with strain rate bias τ*. On approximately constant-speed ramps, a rate-independent loading/unloading offset can mimic this derivative term, and shifting can reduce its loop area. Adding linear time does not remove that ambiguity. **C≤0.70 is therefore not a discriminator.**

   Rename H_lag “consistent apparent alignment,” with mechanism unresolved. Better falsification checks: unequal branch slopes contradict the simple linear pure-delay model, but equal slopes cannot establish it; test whether supported harmonics share one delay and constant gain, using the [phase-delay relationship](https://www.mathworks.com/help/signal/ug/delay.html). Across genuinely different existing ramp rates, pure-delay branch separation scales approximately with rate, whereas rate-independent separation does not. Prespecify support/tolerances; insufficient excitation means unidentified. Fit τ on alternating complete cycles and evaluate collapse on the others to reduce optimization bias; this still cannot establish mechanism.

2. **Thresholds need engineering justification and uncertainty.** Three samples is a search-grid distance, not demonstrated estimation accuracy. At the stipulated 0.009 %/s, the 0.3 s threshold corresponds locally to 0.0027% strain, or 27 microstrain; explain why that defines “material.” The 0.2–0.3 s inconclusive band is coherent, but H_null should require cycle-based uncertainty contained within ±0.2 s, rather than point estimates alone.

   Treat 14/16 as a consistency requirement, without binomial significance: electrodes share sessions. C=0.70 is an arbitrary engineering cutoff. The trailing 21-sample smoother is irrelevant to this raw-signal probe; it does not delay every HGB input, since raw normalized signals also enter.

3. **Remove mechanistic secondary labels.** Eight sessions at four temperatures cannot separate temperature from session/order effects. A positive Spearman threshold does not identify viscoelasticity; a small correlation plus low SD does not identify acquisition offset. Report temperature association and dispersion descriptively. Replace “common lag/electrode-specific” with “apparent-lag agreement/disagreement”; a median conceals individual disagreements and differing uncertainty. Comparing lag-only RMS with HGB error cannot establish an error floor or an attributable fraction of HGB error.

4. **Concrete code corrections.** In the [probe](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_reference_lag.py:24):

   `shifted()` has the stated sign. The parabola formula is correct, but flat/tied minima need unresolved status. Compare candidates on fixed support; use fractional interpolation consistently downstream instead of rounding refined τ for area and RMS.

   At [loop_area](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_reference_lag.py:52), require identical complete reference cycles and complete branch-bin coverage at both shifts. Missing bins currently get bridged; no accepted cycles returns zero. Return invalid status instead. Split branches at the cycle maximum; `~loading` includes flat samples. Refitting slope and drift changes C independently of alignment: report a fixed-nuisance comparison, with refitting as sensitivity. Only linear drift is removed.

   At line 86, unresolved τ incorrectly produces **zero** lag-only RMS: return NaN. Using absolute lag is otherwise valid for this full-overlap squared-difference calculation.

   At [decision counting](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_reference_lag.py:99), replace separate magnitude/sign counts with `max(count(tau>=0.3), count(tau<=-0.3))`, requiring valid paired lag/area results. Declare minimum coverage; prevent silently dropped NaNs from determining the collapse median or secondary summaries.
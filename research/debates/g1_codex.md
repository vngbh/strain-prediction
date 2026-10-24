**REVISE before freezing.** Read-only review completed; no probe execution, outcome computation, or dataset0 access.

1. **F measures gap contraction across changed tasks, not an attributable fraction.** The [pre-registration](/Users/vngbh/Documents/Lab/strain-prediction/research/directions/2026-10-21-preregistration-dynamic-gain-attribution.md:46) changes labels, calibration-window membership, B/S, fusion, normalized inputs, training targets, and scoring targets. Within-label static–dynamic comparisons remain interpretable; dividing their gaps does not isolate alignment’s contribution.

   Assuming \(G_\text{original}>0\):
   - \(F=1\): aligned static and dynamic models tie.
   - \(F>1\): static wins after alignment.
   - \(F<0\): the dynamic advantage increases.
   
   Do not clip F. Rename it **“relative contraction of the mean static–dynamic RMSE gap.”** Replace “absorption” with “gap contraction.” Add a sensitivity retaining original B/S and fusion while shifting training/scoring labels and retraining both readouts. This separates normalization changes from the broader intervention; it still changes the prediction task.

   Also state that F uses absolute RMSE gaps, whereas REPORT’s 37.6% versus 5.1% uses mean per-direction improvements over physics. These are different estimands.

2. **Oracle selection favors static fit; upward bias is plausible, not guaranteed.** The [v2 fitting objective](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_reference_lag_v2.py:48) selects tau by minimizing residuals of `R ~ 1 + eps(t-tau) + t` on the same session. That explicitly selects a more statically predictable reference relationship. It does not optimize the HGB gap directly, so oracle F is neither unbiased attribution nor a guaranteed upper bound.

   Require a cycle-cross-fitted oracle sensitivity: estimate tau on odd cycles, score even cycles, then swap. Existing cross-fitted loop-area results do not cross-fit this outcome. Disclose the shared-corpus, data-informed nature throughout.

3. **Require both alignment variants, with separate conclusions.** Source-constant excludes the held-out temperature’s offsets and tests transfer of one common correction. Oracle tests session-specific apparent alignment. Their difference combines target-informed selection and real offset heterogeneity.

   Given the documented 1.8–9.1 s spread, especially experiment 10, failure of source-constant cannot refute session-specific alignment’s relevance. Conversely, oracle success cannot establish transferable correction. Keep oracle primary only for the explicitly diagnostic question; make source-constant primary for any deployment claim. Require reporting both verdicts and label disagreement **“oracle-specific contraction; transfer unsupported.”** Even source-constant predicts an earlier-strain target.

4. **Keep ratio of means; repair the decision rule.** The script already computes \(1-\overline G_\text{aligned}/\overline G_\text{original}\), avoiding unstable per-direction ratios. Keep that. Report every absolute \(G_d\) and change \(\Delta G_d\); do not average directional fractions.

   The 0.5/0.2 thresholds are conventions without an engineering justification. Six directions provide descriptive consistency, not replication. Add:
   - A predeclared positive denominator floor; otherwise “ratio uninterpretable.”
   - \(F<0\): “gap expansion.”
   - \(F\ge0.5\) but fewer than six decreases: “heterogeneous,” not “partial.”
   - Leave-one-temperature-pair-out summaries to expose dominance by one group.

5. **Code findings and required guards.** In [realign()](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_dynamic_gain_attribution.py:29), `t-tau` has the registered sign. `np.interp` holds the first label before available support; it also holds the last for negative tau. Sorting occurs after interpolation: assert finite, strictly increasing timestamps beforehand.

   Endpoint holding manufactures labels inside calibration. Prespecify a common-support sensitivity excluding extrapolated rows from training, calibration fitting, and scoring in every arm. Preserve preceding resistance history and the original 125 s cutoff; trimming must not restart the calibration clock.

   [evaluate_fold()](/Users/vngbh/Documents/Lab/strain-prediction/src/strain/protocol.py:79) does recompute shifted-label calibration and physics. Models are retrained per label set, then frozen across target directions.

   No inevitable pass/fail bug is apparent. However, grouped means can conceal a missing electrode tau, and NaN F falls through to “partial.” Require exactly two finite electrode offsets per expected session, complete finite fold rows, matching physics scores across variants, and explicit final-variant selection instead of `VARIANTS[0]`.
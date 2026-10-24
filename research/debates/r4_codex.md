All G1–G4 numerical summaries reproduce. I reran both probes with `.venv/bin/python -I`; the rebuilt frame matches [calibration_shift.csv](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/out/calibration_shift.csv) within numerical tolerance. All RMSE values below are **percent strain**.

1. **G1 supports the dynamic representation under this HGB configuration.** Dynamic beats static at every sampled window in seven directions and 96/109 windows (88.07%) in 33°C B→A. The mean of direction-specific median RMSEs is physics 0.087878, static 0.071685, dynamic 0.049559.

   This establishes a persistent within-corpus advantage over the tested static HGB. It does not identify sensor memory as the mechanism or exclude loading-program tracking.

2. **G2 supports lower error across positions, but “reduces sensitivity” is unsupported.** Diversity lowers the mean of direction-specific medians from 0.049559 to 0.044847: **9.51%**. The reported p90 and maximum summaries reproduce, but both average direction-specific statistics. The worst individual RMSE is 0.125795, not 0.058611.

   Direct measures of variation across calibration positions do not improve overall:

   | Mean across directions | Original | Diversity |
   |---|---:|---:|
   | Within-direction standard deviation | 0.007546 | 0.007551 |
   | Within-direction IQR | 0.012497 | 0.012620 |

   Standard deviation and range increase in three directions. In 33°C A→B, standard deviation rises from 0.001281 to 0.004019.

   The exception is material: **43°C A→B median RMSE worsens from 0.064516 to 0.067859 (+5.18%)**, with diversity winning only 9/101 windows. At 0 s, RMSE worsens from 0.111605 to 0.117397.

   The “seven directions” count also hides a marginal window majority: 33°C A→B wins only 51/98 windows. At 0 s, diversity improves only five directions. Nevertheless, the aggregate gain is not driven entirely by one temperature: omitting each temperature group in turn leaves reductions of 7.0–14.8%.

3. **G3 rejects this exact duplication control, not every alternative explanation.** Fourfold replication reproduces 0.0498 versus diversity’s 0.0448. Thus, adding identical rows does not reproduce the gain.

   However, this does not distinguish diversity from simply choosing a better **single source calibration offset**. Also, HGB’s default random-row validation can place replicated rows or augmented versions of the same observation on both sides of its internal split. That is not target leakage, but iteration counts and early-stopping behavior should be audited.

4. **G4 exposes the missing comparator.** Across the 32 matched-offset evaluations, means reproduce as 0.048642 original, 0.045274 matched, and 0.045243 diverse. The matched–diverse gap is only 0.000030; this does not establish equivalence or superiority.

   Matching source calibration time alone recovers nearly the same aggregate performance. The current experiment therefore has not shown that stacking several calibrations is necessary. Matching clock offsets also does not establish matching physical sensor states.

Code inspection of [the shift probe](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_calibration_shift.py) found the held temperature excluded from fitting. Shared relative offsets do not share observations across source and target sessions. All 32 source windows contain 1,251 samples spanning 125 s; the shortest session lasts 979.1 s, so 720–845 s fits completely. All 813 target calibration windows have complete duration, occupied calibration bands, and finite, nonzero slopes.

Diversity accesses later source labels for normalization, but the original model already uses those labels as training targets. This changes preprocessing supervision, not the available source-label set. Target calibration remains 125 s per evaluated scenario.

**Selection risk remains:** the idea and offsets followed F1/F2 inspection on this corpus. Fixing settings before this run prevents tuning during this run; it does not make the analysis prospectively independent. Neither 813 overlapping windows nor eight directions constitute independent replication.

The minimal additional check is a **source-only selected single-offset comparator**. Inside each outer held-temperature fold, use inner validation restricted to source temperatures to choose among the four single offsets. Freeze that choice, then compare its fourfold-replicated model against diversity across all outer target windows. Use a fixed training-iteration policy for both. Prespecify the mean of direction-specific medians as primary.

**Kill criterion:** if diversity does not lower that primary metric versus the selected, row-matched single-offset comparator, drop the diversity-specific advantage claim. Any sensitivity claim additionally requires a prespecified spread metric to decrease. This remains internal validation after exploratory selection.

Use this thesis wording:

> “In an exploratory analysis of this specimen, source calibration-diversity training lowered median RMSE across sampled target calibration positions in seven of eight directions. The mean of these medians fell from 0.04956% to 0.04485% strain; fourfold replication without diversity yielded 0.0498%. One direction worsened. These dependent evaluations support improved accuracy over sampled positions, not uniformly reduced sensitivity or external generalization; the method and offsets were selected after inspection of the same corpus.”
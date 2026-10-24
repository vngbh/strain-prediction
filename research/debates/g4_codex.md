HEAD is `e8a4e616903ab58ea15ddb4f46160b805abd2d7f`. The probe and preregistration are byte-identical to that commit; tracked source is unchanged. Outputs remain untracked. This audit recomputed CSV arithmetic without rerunning model fitting; it does not independently authenticate execution provenance. No files changed; dataset0 remained sealed.

All reported verdicts match the frozen rule:

| Arm | Recomputed F | Smaller gap | Verdict |
|---|---:|---:|---|
| A1 oracle | 0.234244 | 7/8 | Partial contraction |
| A2 source-constant | 0.378060 | 8/8 | Partial contraction |
| A3 fixed normalization | 0.379953 | 7/8 | Partial contraction |
| A4 cross-fit | 0.206084 | 6/8 | Partial contraction |

Every gap and Δgap reconstructs exactly from the per-fold CSV. A0 RMSE matches repository values exactly; paired physics RMSE agrees exactly. Reference mean gaps exceed the 0.005 floor: 0.0241622, or 0.0240438 for A4’s matched subsets. Leave-one-group-out ranges reproduce: A1 **0.099372–0.405224**; A2 **0.268252–0.521355**. All quoted A0→A1 mean RMSEs reproduce.

Claude’s interpretation overreaches in two places:

- **“Not mainly explained” is unsupported.** F measures contraction under a changed prediction task, not an attributable fraction of the original advantage. Targets, training, scored rows, and usually normalization change. Offsets were selected using this corpus and a static-fit objective. F<0.5 licenses “over half the reference mean gap remains under this intervention.” It provides no causal attribution bound. Excluding 52°C also raises A2 above 0.5.
- **“Dynamic insensitive” is unsupported.** Different fitted models predict different targets. Mean RMSE similarity also conceals cancellation: A1−A0 dynamic RMSE changes range from **−0.02211 to +0.01448 percentage points of strain** across directions.

Exact REPORT wording:

> The gap-contraction diagnostic was frozen in `e8a4e61`. Relative contraction of the mean static-minus-dynamic RMSE gap was 0.234 for oracle re-alignment, 0.378 for source-constant re-alignment, 0.380 with original normalization retained, and 0.206 for cross-fitted offsets; gaps decreased in 7/8, 8/8, 7/8, and 6/8 directions, respectively. All met the conventional “partial contraction” rule. The oracle mean gap decreased from 0.02416 to 0.01850 percentage points of strain. Group dependence was substantial: leave-one-group-out contraction ranged from 0.099–0.405 for oracle and 0.268–0.521 for source-constant alignment. These comparisons change the prediction target, training labels, scored rows, and usually normalization. They neither identify the fraction of the original advantage attributable to alignment nor establish dynamic-model insensitivity or a mechanism. Evidence: `research/probes/out/dynamic_gain_attribution_{per_fold,gaps}.csv`.

Exact INDEX replacement for the open item:

> Completed diagnostic (`direction/dynamic-gain-attribution`, frozen `e8a4e61`): all four re-alignment arms yielded partial mean-gap contraction (F=0.206–0.380). Preserve preregistration, probe, and both output CSVs. Avoid interpreting F as a mechanistic attribution fraction or similar mean RMSE as alignment insensitivity. dataset0 remained sealed.
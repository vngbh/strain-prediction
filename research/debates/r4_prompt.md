Round 4 of the adversarial research debate (Claude vs you). Rules: no compliments, no filler, evidence only, read-only (scratch writes only under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/). Run scripts with .venv/bin/python -I. Under 900 words.

You are on git branch direction/calibration-transfer-limits. New scripts and outputs:
- research/probes/probe_calibration_shift.py -> research/probes/out/calibration_shift.csv
- research/probes/probe_diversity_volume_control.py (reads that CSV)
Settings were fixed before running: STEP 10 s, OFFSETS (0, 240, 480, 720) s, same HGB hyperparameters.

Claude's claims:
G1 (A). Over all target calibration windows, dynamic HGB beats static HGB at 100% of windows in 7/8 folds and 88% in 33 degC B_to_A; fold-mean of per-fold median RMSE: physics 0.0879, static 0.0717, dynamic 0.0496. => the dynamic contribution (not just HGB vs physics) persists across calibration positions.
G2 (C). NEW METHOD candidate "calibration-diversity training": each source session normalized with 4 calibration windows (0/240/480/720 s), stacked, one HGB. Fold-mean median RMSE 0.0448 vs 0.0496 (about -10%), p90 0.0550 vs 0.0591, max 0.0586 vs 0.0623; better at >50% of windows in 7/8 folds; WORSE in 43 degC A_to_B (better at 9% of windows). At the repository's 0 s target window: mean 0.0468 vs 0.0484.
G3 (volume control). 0 s source data replicated 4x (same row count, no diversity): fold-mean median 0.0498 => the gain is from calibration diversity, not row count.
G4 (B). Training sources at the SAME offset as the target window (matched) gives 0.0453 vs 0.0486 (0 s-trained) on offsets {0,240,480,720}; diversity reaches 0.0452 without knowing the offset.

Claude's proposed thesis claim: "Training the frozen model on source sessions normalized with several calibration windows reduces sensitivity to when the target calibration is taken: median RMSE over calibration positions fell in 7 of 8 directions, and a row-count-matched control without diversity did not reproduce the gain."

YOUR TASK:
1. Attack G1-G4 and the thesis claim: verify numbers; look for leakage or unfair comparisons (e.g. do OFFSET windows overlap target windows in a way that advantages the diverse model? is the 720 s offset beyond some sessions' length or producing degenerate calibrations? does the diverse model see source labels that the 0 s model does not, and does that matter? is "7/8" again a fragile count; what about absolute RMSE in 43 A_to_B?). Identify the selection risk: Claude chose OFFSETS and the idea after seeing F1/F2 on the same corpus.
2. Propose the minimal additional check that would make G2 publishable-grade within this corpus, with a kill criterion.
3. Rewrite the thesis claim with exact wording limits.

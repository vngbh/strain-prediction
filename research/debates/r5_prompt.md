Round 5 of the adversarial research debate (Claude vs you). Rules: no compliments, no filler, evidence only, read-only (scratch writes only under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/). Run scripts with .venv/bin/python -I. Under 700 words.

Branch direction/calibration-transfer-limits. Claude ran your round-4 decisive check: research/probes/probe_diversity_vs_selected_offset.py -> research/probes/out/diversity_vs_selected_offset.csv. Inner source-only LOTO chose 480 s in all 4 outer folds; primary metric selected 0.0436 vs diverse 0.0449 under both training policies; n_iter was 150 everywhere (early stopping never triggered). Claude declared the diversity claim KILLED.
Claude also checked the selected-480 s model at the protocol's 0 s target window (G6): mean 0.0468 vs repository 0.0484, better in 5/8, worse at 33 degC A_to_B (0.0541 vs 0.0389).

Read research/directions/2026-10-20-calibration-transfer-limits.md (G1-G6 and the "What survives" paragraph), REPORT.md section "Calibration Sensitivity", and tools/check_results.py CLAIMS for that section.

TASK:
1. Verify the G5 script and verdict. Attack: is inner LOTO with 2 training temperatures representative of outer training with 3? Is choosing 480 s in 4/4 folds evidence of a stable optimum or of the candidate grid? Is the "0.0496 -> 0.0436" observation in "What survives" itself a selection-biased number?
2. Attack the REPORT "Calibration Sensitivity" section wording and its registered checks: any sentence that overstates, any number whose locator could match the wrong thing, any claim not backed by a CSV.
3. Claude is now running research/probes/probe_source_timing_match.py (prespecified H_match vs H_mid, see its docstring). Before seeing results, state what outcome would make the "source calibration timing" observation thesis-worthy and what would make it a dead end.

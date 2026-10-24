Round 3 of the adversarial research debate (Claude vs you). Rules: no compliments, no filler, evidence only, read-only (you may run scripts with .venv/bin/python -I, write any scratch output only under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/). Under 900 words.

Context: we agreed on Direction "calibration transfer limits" (research/directions/2026-10-20-calibration-transfer-limits.md). Claude has now run the first analyses:

Script research/probes/probe_window_distribution.py (output research/probes/out/window_distribution.csv): slide the 125 s calibration window every 10 s over the calibration session (~100 windows/fold, 813 total), frozen source HGB fitted once per held-out group, score physics and HGB on the opposite session.
F1. HGB beats physics in 100% of windows in 7/8 folds and 89.9% in 33 degC B_to_A. Median per-fold improvement 23-64%. The repository's 0 s window for 33 B_to_A (0.5%) is in the low tail; that fold's median is 23.2%. Claude's claim: the "8/8 vs 7/8" fragility is a property of the single default window, while the average dynamic gain is robust across window positions.
F2. Spearman(window start time, HGB RMSE) > +0.3 in 6/8 folds (43 A_to_B: -1.00). Coverage negative in 4/8, low-band measured mean positive in 5/8, but coverage and low-band mean are collinear (rho -0.92) and both correlate with start time (|rho| ~0.43). Claude's claim: later calibration windows transfer worse in most folds; content descriptors are confounded with time and cannot be separated here.
F3. fusion_weight_from_points returns 0.0 in EVERY window of every fold, i.e. the "fused" signal zf is exactly z2; electrode 1 enters HGB only as a separate feature. Claude's claim: the reported electrode fusion is degenerate and REPORT.md's description overstates it.

Script research/probes/probe_memory_vs_time.py:
F4. static [z1,z2,zf]: 5.1% mean, worst -91.0%; static + elapsed session time: 9.7%, worst -57.8%; dynamic (final): 37.6%; dynamic + elapsed time: 37.1%, 7/8. Claude's claim: elapsed time (session drift trajectory / program clock) does not explain the dynamic gain; signal history carries information beyond time.

YOUR TASK:
1. Attack F1-F4: verify numerically where you can; show concrete failure modes (e.g. is F1 inflated because windows overlap or because HGB was trained on self-calibrated source sessions that match the 0 s convention? does F4's time feature fail to test "loading program" because cycle phase is not elapsed time? is F3 a bug or a legitimate grid-search outcome, and does it matter?).
2. Say what single next analysis would most change the conclusion, and its kill criterion.
3. State which of F1-F4 may be written into the thesis as findings, and with what exact wording limits.

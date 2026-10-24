New pre-registration review (round g1), adversarial. Rules: no compliments, evidence only, READ-ONLY; do NOT execute research/probes/probe_dynamic_gain_attribution.py or compute its outcomes (design must be frozen first). dataset0 SEALED. Scratch under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/ (MPLCONFIGDIR there). Under 700 words.

Read research/directions/2026-10-21-preregistration-dynamic-gain-attribution.md and the script. Context: research/INDEX.md, the reference-alignment v1/v2 notes, REPORT.md (main ablations: dynamic 37.6 % vs static 5.1 %).

Attack the design:
1. Is F = 1 - G_oracle/G_original a valid measure of "dynamic gain absorbed by alignment"? Re-aligning labels changes the target, the calibration (125 s window now on shifted labels), and the static baseline itself; can F exceed 1 or go negative, and how to read it? Is the oracle tau circular in a way that biases F upward (tau was fitted to maximize a static linear fit, which favors the static model)?
2. Is the source-constant variant fair given offsets range 1.8-9.1 s (62 degC exp10 ~8.8 s)? Should the primary instead be source-constant, or both required?
3. Thresholds (F >= 0.5 with 6/8; F <= 0.2): sensible? Should per-direction instability (gains near zero in some directions) be handled (e.g. ratio of means vs mean of ratios)?
4. Code: realign() edge handling, tau sign (positive tau = resistance aligns with EARLIER strain, so label at t becomes eps(t - tau)), whether evaluate_fold recomputes physics/calibration on the shifted labels, any bug that trivially passes/fails.
Give concrete edits, or FREEZE.

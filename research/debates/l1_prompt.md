New direction (reference lag), design review before freezing. Rules: no compliments, evidence only, READ-ONLY. Do NOT execute research/probes/probe_reference_lag.py and do NOT compute lag/collapse results (the design must be frozen first); static review plus input checks that do not compute those outcomes. dataset0 is SEALED: never load it. Scratch under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/ (MPLCONFIGDIR there). Under 700 words.

Also: you stopped mid-audit last time (usage limit) while checking Claude's withdrawn interpretation of the latent gates. Claude since showed post hoc that the high cross-electrode residual correlation lowers the Gate-B bound (rho=0 gives 0.0103-0.0195 vs 0.0065-0.0121), withdrew the "little independent information" claim and the E6/fusion link; see the Result section of research/directions/2026-10-21-preregistration-latent-gates.md. One line: do you accept that corrected wording?

Then attack the DRAFT: research/directions/2026-10-21-preregistration-reference-lag.md and research/probes/probe_reference_lag.py.
1. Does the OLS-with-time lag estimator confound lag with hysteresis/rate-dependent effects (a loading/unloading-asymmetric response also biases tau*)? Is the collapse ratio a valid discriminator between pure lag and rate-independent hysteresis? Better discriminators (e.g. branch-wise slope symmetry, harmonic phase linearity)?
2. Are thresholds (0.3 s, 14/16, C <= 0.70, H_null 0.2 s) sensible given 0.1 s sampling, the strain rate (~0.009 %/s), and the trailing 21-sample smoother in the HGB features (irrelevant here?)?
3. Interpretation rules (fixed offset vs viscoelastic vs electrode-specific): fair, given temperature and session are nested and only 8 sessions?
4. Code bugs: sign convention of shifted(), parabola refinement, loop area with drift removal, lag-only RMS, decision counting with unresolved values.
Give concrete edits, or FREEZE.

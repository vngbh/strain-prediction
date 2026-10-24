Latent-state direction, round 2 (Claude vs you). Rules: no compliments, evidence only, READ-ONLY. Do NOT run research/probes/probe_latent_gates.py or compute its gate results (the design must be frozen before results exist). You may inspect inputs needed to judge the design. dataset0 is SEALED: never load it. Scratch under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/ (MPLCONFIGDIR there). Under 700 words.

Claude conceded your round-s1 points (regime misreading, trivial PC1, label-based coefficients, eps/state trade-off) and wrote a DRAFT pre-registration of your proposed gates: research/directions/2026-10-21-preregistration-latent-gates.md, implemented (unexecuted) in research/probes/probe_latent_gates.py.

Attack the DRAFT before freezing:
1. Gate A: is the 2x2 map L vs controls T0/T1/T2 a fair test of "transferable coupling"? Is T2's extrapolation rule fair? Is pooling dB2 and dS2 in one RMSE sensible given different scales? Does "every temperature group" make the gate impossible by construction?
2. Gate B: check the Fisher matrix code against the stated model (dR_i/ds = B_i0 b_i + eps S_i0 g_i with s constant per horizon). Is a 0.01 % strain CRB threshold meaningful? Is the noise estimate (MAD of diff) valid when strain changes every sample? Is the singular vector of relative changes the right loading definition (units: relative vs absolute)?
3. Any code bug that would make either gate trivially pass or fail.
Give concrete edits; say "freeze" if nothing blocks.

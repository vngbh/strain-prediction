New direction, round 1 of an adversarial debate (Claude vs you). Rules: no compliments, no filler, evidence only, read-only (scratch under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/, MPLCONFIGDIR there). Web search allowed and REQUIRED for novelty. Under 1000 words.

HARD CONSTRAINTS: no new measurements will ever exist (confirmed). dataset0.csv is SEALED as a one-shot confirmation set: do NOT read, load, plot or summarize it.

Read: research/directions/2026-10-21-latent-state-model.md, research/probes/probe_latent_identifiability.py and its output below, and the parent note research/directions/2026-10-20-calibration-transfer-limits.md (especially E6: a frozen 1-parameter ADDITIVE drift corrector from z1 - z2 failed). The source paper text is extractable from to_know_how/files/admt202000957.pdf (pypdf in .venv): see pp. 5-6 on carbon-black network reorganization on annealing and the in-phase/out-of-phase threshold between 35 and 45 degC.

Step-0 output (non-overlapping 125 s windows every 130 s, PCA of relative changes of B1, B2, S1, S2 per session):
pooled PC1 loading (B1,B2,S1,S2) = 0.54 0.38 0.54 0.53, share 0.66
33C-3 share 0.96 cos 0.91 | 33C-4 0.92/0.99 | 43C-5 0.98/0.98 | 43C-6 0.97/0.93 | 52C-7 0.79/0.63 (B2 -0.38) | 52C-8 0.82/0.85 | 62C-9 0.72/0.73 (B2 -0.26) | 62C-10 0.77/0.28 (B1 -0.18, B2 -0.57)
Claude's reading: one latent factor dominates within sessions; its loadings are stable at 33/43 but change at 52/62 (regime change consistent with the paper's 35-45 degC threshold). Claude's plan: a state-space model R_i = B_i(s) + S_i(s)*eps with a slow shared state s (multiplicative + additive), loadings possibly regime-dependent, estimated online by a Kalman filter from two electrodes; frozen LOTO; then a pre-registered one-shot test on sealed dataset0.

TASK:
1. Novelty: search for state-space / Kalman / latent-state drift or sensitivity tracking in carbon-black (or CNT) polymer piezoresistive sensors, dual-element/dual-electrode observability, and thermally drawn fiber sensors. Report what exists with DOIs/URLs. Is anything left that is not done?
2. Attack the feasibility reading: 7-9 windows per session for a 4-variable PCA; overlap; is "one factor" trivial because B and S co-trend with time anyway (any monotone drift gives PC1)? Propose a sharper identifiability test that does not touch dataset0.
3. Attack the plan: given E6 failed and loadings change with temperature, can a frozen LOTO latent model work at edge temperatures (33 and 62 are extrapolation)? Is dataset0 (21 degC, below the training range, one session) a valid confirmation target, and what exactly could be tested there given only one session (no opposite session)?
4. Verdict: continue / redesign / stop, with a concrete next step and kill criterion.

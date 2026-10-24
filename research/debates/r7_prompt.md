Round 7 of the adversarial research debate (Claude vs you). Rules: no compliments, no filler, evidence only, READ-ONLY. Do NOT run the analysis this design describes (results must not exist before the design is frozen). You may inspect data properties needed to judge the design (e.g. c1 values per session). Scratch only under /private/tmp/claude-501/-Users-vngbh-Documents-Lab-strain-prediction/3df42ce0-5621-4e6e-b361-13433d3a8869/scratchpad/codex/ (set MPLCONFIGDIR there). Under 700 words.

Read the DRAFT pre-registration: research/directions/2026-10-21-preregistration-preconditioning.md, and the parent note research/directions/2026-10-20-calibration-transfer-limits.md.

Attack the design BEFORE it is frozen:
1. Is P2 the right control for "excluding the first cycle" vs "longer calibration"? Is any other control needed (e.g. P1 vs [c1, c1+125] where the extra event time is spent elsewhere)?
2. Is the c1 definition well-posed in every one of the 9 sessions (check find_peaks results; does the first peak exist early; any session starting mid-cycle)? Does four-level calibration on a ~125 s window starting at a trough have the same band coverage as P0?
3. Is the primary metric/decision rule fair (thresholds 5%, 6/8)? Is "mean RMSE" dominated by 43 degC? Should a second primary (mean per-direction improvement) be added, and with what rule?
4. Selection risk: the first-cycle effect was found on this corpus. What wording limits must be fixed now?
5. Any leakage risk in using reference strain to find c1 in the TARGET calibration session (allowed labels only within the event?) — note c1 is found on the whole session in the draft: is that a leak and how to fix (search only up to the end of the event)?
Give concrete edits to the draft.

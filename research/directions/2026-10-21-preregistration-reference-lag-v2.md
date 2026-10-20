# Pre-registration v2: Wider-Grid Sensitivity Analysis Of Apparent Alignment

Status: frozen by the git commit that adds this file together with
`research/probes/probe_reference_lag_v2.py`, before that script is run.
Branch `direction/reference-lag`. dataset0 stays SEALED and is not needed.

## Nature Of This Analysis (disclosed)

This is a **data-informed sensitivity analysis**, not a confirmation. The
original test (frozen in `b82b769`, design and result in
`2026-10-21-preregistration-reference-lag.md`) returned "insufficient
coverage" (13/16 valid; minimum 14). Before choosing this analysis, the
following was known: the 13 resolved offsets (1.84-4.61 s, all positive,
intervals excluding 0); that three 62 degC electrode-sessions had unique
minima at the +5.0 s grid edge with RMS still decreasing; and the descriptive
collapse ratios and harmonic phases. The original verdict stands and is
reported alongside this one; nothing here replaces it.

## What Changes And Why

- Only two constants: shift grid [-15, +15] s (was [-5, +5] s) and the fixed-
  support margin 15 s (was 5 s), so every candidate shift is compared on the
  same support. Everything else (0.1 s resolution, cross-fitting, 100 paired
  cycle-bootstrap resamples with seed 0, unresolved-resample policy, loop-area
  rules, decision rule, coverage minimum 14) is identical; the script differs
  from the frozen one only in these constants and its docstring.
- The range is an engineering search range, not a physical bound: no
  documented timing bound exists. +/-15 s is three times the original range
  and about 13 % of the ~115 s loading period, beyond which a "shift" would
  start to alias a fraction of a cycle.
- The procedure is applied uniformly to all 16 electrode-sessions; the
  original 13 estimates are preserved in their original artifact and changes
  are reported, not spliced.

## Stopping Rule

This is the last widening. Any minimum at the new +/-15 s edge stays
unresolved, and the primary verdict is then reported as it falls.

## Decision Rule (unchanged)

Valid = resolved `tau*` and a valid bootstrap interval; at least 14 valid,
else "insufficient coverage". "Consistent apparent alignment" if
`max(count(tau* >= 0.3 s with interval > 0), count(tau* <= -0.3 s with
interval < 0)) >= 14`; "no material apparent alignment" if at least 14
intervals lie within [-0.2, +0.2] s; otherwise "inconclusive". The
identifiability limit of the original pre-registration applies unchanged: no
mechanism is inferred from any result.

## Provenance Frozen Before Running

- Command: `.venv/bin/python -I research/probes/probe_reference_lag_v2.py . research/probes/out/reference_lag_v2`
- Script SHA-256: `4d33754856d547c4790a04d4cfc55d0ef7bbf2ab9247b4dfbc7ad1499f8e3c5f`
- Input SHA-256: dataset1 `81d40113...b29ea`, dataset2 `4ea77e04...039a`,
  dataset3 `aec2a927...d9299`, dataset4 `911cdc50...7c2d5` (full hashes in
  `results/manifest.json`).
- Python 3.13.11; numpy 2.4.4; pandas 3.0.2; scipy 1.17.1.
- Also report: margin effect on support (samples per session under 5 s vs
  15 s margins).

## Result (appended after running; the sections above are unchanged since commit 81a4d86)

Output: `research/probes/out/reference_lag_v2_per_electrode_session.csv`.
Audited by Codex (round l6): source identical to 81a4d86 and to the registered
hash; verdict reproduced from the CSV; support masks identical at both margins
(separate disclosed check: no session's support changed).

**Primary: consistent apparent alignment (no mechanism inferred).** 16/16
valid; every estimate exceeded 0.3 s with a 95 % cycle-bootstrap interval
excluding zero. All 13 original point estimates are unchanged; experiment 8
electrode 1 gained a previously unresolved bootstrap draw and odd-cycle fit,
which changed its interval and collapse values. This sensitivity result does
not replace v1's verdict ("insufficient coverage") or constitute independent
confirmation.

Electrode differences (paired bootstrap): `tau1 - tau2` = +0.36..+0.64 s with
intervals excluding zero in 7 sessions, and -0.44 s (interval excluding zero)
at 33 degC experiment 4. Agreed wording: under a model where both channels
differ from the reference solely through one shared acquisition offset and
otherwise follow the fitted response model, equal offsets would be expected.
The electrode differences challenge that restricted explanation. However,
apparent alignment also absorbs electrode-specific branch asymmetry and other
response differences; it cannot separate these from a shared timing
contribution. The acquisition arrangement is undocumented.

Experiment 10 caveat: its offsets were 9.121 and 8.478 s. Neither electrode had
an estimable cross-fitted loop-area collapse ratio (no evaluation cycle met
the bin-coverage rule), so these estimates lack that secondary check and must
not be interpreted as measured acquisition delays.

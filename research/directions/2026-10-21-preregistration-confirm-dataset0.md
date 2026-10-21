# Pre-registration: One-Shot Confirmation Of Apparent Alignment On dataset0

Status: frozen by the git commit that adds this file together with
`research/probes/probe_reference_lag_confirm_dataset0.py`, before that script
is run. The script is run ONCE. Design and literature reviewed by Codex
(round c1; all four blockers accepted). Branch `direction/reference-lag`.

## Why This Test

The apparent-alignment finding (resistance aligning with reference strain
from earlier times) rests on v2, a data-informed sensitivity analysis after v1
returned insufficient coverage. `dataset0.csv` (21 degC, below the 33-62 degC
development range) was sealed on 2026-10-20 for a one-shot confirmation.

Exposure, disclosed exactly: dataset0 was not used in any step of the
alignment development (v1, v2) or in any analysis after the seal. Before the
seal it was used in April 2026 EDA, in July 2026 Phases E/F as a temperature
point for B(T)/S(T), and on 2026-10-20 in one probe
(`probes/probe_calibration_transfer.py`) that used its full-label B/S for
temperature interpolation. None of these estimated an alignment offset. Its
SHA-256 was computed on 2026-10-21 with `shasum`, which reads the bytes; no
content was displayed or analysed.

## Literature Status (Codex round c1, web search)

Adjacent work exists: embedded FBG versus surface gauge agreement (Ling et al.
2005, doi:10.4028/www.scientific.net/KEM.295-296.21); resistance hysteresis of
embedded CNT yarns with a reference extensometer (Bai et al. 2021,
doi:10.1155/2021/2427954); time-dependent CNT yarn and thermally drawn fiber
responses (Anike et al. 2016, doi:10.3390/c2010003; Leber et al. 2023,
doi:10.1002/advs.202207573); viscoelastic strain transfer to FBGs (Liang et
al., doi:10.1364/AO.57.005837; Zhai et al., doi:10.1364/AO.58.004306); a
seconds-scale acquisition lag between ultrasonic measurements and a surface
gauge on carbon-epoxy (Bulletti et al. 2015, doi:10.1155/2015/259621);
explicit synchronization of unsynchronized acquisition systems in a composite
dataset (Lueders et al., doi:10.1016/j.dib.2025.111333). No verified report of
seconds-scale apparent alignment between an embedded piezoresistive fiber and
a surface gauge in a fiber-reinforced composite was found; search absence is
not priority.

## Procedure (fixed)

- Estimator: the frozen v2 estimator, executed verbatim from
  `probe_reference_lag_v2.py` after checking its SHA-256
  (`4d33754856d547c4790a04d4cfc55d0ef7bbf2ab9247b4dfbc7ad1499f8e3c5f`): +/-15 s
  grid, 0.1 s resolution, 15 s margin, cycles from reference troughs,
  parabola refinement, unresolved rules, 100 paired cycle-bootstrap resamples,
  seed 0 with a fresh generator per session, interval invalid if > 10 % of
  resamples are unresolved. Only `tau*`, intervals, the paired difference
  interval, and cross-fitted `tau_odd`/`tau_even` are computed; v2's collapse
  ratios and harmonic phases are omitted. Results remain conditional on the
  cycle-bootstrap assumptions and only 100 resamples.
- Gate before opening dataset0: recompute dataset1 experiment 3 (first session
  of the v2 loop, same generator start) and reproduce the v2 CSV values of
  `tau_s`, `ci_low`, `ci_high`, `tau_odd`, `tau_even`, and the paired difference
  interval within 1e-12; otherwise stop without opening dataset0.
- One-shot enforcement: before the first byte of dataset0 is read, the script
  creates `research/probes/out/dataset0_CONSUMED.json` exclusively (opened at,
  git HEAD, environment). If that file exists the script refuses to run. The
  bytes are read once, hashed (must equal
  `7862bef32dbed29c7f249939aff2057d451b70d924d60606746a477132af0d7e`), and
  parsed from the same buffer. All sessions in the file are analysed.
- Technical failure after access (any exception after the marker exists,
  including a hash mismatch): the marker records the failure, P1 is reported
  as NOT CONFIRMED, and no confirmatory retry is allowed. Verdicts are printed
  only after every step has succeeded.
- Pinned inputs, ENFORCED before the gate and before any access (the script
  stops if any differs): gate CSV `reference_lag_v2_per_electrode_session.csv`
  SHA-256 `43533eb5...7e17f`; `src/strain/spec.py` SHA-256 `befaf426...a2005d`;
  Python 3.13.11, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1. P3 reads the
  threshold from the verified gate CSV.

## Predictions

- **P1 (primary):** in every electrode-session of dataset0, `tau*` is
  resolved, `tau* >= 0.3 s`, and the 95 % bootstrap interval is valid and lies
  entirely above 0. "Confirmed" or "not confirmed"; unresolved or invalid
  counts as not confirmed.
- **P2 (secondary):** the paired interval of `tau*_1 - tau*_2` is valid and
  lies entirely above 0 in every session (the sign seen in 7 of 8
  development sessions).
- **P3 (exploratory):** in every session exactly two electrode estimates exist, both finite,
  and their mean is below the 33 degC development mean computed from the gate
  CSV (2.6175 s). The development data did not establish a temperature trend.

## Interpretation Limits (fixed now)

- A confirmation means the apparent alignment reappears in a session not used
  in alignment development, at a temperature outside the development range,
  with the same estimator. It does not identify a mechanism (one loading
  program; delay and branch asymmetry remain indistinguishable) and is one
  session of the same specimen, not an independent replicate.
- A non-confirmation is reported as such; the v2 result is then limited to
  33-62 degC, and no further estimator change may be tried on dataset0.
- After this run dataset0 is no longer sealed; no untouched data remain.

## Result (appended after the single run; the sections above are unchanged since commit ea86e3b)

Run once at 2026-10-21 11:06:06 +09:00 (marker
`research/probes/out/dataset0_CONSUMED.json`, status completed). Gate passed
before access. Output `research/probes/out/reference_lag_confirm_dataset0.csv`.
Audited by Codex (round c5): source identical to ea86e3b, pins matched,
verdicts recomputed from the CSV.

| Prediction | Result |
|---|---|
| P1 (primary) | **NOT CONFIRMED**: electrode 1 +0.095 s [0.046, 0.153]; electrode 2 -0.159 s [-0.217, -0.090] |
| P2 (secondary) | met: paired difference [0.236, 0.268] s |
| P3 (exploratory) | met: two-electrode mean -0.032 s < 2.6175 s |

Agreed wording (Codex round c5): the prespecified positive apparent-alignment
finding remains limited to the eight development sessions at 33-62 degC. This
single additional session from the same specimen establishes neither
temperature dependence nor mechanism and does not exclude a shared
acquisition-offset contribution. Acquisition comparability is undocumented.
"Near zero" describes magnitude only (both intervals exclude zero).

Withdrawn (Claude): "a fixed acquisition offset would appear at every
temperature, so this argues against a constant acquisition offset" (apparent
alignment also absorbs response differences; a constant timing contribution
could coexist with changing response effects). Viscoelastic strain transfer
may be named only as one untested possibility alongside acquisition/setup
differences and branch asymmetry.

dataset0 was consumed on 2026-10-21 and is no longer sealed; no untouched
data remain. No confirmatory retry.

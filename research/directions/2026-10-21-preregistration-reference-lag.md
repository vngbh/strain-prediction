# Pre-registration: Apparent Alignment Between The Embedded Sensor And The Reference Gauge

Status: frozen by the git commit that adds this version together with
`research/probes/probe_reference_lag.py`, before that script is run. Design
reviewed by Codex (round l1). Branch `direction/reference-lag`. dataset0 stays
SEALED and is not needed.

## Motivation

- The latent-state gates found that both electrodes' deviations from a
  branch-wise linear response to the reference strain are strongly correlated
  (rho 0.93-0.98) and autocorrelated. A shared deviation could come from the
  sensor, the host composite, or the reference measurement.
- The reference is a surface-bonded gauge (HBM LY11-6/350); the fiber is
  embedded. Strain transfer and the synchronization of two acquisition chains
  are not documented in the source paper.

## Overlap Check (research/INDEX.md)

`git grep` for lag, cross-correlation, phase shift, synchronization, and delay
in `archive/2026-04-dl-exploration`, `archive/2026-07-diagnosis`, and
`archive/2026-09-hgb-development` found no prior lag or synchronization
analysis. July Phase E reported that loading and unloading sensitivities are
not consistently separable (2/8 combinations).

## Identifiability Limit (fixed before running)

Measured median absolute strain rates are 0.0088-0.0095 %/s in all eight
sessions (one loading program). On near-constant-rate ramps,
`b eps(t - tau) ~ b eps(t) - b tau deps/dt`, and `deps/dt` only changes sign
with the branch, so a pure time delay and a rate-independent loading/unloading
offset produce the same signature. This corpus has no genuinely different ramp
rates. Therefore **no result of this study is attributed to a mechanism**
(acquisition offset, viscoelastic strain transfer, gauge/adhesive, or sensor
hysteresis). The study asks only whether a consistent **apparent alignment
offset** exists.

## Fixed Definitions

- Sessions: dataset1-4 (8 sessions), both electrodes: 16 electrode-sessions.
- Cycles: consecutive troughs of the reference strain
  (`find_peaks(-strain, prominence=0.25, distance=600)`), complete cycles only.
- Shift `tau` (positive = resistance aligns with EARLIER reference strain):
  model `R(t) = a + b * eps(t - tau) + c * t`, with `eps(t - tau)` linearly
  interpolated in time. Candidate shifts on a 0.1 s grid in [-5, +5] s are
  compared on one fixed support (samples whose shifted reference exists for
  every candidate). `tau*` minimizes residual RMS, refined by a parabola
  through the minimum and its neighbours. Unresolved if the minimum is at a
  grid edge, tied, or the parabola is not convex.
- Uncertainty: 95 % percentile interval from 100 bootstrap resamples of
  complete cycles (seed 0), refitting `tau*` on the 0.1 s grid with the same
  refinement. The same cycle resamples are used for both electrodes of a
  session (paired), which also gives the interval of `tau*_1 - tau*_2`.
  Unresolved-resample policy: if more than 10 % of resamples are unresolved,
  the interval is invalid (the electrode-session is not valid for the primary
  rule); otherwise the percentiles use the resolved resamples and the
  unresolved fraction is reported.
- Cross-fitting: `tau*` is also fitted on odd cycles only and on even cycles
  only; the loop-area collapse of each half is evaluated with the shift fitted
  on the other half.
- Loop area (descriptive only): per complete cycle, branches split at the
  cycle's strain maximum; branch means of `(R - a - c t) / b` in 0.01 % bins
  over 0.05-0.45 % of the (shifted) reference strain; a cycle is valid only if
  every bin is populated in both branches at both compared shifts; area =
  trapezoid of the absolute branch gap. Primary nuisance `(a, b, c)` fixed from
  the zero-shift fit; refitted nuisance reported as sensitivity.
  Collapse `C = A(shift) / A(0)`.

## Primary Decision Rule

Valid electrode-session: resolved `tau*` and a bootstrap interval. Coverage
minimum: at least 14 valid, else "insufficient coverage".

- **Consistent apparent alignment:** `max(count(tau* >= 0.3 s),
  count(tau* <= -0.3 s)) >= 14` among valid electrode-sessions, AND the
  bootstrap interval excludes 0 in each of those.
- **No material apparent alignment:** in at least 14, the bootstrap interval
  lies within [-0.2, +0.2] s.
- Otherwise: **inconclusive**.

"Material" is an engineering convention: at 0.009 %/s, 0.3 s corresponds to
about 0.0027 % strain (27 microstrain) of local misalignment. Counts are a
consistency requirement, not a significance test (electrodes share sessions).

## Descriptive Secondaries (no decision role, no mechanism labels)

- Electrode agreement: per session `tau*_1 - tau*_2`, with intervals.
- Temperature association: session-mean `tau*` versus temperature, listed (no
  test, no labels; temperature and session are nested).
- Cross-fitted collapse ratios (fixed and refitted nuisance).
- Harmonic phases: per cycle, DFT of R and reference strain, both with the same
  linear-in-time detrend, at the
  1st, 3rd, and 5th harmonics of that cycle; implied delays
  `tau_k = -angle(H_k) / (2 pi k / T_cycle)`, medians over cycles. Static
  nonlinearity also alters harmonic content, so these are not decisive.
- Strain-equivalent RMS of the shift alone, `RMS(eps(t) - eps(t - tau*))`.
  Not an error floor and not an attributable share of HGB error.

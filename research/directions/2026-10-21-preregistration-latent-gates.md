# Pre-registration: Two Gates Before Building A Latent-State Model

Status: frozen by the git commit that adds this version together with
`research/probes/probe_latent_gates.py`, before that script is run. Design
reviewed by Codex (rounds s1, s2). Branch `direction/latent-state-model`.
dataset0 stays SEALED throughout.

## Why Gates

Codex (round s1) showed the step-0 PCA cannot support the direction: per-
coefficient time trends alone explain 47-97% of the variation, the B/S
estimates use labels all session long, and without restrictions strain and a
latent state can trade off (`eps' = eps + a(s)`, `B_i' = B_i - S_i a(s)` leaves
resistance unchanged). Prior work already occupies Kalman compensation,
latent sensor dynamics, and two-channel decoupling for carbon/polymer sensors
(Wang 2021, Yong & Aw 2022, Amindehghan 2021, Olowo 2023, Wu 2019; see the
direction note). A model is built only if both gates pass.

A failure ends this shared-state method direction on this corpus. It does not
disprove shared-state models in general: the gates are conservative
engineering requirements on a small, dependent corpus.

Corrected: the source paper's 35-45 degC in-phase/out-of-phase threshold
concerns orthogonally placed fibers. No regime claim is made.

## Coefficient Estimation (both gates)

- Sessions: dataset1-4 only (8 sessions).
- Cycles: consecutive detected troughs of the reference strain
  (`find_peaks(-strain, prominence=0.25, distance=600)`). The session start is
  used as the first cycle's start only if `strain[0] <= min(strain) + 0.02`;
  otherwise the first segment is discarded. Trailing segment after the last
  trough is discarded.
- Per cycle and electrode: equal-weight 0.05 % strain bins over 0.20-0.45 %,
  loading and unloading branch means pooled with equal weight, OLS of mean R
  on mean measured strain gives slope `S_i,c` and intercept `B_i,c`. Cycles
  with an empty branch-bin are dropped and counted.
- Relative change `d_c = coef_c / median_session(coef) - 1` for B1, S1, B2, S2.
- Coverage minimum: every session keeps >= 6 cycles, otherwise the run stops
  as "insufficient coverage".

## Shared Rank-One Model (both gates)

Per fold (leave one temperature group out), on the 6 source sessions only:
standardize the four relative changes by their source standard deviations
`sigma_k`; the loading `v` is the first right-singular vector of the
standardized source matrix. The oracle state of a target cycle is the least-
squares coordinate of its standardized `[dB1, dS1]` on `v[0:2]`; the
prediction of `[dB2, dS2]` is `state * v[2:4]`, rescaled by `sigma`.
Relative loadings for Gate B are `v_k * sigma_k` (dimensionless).

## Gate A: Transferable Cross-Electrode Coupling (oracle state)

Loss for a held-out group: mean over its 2 sessions of the per-session RMSE of
the standardized `[dB2, dS2]` (both components pooled; component RMSEs are
reported too). Target cycles are restricted to cycle indices present in the
source sessions (common support); excluded cycles are counted.

- A1 (raw coupling). Rank-one predictor L vs controls fitted on sources:
  T0 zero change; T1 per-coefficient linear trend in cycle mid-time; T2 per-
  coefficient source mean by cycle index.
  Pass A1: `loss(L) <= 0.9 * min(loss T0, T1, T2)` in every temperature group.
- A2 (coupling beyond time/program). Subtract the better (on sources) of T1/T2
  from all four relative changes, per coefficient; refit the rank-one model on
  source residuals; predict target electrode-2 residuals from electrode-1
  residuals. Pass A2: `loss(L_resid) <= 0.9 * loss(zero residual)` in every
  temperature group.
- Pass A = A1 and A2. Guard: a zero control loss is reported as "inconclusive"
  for that group and counts as a failure.

## Gate B: Strain/State Separability At Estimated Noise

- Model: `R_i = B_i0 (1 + b_i s) + S_i0 (1 + g_i s) eps`, with `b_i, g_i` the
  relative loadings of the fold's rank-one model and `B_i0, S_i0` from the
  target session's first 125 s (repository four-level calibration). `s = 0`
  at calibration. Applying median-referenced loadings around prefix values is
  an approximation, disclosed.
- Noise: per session, residual of R after removing the local branch response:
  within each cycle and branch, OLS of R on reference strain; residuals pooled.
  Noise covariance `Sigma` (2x2) = covariance of the two electrodes' residuals
  (robust scale: `1.4826 * MAD` per electrode, Pearson correlation of
  residuals). Lag-1 autocorrelation of residuals is reported as a diagnostic
  (white-noise assumption).
- Bound: retrospective smoothing Cramer-Rao bound (not a causal online bound),
  `s` assumed constant over 100 s horizons (1000 samples), unknowns are every
  sample's strain and `s`. With `a = (S_10, S_20)`, `h_k` the per-sample state
  derivative vector, `d = a' Sigma^-1 a`, `c_k = a' Sigma^-1 h_k`,
  `q = sum_k h_k' Sigma^-1 h_k`, Schur term `m = q - sum_k c_k^2 / d`:
  `var(eps_k) >= 1/d + (c_k / d)^2 / m`. If `m <= 1e-12 * q`, the state is
  unidentified and the bound is infinite.
- Criterion (a chosen noise budget, not a necessary condition for beating
  HGB): pass B if the RMS over horizons of the strain bound is below
  0.01 % strain (100 microstrain) in every temperature group.

## Decision

- A and B pass: build the latent-state estimator; pre-register its
  cross-session test and the one-shot dataset0 test separately.
- Otherwise: stop the shared-state method direction on this corpus.
- Invalid inputs (empty, non-finite, zero noise, insufficient coverage) stop
  the run and are reported, never repaired.

## Known Limits

Retrospective corpus; dependent cycles; four temperature groups nested with
sessions; the oracle state in A uses labels; the bound in B is optimistic
(model known, white noise, smoothing horizon). Passing is necessary, not
sufficient.

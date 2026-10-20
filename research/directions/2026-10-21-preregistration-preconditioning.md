# Pre-registration: Preconditioning Cycle Before Calibration

Status: FROZEN by the git commit that adds this file together with
`research/probes/probe_preconditioning.py`, before that script is run. Any
later change needs a new dated pre-registration.
Branch: `direction/calibration-transfer-limits`. Parent note:
`2026-10-20-calibration-transfer-limits.md` (finding H3: the first loading
cycle of every session shows a higher apparent slope). Design reviewed by
Codex (round 7) before freezing.

## Idea

Force metrology already treats the first load cycles as non-representative:
force-transducer calibration practice includes preloading before the
calibration series (e.g. the ISO 376 procedure). This study asks whether
excluding one preconditioning cycle from calibration changes cross-session
strain estimation for this embedded resistive fiber sensor with a frozen
causal HGB readout. The practice itself is not new; the question is its
measured effect for this sensor and readout.

## Selection Limitation (fixed now)

This is a prespecified reanalysis following corpus-informed hypothesis
generation and extensive calibration-window exploration on the same corpus.
Freezing prevents further outcome-dependent changes; it does not create
untouched confirmation or an unbiased generalization estimate. The parent
note's exclusions stay: no mechanism (network conditioning, strain transfer,
gauge settling) is claimed, and results are bounded to one specimen and one
loading program.

## Fixed Definitions

- Calibration event: the first 250 s of a session. Only labels inside the
  event may be used by any calibration rule.
- First cycle end `c1`: time of the first strain trough after the first strain
  peak, found with `find_peaks(prominence=0.25, distance=600)` applied ONLY to
  the event prefix (t <= 250 s). Observed values before freezing (Codex):
  121.1-124.2 s in the eight evaluated sessions; first peaks at 62.2-62.4 s.
- Protocol failure (predefined): no peak or no following trough inside the
  prefix, or `c1 + 125 > 250`. A failure is reported, never repaired.
- P0 never invokes cycle detection.
- Four-level calibration, fusion, normalization, features, HGB
  hyperparameters, and frozen LOTO folds: unchanged from the repository.
  Calibration is computed on the full given window (no 125 s truncation).
- Only the eight sessions of dataset1-4 enter evaluation.

## Protocols (source and target sessions use the same rule unless stated)

| ID | Calibration window | Event used | Role |
|---|---|---|---|
| P0 | `[0, 125]` s | 125 s | repository baseline |
| P0d | `[0, 125]` s, event continues to 250 s | 250 s | duration-matched P0; predictions identical to P0 by construction |
| P1 | `[c1, c1 + 125]` s | 250 s | preconditioning cycle excluded, then calibrate |
| P2 | `[0, 250]` s | 250 s | same event, all data pooled (first cycle included) |
| P3 | `[125, 250]` s | 250 s | secondary: fixed-time skip, no cycle detection |
| P1t | target P1, sources P0 | 250 s | secondary: change target only (P1 also changes the fitted HGB) |

P1 versus P2 estimates the effect of retaining only the later window under this
calibration estimator; it does not isolate physical conditioning. Separating
cycling from elapsed time would need a matched rest/no-load condition that this
corpus lacks.

## Metrics And Decision Rule

Per direction `d`, `R_X,d` is HGB RMSE (percent strain) on the opposite
session. For comparator `Q`: mean RMSE `M_Q`, and mean relative improvement
`I_Q = (100 / 8) * sum_d (1 - R_P1,d / R_Q,d)`.

**Support this candidate calibration convention within the observed corpus**
only if ALL hold, for Q in {P0, P2}:

1. `M_P1 <= 0.95 * M_Q`;
2. `I_Q >= 5 %`;
3. `R_P1,d < R_Q,d` in at least 6 of 8 directions.

(P0d equals P0 in predictions, so the P0 conditions also cover the
duration-matched comparison.)

Outcomes:

- All conditions met: support as above.
- Conditions met against P0 but not P2: "No demonstrated advantage from
  exclusion over pooling." This does not establish that a longer event caused
  the benefit.
- Any other failed conjunction: "Decision criterion unmet."

Thresholds are practical conventions, not power or significance claims; the
eight directions are dependent.

## Mandatory Reporting

- Per direction and protocol: HGB and physics RMSE, worst relative
  deterioration of P1 versus each comparator.
- Per calibration window: start/end, strain extrema, the four level locations,
  rows per band, loading/unloading rows. Calibration content is not called
  "matched" across protocols.
- Secondary protocols P3 and P1t, never used to rescue a failed primary.

## Pre-scoring Leakage Check

Before any scoring, replace every label after 250 s in each calibration
session with corrupted values and require identical `c1`, calibration
parameters, and predictions for P1, P2, P3. If this fails, stop and report.

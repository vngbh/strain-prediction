# Research Index: Every Direction Ever Tried

Rule: nothing tried is deleted from reach. Every direction, including failed or
superseded ones, keeps a branch or an `archive/*` tag and an entry here, so it
can be reused, cited, or deliberately avoided. Read this before starting
anything new: several "new" analyses in October 2026 turned out to repeat July
2026 work (see Overlaps).

How to open an entry: `git checkout <branch-or-tag>` (read-only browsing), or
`git show <tag>:<path>` for a single file.

## Timeline

| Period | Where | Question | Status |
|---|---|---|---|
| 2026-04 | tag `archive/2026-04-dl-exploration` (`code/`, `history/`) | Which ML/DL model predicts strain best (ANN, LSTM, CNN-1D, CNN+LSTM, CWT-CNN, Optuna; cross-temperature LOO)? | Superseded. Several numbers invalid (see below). |
| 2026-07 | tag `archive/2026-07-diagnosis` (`research_diagnosis/`) | Why does cross-temperature generalization fail? Phases A-H2. | Superseded by the HGB protocol; many findings still valid. |
| 2026-09 | tag `archive/2026-09-hgb-development` (`src/physics_normalized_tree_boosting.py`) | Physics-normalized tree boosting, within-temperature development, dynamic-feature ablation. | Superseded by the frozen protocol. |
| 2026-10 | `main`, tag `baseline-frozen-hgb-2026-10-20` | Frozen LOTO HGB with fixed 125 s calibration (current thesis method). | Kept (backup baseline). |
| 2026-10 | branch `direction/calibration-transfer-limits` | What does short calibration limit, and what does the frozen model add? | Active findings + 6 rejected sub-ideas. |
| 2026-10 | branch `direction/latent-state-model` | Shared latent network state observable from two electrodes? | Rejected by pre-registered gates. |
| 2026-10 | branch `direction/reference-lag` | Is there a consistent apparent time alignment between the sensor and the reference gauge? | Completed diagnostic: v1 insufficient coverage, v2 consistent positive offsets; no mechanism. |
| 2026-10 | branch `direction/dynamic-gain-attribution` | Does re-aligning labels by the apparent offset contract the static-dynamic gap? | Completed diagnostic: partial contraction in all arms (F 0.21-0.38). |

## 2026-04: Deep-learning exploration (`archive/2026-04-dl-exploration`)

- Contents: `code/01_eda.py` ... `code/10_cwt_cnn.py`, `code/08_cross_temperature.py`
  (LOO ANN/LSTM), `code/09_hyperparameter_tuning.py` (Optuna), diagnostics,
  `code/docs/experiment_log.md`.
- Reusable: EDA scripts, data loader, the experiment log.
- Avoid / invalid (per the July audit ledger): Optuna `optuna_best_r2` is
  test-set leakage into hyperparameter search; LSTM/CNN windows splice across
  experiment or dataset boundaries (small for same-temperature, real for the
  LOO LSTM branch). Same-temperature R2 values (e.g. ANN 0.88, LSTM 0.83)
  measure an easier in-domain task, not transfer.
- Lesson: model shopping on 8 sessions produced numbers that later had to be
  invalidated; do not reopen without the user asking (`AGENTS.md`).

## 2026-07: Research diagnosis (`archive/2026-07-diagnosis`, `research_diagnosis/REPORT.md`)

Still-valid findings worth citing or reusing:

- B: temperature dominates raw resistance far more than strain.
- C: partial reproduction of the source paper.
- E: B(T) and S(T) are strictly monotonic; a full exponential `a e^(bT) + c`
  (the paper's form) gives the lowest LOO error: B 0.49 %, S 6.42 % (quadratic,
  linear, PCHIP much worse). Loading vs unloading S separation is not
  consistently justified (2/8 combinations).
- F: analytical inverse; Jacobian condition number median 12.1, max 7243,
  det(J) changes sign (only locally invertible); error decomposition: deployable
  error is dominated by B/S temperature-extrapolation error at 33 and 52 degC.
- G: calibration levels G0-G4; with any in-session calibration, electrode 2 is
  the more reliable electrode in every dataset, and a learned fusion weight puts
  nearly all weight on E2; equal-weight fusion is worse. Session-shift penalty
  flips sign between directions; 43 degC has the largest between-session
  baseline spread (E2 baseline std 5035 vs 201-1399) and the largest smooth
  same-signed drift; cause undetermined (no experiment records).
- H: fixed additive residual models on raw features rejected by a strict rule.
- H2: ANN on calibration-normalized features blended with physics passed a
  pre-registered rule (+16.4 % mean, worst +6.5 %); later superseded by HGB.

## 2026-09: HGB development (`archive/2026-09-hgb-development`)

- Contents: `src/physics_normalized_tree_boosting.py`, tree-boosting tables,
  dynamic-feature ablation.
- Avoid: the historical within-temperature 54 % improvement table must not be
  cited for final ablations (`AGENTS.md`).

## 2026-10: Frozen HGB protocol (`main`)

See `REPORT.md`, `results/tables/`, and `research/decisions.md`.

## 2026-10: Calibration-transfer limits (branch `direction/calibration-transfer-limits`)

Notes: `research/directions/2026-10-20-calibration-transfer-limits.md`,
`2026-10-21-preregistration-preconditioning.md`; probes in `research/probes/`.

- Kept: frozen HGB beats physics at every calibration position in 7/8
  directions (813 windows); dynamic beats static HGB likewise; elapsed time does
  not explain the dynamic gain; a loading-program control recovers about half
  of it; first-cycle apparent slope is higher (16/16, branch-balanced);
  "8/8 directions" depends on calibration bookkeeping (7/8 with measured band
  means).
- Rejected: label-free temperature-only calibration; frozen additive drift
  corrector from z1 - z2; a general calibration-window selection rule;
  calibration-diversity training; matching source/target calibration timing;
  excluding the first cycle before calibration (pre-registered, criterion
  unmet).

## 2026-10: Latent-state model (branch `direction/latent-state-model`)

Notes: `research/directions/2026-10-21-latent-state-model.md`,
`2026-10-21-preregistration-latent-gates.md`.

- Rejected: Gate A (transferable rank-one coupling) failed at 43 and 62 degC;
  Gate B (strain/state separability bound) passed only at 33 degC.
- Reusable: per-cycle coefficient table; residual coupling beyond time and
  program exists in all four groups (A2); two-electrode residuals from a
  branch-wise linear response are highly correlated (0.93-0.98) and
  autocorrelated; this correlation helps rather than hurts separability.
- Prior work occupying the broad idea: Wang 2021, Yong & Aw 2022, Amindehghan
  2021, Olowo 2023, Wu 2019 (DOIs in the notes).

## 2026-10: Reference alignment (branch `direction/reference-lag`)

Notes: `research/directions/2026-10-21-preregistration-reference-lag.md` (v1)
and `...-reference-lag-v2.md` (v2); REPORT section "Reference Alignment" on
that branch.

- Completed diagnostic: v1 (+/-5 s) returned insufficient coverage (13/16);
  data-informed v2 (+/-15 s; frozen in 81a4d86) resolved 16/16 and met the
  apparent-alignment rule. Original point estimates unchanged; no mechanism
  identified. Both verdicts and artifacts preserved. dataset0 remained sealed.
- Offsets 1.8-4.6 s in most sessions, 8.5-9.1 s at 62 degC experiment 10
  (no secondary collapse check possible there). Electrode offsets differ by
  0.4-0.6 s in 7/8 sessions.
- Avoid: reading harmonic-phase signs as evidence for or against a pure delay
  (a synthetic exact delay reproduces the observed sign pattern); claiming a
  temperature trend; calling the shift-only RMS an error floor.
- Permanent limit: one near-constant ramp rate, so delay and rate-independent
  branch asymmetry cannot be separated in this corpus.
- Follow-up done on branch `direction/dynamic-gain-attribution` (below).

## 2026-10: Static-dynamic gap under label re-alignment (branch `direction/dynamic-gain-attribution`)

Notes: `research/directions/2026-10-21-preregistration-dynamic-gain-attribution.md`.

- Completed diagnostic (`direction/dynamic-gain-attribution`, frozen
  `e8a4e61`): all four re-alignment arms yielded partial mean-gap contraction
  (F=0.206-0.380). Preserve preregistration, probe, and both output CSVs.
  Avoid interpreting F as a mechanistic attribution fraction or similar mean
  RMSE as alignment insensitivity. dataset0 remained sealed.
- Reusable: a re-implemented frozen protocol with explicit row masks
  (`probe_dynamic_gain_attribution.py`) that reproduces the repository RMSE
  exactly; useful for any future label or row manipulation.

## Overlaps Found (do not repeat)

| October analysis | Already done in | Difference |
|---|---|---|
| Fusion weight always picks electrode 2 (F3) | July Phase G (electrode-fusion calibration) | October shows it for the frozen protocol's four-level selector over 813 windows. |
| Label-free temperature-only calibration fails (E2) | July Phases E, F, G0 (exponential B(T), S(T)) | July used the better exponential form; both conclude it is not deployable. |
| 43 degC is drift- and session-shift-heavy | July Phase G session-shift analysis | October adds slow/fast residual decomposition. |
| Two-electrode identifiability | July Phase F Jacobian map | October gates test a latent state, not temperature. |

## Data Constraints (permanent)

- No new measurements exist or will exist (user, 2026-10-20).
- `dataset0.csv` (21 degC) is sealed as a one-shot confirmation set from
  2026-10-20 on the direction branches; earlier exposure: April EDA, July
  Phase E/F as a temperature point, one October probe.

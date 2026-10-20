# Decision Log

Append-only. Newest first. Each entry: date, decision, evidence (file or table),
status (`kept`, `rejected`, `superseded by <date>`). Read this before reviving
or changing a design choice. Dates before 2026-10-20 are reconstructed from git
history.

## 2026-10-21: Static-dynamic gap under label re-alignment (branch `direction/dynamic-gain-attribution`)

- Frozen `e8a4e61` after Codex rounds g1-g3. All four arms gave partial
  contraction of the mean static-dynamic RMSE gap (F 0.206-0.380); over half
  the reference gap remains under the intervention; strongly group-dependent.
- Not licensed: attribution fractions, dynamic-model insensitivity, mechanism.
- Status: completed diagnostic.

## 2026-10-20: Reference apparent alignment (branch `direction/reference-lag`)

- v1 (frozen `b82b769`, +/-5 s): insufficient coverage (13/16; three +5 s edge
  minima at 62 degC). v2 (data-informed sensitivity, frozen `81a4d86`,
  +/-15 s, stopping rule): 16/16 valid, consistent positive apparent alignment
  (1.8-9.1 s). Both verdicts kept; no mechanism inferred (one ramp rate makes
  delay and branch asymmetry indistinguishable).
- Electrode offsets differ consistently (7/8 sessions +0.36..+0.64 s), which
  challenges a shared-acquisition-offset-only explanation without excluding a
  shared contribution.
- Status: completed diagnostic.

## 2026-10-20: Latent-state direction stopped by pre-registered gates

- Branch `direction/latent-state-model`. Gates frozen in commit `052b2d0`
  after four Codex review rounds. Gate A (transferable coupling) failed at 43
  and 62 degC in raw form, passed as residual coupling everywhere; Gate B
  (strain/state separability bound) passed only at 33 degC.
- Decision: stop the shared-state method direction on this corpus.
- dataset0 remains sealed and unused.
- Status: rejected.

## 2026-10-20: Recommended research direction (branch `direction/calibration-transfer-limits`)

- Decision: proposed reframing to "calibration transfer limits" plus "static vs
  temporal information" (see `directions/2026-10-20-calibration-transfer-limits.md`).
- Open issue: under measured-band-mean calibration bookkeeping the final variant
  improves 7/8 directions (worst -5.5%), not 8/8. Not yet reflected in REPORT.md;
  needs a declared sensitivity analysis, not a silent substitution.
- Process: Claude vs Codex adversarial debate, evidence in `probes/`.
- Status: proposed. First analyses F1-F5 done the same day (see the direction
  file): no general calibration-window rule (kill criterion met for that
  sub-claim); fusion is degenerate (electrode 2 only); the program control
  explains about half of the dynamic gain but does not match it.
- Update (same day): calibration-diversity training rejected by a
  prespecified check (a source-selected single calibration offset is better,
  G5). REPORT now carries a Calibration Sensitivity section (bookkeeping,
  window position, static vs dynamic over positions) on this branch only.
- Update (same day): matching source/target calibration timing rejected
  (prespecified, 1/4 bins). First-cycle apparent sensitivity elevation kept as
  a descriptive calibration finding (16/16 under branch-balanced cycle bins);
  mechanism and novelty explicitly not claimed. Candidate literature still to
  be checked against full text.
- Update (same day): pre-registered preconditioning test (frozen in commit
  `303ccdb` before running) gave "Decision criterion unmet" (P1 vs P0: 4.7%
  mean per-direction improvement, 5/8, worst +30%). Excluding the first cycle
  is not adopted as a calibration protocol. Two compliance gaps disclosed and
  closed post hoc (prediction-level leakage check: exact 0; P0d rows).

## 2026-10-20: Split the single script into the `src/strain/` package

- Decision: one module per responsibility; calibration (label access) and
  features (time causality) stay separate because each owns one audited leakage
  boundary. Entry script remains `src/frozen_protocol_audit.py`.
- Evidence: all 8 CSVs and both PNGs byte-identical to the pre-refactor run;
  `tools/check_results.py` passes reproduction, audit-sensitivity, and docs gates.
- Process: Claude proposal, Codex adversarial review (2 rounds). Accepted from
  Codex: repo-root path bug, empty-dir regeneration, pinned versions, manifest,
  injected-leak negative controls, locator-based doc checks, glossary; in the
  review round, stored-output hashing, an inference-mutation control, and README
  prose locators. Rejected:
  merging calibration+features, separate runner module, pytest dependency,
  generated doc blocks.
- Status: kept.

## 2026-09-11: Frozen LOTO with fixed 125 s calibration is the final protocol

- Decision: HGB trained on non-target temperatures and frozen; target labels
  used only inside the fixed 125 s calibration prefix; direct strain target;
  causal features; no `T`.
- Evidence: `results/tables/frozen_protocol_ablations_summary.csv`,
  `causality_label_access_audit.csv`.
- Replaced the peak-detected "first cycle" calibration selector, which failed
  the calibration-scope audit (centered peak detector read past the boundary).
- Status: kept.

## 2026-09-10: Focus on physics-normalized tree boosting

- Decision: stop model shopping; one method (physical normalization + HGB with
  dynamic features).
- Evidence: commits `9cfcc1b`, `759de8b`.
- Status: kept.

## 2026-07-15: Reject residual learning; lock physics-based calibration

- Decision: residual-target learning rejected; physics calibration as the
  front end.
- Evidence: commits `66b8478`, `a6e7cf6`. Re-confirmed under the final protocol
  (residual-target HGB fails one direction in the ablation summary).
- Status: kept.

## 2026-04-21: ANN / CNN / LSTM / CNN+LSTM / Optuna exploration

- Decision: exploratory deep and hybrid models, within-temperature splits.
- Evidence: commits `8e9b5c0` to `5415540`.
- Status: superseded by 2026-09-10. Do not revive unless the user asks.

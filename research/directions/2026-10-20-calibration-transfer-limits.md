# Direction: Calibration Transfer Limits (recommended thesis core)

Status: proposed, 2026-10-20. Outcome of a Claude vs Codex adversarial debate
(3 rounds). Both sides converged on this direction; numbers below were
reproduced by both.

## Question

When does a short physical calibration support cross-session strain estimation
in this embedded resistive fiber sensor, and what does the frozen source-trained
model add that the short calibration cannot provide?

## Why The Current Framing Is Weak

- The data are a course-lab dataset (MECE4120 Lab 3 handed `dataset1-4.csv` to
  student groups), and the source paper (Caglar et al. 2021,
  doi:10.1002/admt.202000957, pp. 5-6) already did neural temperature-strain
  decoupling with 43 degC excluded from training. "ML on this sensor" and
  "prediction at a held-out temperature" are not contributions.
- Sessions are near-isothermal (within-session temperature range about
  0.03-0.14 degC). Dynamic temperature compensation is not testable.
- Temperature and session are nested; their effects cannot be separated.

## Evidence Collected (existing data only)

| ID | Finding | Value | Script |
|---|---|---|---|
| E1 | Calibration points use nominal band levels; the "zero" band actually averages about 0.010-0.015% strain. Using measured band means instead: | mean improvement 35.8%, worst -5.5%, 7/8 directions | `probes/probe_mechanism_ablations.py` (part A) |
| E2 | Label-free calibration from temperature alone (B, S interpolated in T, LOTO) | mean RMSE 1.18% strain: fails | `probes/probe_calibration_transfer.py` |
| E3 | Two-electrode affine map `a + b R1 + c R2`, fitted with ALL labels of the test session (oracle, not deployable) | mean RMSE 0.0365%, better than HGB 0.0484% | `probes/probe_affine_comparators.py` |
| E4 | Same affine map fitted on the 125 s prefix of the opposite session | mean RMSE 0.0879% (= physics 0.0881%); beats HGB in 0/8 | `probes/probe_affine_comparators.py` |
| E5 | HGB on fused signal only (no separate electrodes) | 29.6% mean, worst -30.0%, 6/8 | `probes/probe_mechanism_ablations.py` (part B1) |
| E6 | Frozen 1-parameter drift corrector `zf - k * mean(z1 - z2)` | -62% mean (Codex variant with opposite-session source transforms: -28%, 5/8): fails | `probes/probe_mechanism_ablations.py` (part C) |

Interpretation agreed by both sides:

- A good static two-electrode linear map exists (E3), but 125 s of one session
  does not contain enough information to estimate it for another session (E4).
- The frozen HGB recovers part of that calibration-estimation error using
  structure learned from source temperatures. This, not "hysteresis learning",
  is the mechanism the evidence currently supports.
- The headline "improves all 8 directions" is calibration-bookkeeping sensitive
  (E1). The average gain survives; the universal-consistency claim does not.

Contested (not settled): whether the dominant shift is session state or
temperature (Codex: unidentifiable because nested), and whether slow/fast
residual components can be called drift/hysteresis (Codex: no, components are
filtered and correlated; call them slow/fast).

## Planned Analyses

1. Fixed-window calibration distributions: slide the 125 s window over cycle
   starts of the calibration session, apply to the opposite session; report
   per session pair, never pooled as independent events.
2. Calibration content vs transfer error: strain coverage, branch balance,
   measured-mean vs nominal bookkeeping, window position (time since start).
3. Static vs temporal information: matched static two-electrode comparators,
   frozen static/dynamic ablations, and a source-derived cycle-template control
   to test whether "memory" is sensor state or the repeated loading program.
4. Electrode difference as a diagnostic only: source-fixed thresholds, false
   alarm and miss rates on opposite sessions.

## Results Of The First Analyses (2026-10-20)

Reviewed adversarially by Codex (round 3), except F5, which ran after that
review. Wording in quotes is the agreed thesis-safe phrasing.

| ID | Analysis | Result | Script |
|---|---|---|---|
| F1 | 125 s window slid every 10 s over the calibration session (813 windows, frozen HGB per group) | HGB beats physics at all windows in 7 folds and 98/109 in 33 degC B_to_A (min -4.0%). In that fold, median HGB RMSE (0.0505%) is WORSE than at the default window (0.0383%); the higher % improvement comes from physics degrading. | `probes/probe_window_distribution.py` |
| F2 | Window descriptors vs HGB RMSE | Later windows worse in 6 folds, better in 1 (43 degC A_to_B, rho -0.998 vs +0.999 for B_to_A), ~none in 1. Position, coverage and low-band mean are near-collinear within folds (abs rho up to 0.995). Kill criterion met: no general calibration-selection rule. | same |
| F3 | Fusion weight | Electrode-1 weight is 0 in all 813 windows and all 8 prefixes (unconstrained optimum -2.36..-1.22). The fused physics signal is exactly electrode 2. Legitimate constrained optimum, not a bug. | same |
| F4 | Static + elapsed session time | 9.7% mean, worst -57.8%, 5/8 (dynamic 37.6%). | `probes/probe_memory_vs_time.py` |
| F5 | Loading-program control (prespecified kill rule): static + causal branch, time since trough/peak, cycle count, last cycle length; no amplitude history | 24.3% mean, worst -42.7%, 7/8; RMSE ratio to dynamic 1.23, within 10% in 2/8: does NOT meet the kill rule. Program state recovers about half of the static-to-dynamic RMSE gap (0.0726 -> 0.0593 vs 0.0484). | `probes/probe_program_control.py` |

Agreed wording:

- F1: "Under nominal-level calibration, the frozen dynamic HGB outperformed
  matched physics across all sampled calibration positions in seven directions
  and 98/109 positions in the eighth. These dependent sensitivity evaluations
  demonstrate a broadly persistent relative advantage within this corpus,
  without establishing improved absolute accuracy at later positions."
- F2: "Later calibration windows were associated with higher HGB RMSE in six
  directions, lower RMSE in one, and little monotonic association in one.
  Strong within-session dependence among window position and content
  descriptors prevents causal attribution; no general calibration-selection
  rule was established."
- F3: "The constrained calibration-point selector assigned zero weight to
  electrode 1 throughout the evaluated windows, making the fused physics
  estimate identical to electrode 2. Electrode 1 remained available separately
  to HGB through its normalized signal and temporal features."
- F4: "Adding elapsed session time to static normalized inputs did not recover
  the dynamic model's performance under the tested HGB configuration. This
  supports the usefulness of the temporal representation, while leaving
  loading-program tracking and sensor-state mechanisms unresolved."
- F5 (Claude, not yet reviewed): "An explicit causal loading-program state
  recovered about half of the gap between static and dynamic inputs but did not
  match the dynamic model under the prespecified criterion. This rejects this
  program control, not every program-tracking explanation, and does not prove
  a sensor-memory mechanism."

### Round 2 of analyses (2026-10-20, branch `direction/calibration-transfer-limits`)

| ID | Analysis | Result | Script |
|---|---|---|---|
| G1 | Static vs dynamic HGB over all target calibration positions | Dynamic lower RMSE at every position in 7 directions, 96/109 in 33 degC B_to_A. Mean of per-direction medians: physics 0.0879, static 0.0717, dynamic 0.0496. Written into REPORT. | `probes/probe_calibration_shift.py` |
| G2 | Calibration-diversity training (sources normalized at 0/240/480/720 s, stacked) | Mean of medians 0.0448 vs 0.0496; spread (std, IQR) NOT reduced; 43 degC A_to_B worse; 33 degC A_to_B wins only 51/98 positions. | same |
| G3 | Row-count control (0 s data replicated 4x) | 0.0498: identical-row duplication does not reproduce G2. | `probes/probe_diversity_volume_control.py` |
| G4 | Source calibrated at the same offset as the target window | 0.0453 vs 0.0486 (0 s-trained) vs 0.0452 (diverse) over offsets 0/240/480/720. | `probes/probe_calibration_shift.py` |
| G5 | Decisive check (prespecified with Codex): diversity vs ONE source offset chosen by inner LOTO on source temperatures only, row-matched | Inner validation chose 480 s in all 4 outer folds (within a 4-point grid; at 43 degC its inner margin over 240 s is ~0.0003). Primary metric: selected 0.0436 vs diverse 0.0448 (repository policy), 0.0435 vs 0.0448 (fixed 150 iterations, no validation hold-out); early termination never occurred (n_iter 150). **Kill rule met: diversity accuracy claim dropped.** Diversity does have lower mean IQR across positions (0.0126 vs 0.0139), so a spread benefit is not excluded. Inner selection used 2 training temperatures and 1 copy, outer fits 3 temperatures and 4 copies. | `probes/probe_diversity_vs_selected_offset.py` |
| G6 | The selected 480 s model at the protocol's 0 s target window | Mean 0.0468 vs 0.0484; better in 5/8 directions, clearly worse at 33 degC A_to_B (0.0541 vs 0.0389). | same (+ shift CSV) |

What survives:

- G1 (dynamic over static, across calibration positions) is a finding and is
  in REPORT.
- Calibration diversity is NOT a method contribution (G5).
- Exploratory sensitivity evidence, not an unbiased improvement estimate: the
  source calibration convention (when source sessions are calibrated) is a
  design variable. Choosing it by source-only inner validation lowered the
  mean of per-direction medians over target positions from 0.0496 to 0.0436
  (better median in 7/8 directions), but the candidate grid followed F1/F2
  inspection of this corpus, the baseline used one training copy versus four,
  and the advantage at the 0 s protocol window is mixed (G6).

Codex's pre-registered reading for the timing probe (written before results):
thesis-worthy only if H_match holds in >= 3/4 bins AND matching advantages are
consistent in >= 3 temperature groups; an actionable selection claim needs
>= 5% lower primary RMSE than a source-only selected fixed offset. One offset
winning every bin rejects matching; tiny or reversing effects make timing a
dead end. Either way timing is not identified causally (window content is
confounded).

Still open: whether the convention effect is "match the source calibration
timing to the deployment timing distribution" (testable by evaluating 0
s-trained vs 480 s-trained models separately on early vs late target windows),
and a Codex review of G5/G6 wording.

### Round 3 of analyses: source calibration timing and the first cycle

| ID | Analysis | Result | Script |
|---|---|---|---|
| H1 | Prespecified H_match vs H_mid: one model per single source offset, target calibration positions binned | H_match 1/4 bins: rejected. H_mid also fails literally (no single winner). Source offsets 240-480 s generally best; 0 s source wins some individual cells (33 and 43 degC A_to_B early windows). Bins are calibration positions, not test times. | `probes/probe_source_timing_match.py` |
| H2 | Does F2 ("later calibration windows worse") depend on the source convention? | Under 480 s sources, 4 directions positive and 4 negative position-RMSE correlation; 43 degC directions keep opposite signs under every source offset. F2 holds for the 0 s convention and depends partly on source preprocessing. | same |
| H3 | First-cycle apparent sensitivity | First cycle higher in 16/16 electrode-sessions under OLS (+1.6..+13.8%) and branch-balanced cycle bins over 0.20-0.45 % (+2.4..+14.4%); faster initial decline 13/16 (cycle bins), 14/16 (OLS); with a linear time term 12/16 higher, 4/16 faster decline. Codex additionally found 15-16/16 for band-mean, loading-only, unloading-only, and fixed-bin estimators. Temperature settling (<=0.1 degC, inconsistent sign) and strain-range change (<4%) do not explain it. | `probes/probe_first_window.py`, `probes/probe_first_cycle_estimators.py` |

Agreed wording (Codex round 6): "Across eight sessions from one embedded
sensor/specimen, initial-cycle apparent resistance-reference-strain slopes
exceeded later-cycle slopes under several estimators, including
branch-balanced fitting over a common strain interval." Unsupported: a named
mechanism (network conditioning, after-rest recovery, Mullins softening),
that it explains the history-feature gain, and any generalization beyond this
specimen. Changing strain transfer to the surface gauge and gauge/adhesive
settling are not excluded.

Literature status: Montazerian et al. 2019 (Compos. B 164:648, CF/PDMS
yarns) and Mu et al. 2023 (Polymers 15:3379, filler dimensionality and
resistive viscoelasticity) exist (Crossref-verified titles). Codex states they
report first-cycle overshoot / cycling attenuation; Claude could not confirm
this from the abstracts. The student must check the full texts before citing
them for that specific claim.

## Kill Criteria

- If relationships between calibration content and transfer error reverse
  across session pairs, drop the predictive claim and keep only a descriptive
  finding about calibration fragility.
- If a cycle-template control matches HGB, drop any claim that history encodes
  transferable sensor memory.

## Scope Boundary

One specimen, near-isothermal sessions, one repetitive loading program. Other
sensors, thermal transients, loading rates, and random load histories need new
measurements and are permanently outside this thesis.

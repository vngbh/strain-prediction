# Research Diagnosis — Why Does Cross-Temperature Generalization Fail?

Scope: this project has been consolidated onto a single direction. The original multi-model
deep-learning codebase (`code/`) and an earlier abandoned diagnostic framing (`history/`) have been
deleted from the working tree to keep the project to one direction only — both are still recoverable
from git history (commits up to `889da86`) if ever needed again. `research_diagnosis/` is now the
entire active project.

Reproducibility: seeds `[0, 42, 123]`; package versions in `logs/package_versions.txt`; all scaler
fitting is train-split-only in every script under `scripts/`.

Status: **Phase A–H2 complete.** Phase D-fix corrected/re-confirmed the same-temperature and LOO
baseline comparison under an explicit "Task A: nowcasting" label with provenance-tracked windowing;
Phase E identified B(T)/S(T); Phase F built and evaluated the analytical inverse baseline (v2,
corrected after review); Phase G ran the calibration-level study (G0-G4) and found a clear, deployable
answer; Phase H tested a fixed additive residual model against a strict per-direction keep/drop rule
and **rejected it**; Phase H2 then tested a different combination (ANN on calibration-normalized
features, blended with the physics prediction via a calibration-validated weight) against an even
stricter rule and **it passed** — the current best-supported method combines the Phase G calibrated
physical inverse with this calibrated-ANN blend. This is a
checkpoint, not a final decision — see "Current Status / Next Decision" at the end of this report.

### Authoritative-results ledger (read this before citing any number for the thesis)

Per your instruction, corrected diagnostic results are authoritative for final thesis conclusions.
Historical repo results affected by the confirmed bugs below are marked **INVALID FOR FINAL
COMPARISON** and must not be cited as generalization numbers without the caveat attached:

| Result | Status | Reason |
|---|---|---|
| `code/results/phase9_optuna/optuna_results.json` (`optuna_best_r2` column) | **INVALID** | Test-set leakage into HPO (Phase A bug #1). `final_r2_mean` in the same file is a re-trained number and is *not* subject to this specific leak, but still inherits phase 9's temporal-70/30 split (see next row). |
| `code/results/phase5-7,10` LSTM/CNN/CWT same-temperature R² (temporal 70/30 split) | **CAVEAT, not fully invalid** | A small fraction (~0.12-0.13%) of training windows splice across an internal experiment boundary (Phase A bug #2); numerically negligible but the split itself is not a clean per-experiment holdout — see Phase A Q12. |
| `code/results/phase8_cross_temp/cross_temp_metrics.json` (LOO ANN/LSTM) | **CAVEAT, not fully invalid** | LSTM branch windows splice across dataset/temperature boundaries in `load_combined()` (Phase A bug #3); ANN branch (flat features) is not affected by this specific bug. Both branches still use the repo's own (uncorrected) split protocol, superseded below. |
| `research_diagnosis/tables/simple_baseline_results.csv` (original Phase D, this directory) | **SUPERSEDED, not invalid** | Correct at the time (strict split, no windowing), but superseded by `simple_baseline_results_corrected.csv` which adds explicit Task-A/nowcasting labeling and provenance. Numbers are consistent between the two (e.g. Linear+T @43C: 0.566 in both). |
| `research_diagnosis/tables/simple_baseline_results_corrected.csv` | **AUTHORITATIVE** | Strict experiment-block split, Task A (nowcasting), no leakage found. |
| `research_diagnosis/tables/analytical_inverse_results.csv` (v2) | **AUTHORITATIVE** | Phase F mandatory comparison, identical test rows across all methods; v1 (F2 clipping bug, merged oracle, post-hoc electrode pick) is superseded in place — see "Analytical Inverse Baseline (Phase F, v2 — corrected after review)". |
| `research_diagnosis/tables/B_temperature_model_LOO.csv`, `S_temperature_model_LOO.csv` | **AUTHORITATIVE** | Phase E LOO backtest, bootstrap-by-cycle. |
| `research_diagnosis/tables/calibration_results_all.csv`, `calibration_summary.csv` | **AUTHORITATIVE** | Phase G calibration-level study (G0-G4), both calibration directions reported, all selection done on calibration-internal data only. |
| `research_diagnosis/tables/residual_per_direction_rmse.csv`, `residual_decision_rule.csv` (v2) | **AUTHORITATIVE, superseded conclusion** | Phase H final decision run on fixed additive residual models. Decision: DROP (see Phase H2 for the combination that passed instead). v1 (mean-only reporting, `cycle_idx` leakage bug) is superseded in place. |
| `research_diagnosis/tables/calibrated_ann_per_direction_rmse.csv`, `calibrated_ann_decision_rule.csv` | **AUTHORITATIVE — current final decision** | Phase H2 — calibrated-feature ANN blended with physics via a calibration-selected weight. Decision: KEEP (blended variant only, not the ANN alone). |

---

## A. Pipeline Audit

Full 15-question table: `tables/pipeline_audit.csv` (unchanged from the prior pass).

**Confirmed bugs**
1. **Phase 9 (Optuna) leaks the test set into hyperparameter search** — `code/09_hyperparameter_tuning.py:110-121`, the Optuna objective directly maximizes held-out test R² across 50 trials (separate from the val-loss-based early stopping inside each trial). Explains why `optuna_best_r2` (e.g. 0.933 for ds1) is consistently higher than the re-trained `final_r2_mean` (0.79).
2. **Sequence windows cross an internal experiment boundary sitting inside the training region.** Every `dataset{1-4}.csv` contains 2 experiments back-to-back; the boundary is always before the repo's 70% split point (e.g. ds1: boundary row 10,996 vs split row 16,137). `make_sequences` (phases 5,6,7, and CWT phase 10) slides across this boundary, splicing unrelated experiment tails/heads into a minority of training sequences.
3. **Phase 8 LOO/extrapolation splices windows across different temperatures** — `load_combined()` concatenates whole datasets from different temperatures before windowing.

**Design inconsistencies (not leakage, but affect comparability)**
4. ANN predicts `y[i]` from `X[i]` (contemporaneous); all sequence models predict `y[i]` from `X[i-window:i]` (excludes the same-instant reading) — undocumented, affects "ANN vs LSTM" comparability.
5. `KFold(shuffle=False)` in `diagnostics.py` trains on chronologically later data to predict earlier data in some folds.

**Verified-correct**: no `shuffle=True` anywhere; scaler fit train-only everywhere except one explicitly-documented exception (`compare_baseline_vs_phase4.py`, intentional mentor-baseline replication); target `Strain` never scaled (no inverse-transform question); early stopping monitors `val_loss`, never test; row order within each CSV is already correct (2 contiguous experiment blocks, `Idx` monotonic).

None of bugs #1-3 are large enough on their own to explain a catastrophic negative R² at LOO — they corrupt a small minority of windows / only affect Phase 9. **Phases C and D below independently confirm this**: the interpolation failure reproduces even in a from-scratch paper-style model and in classical (non-neural) baselines under a clean strict split, so it is not merely an artifact of these three bugs.

---

## B. Data Mechanism Diagnosis

Script: `scripts/phase_b_mechanism.py`. Granular per (dataset, experiment, cycle, electrode) table:
`tables/mechanism_statistics.csv` (156 rows, ~10 cycles/experiment x 2 experiments x 4 datasets x 2 electrodes).
Aggregates: `tables/mechanism_aggregated_by_temperature.csv`, `..._between_experiment_variability.csv`,
`..._drift_first_vs_last_cycle.csv`, `..._43C_between_check.csv`.

**Method note (deviation from literal brief):** raw sample-to-sample `dStrain/dt` is dominated by
measurement noise (~3000 sign changes per ~11000 samples), so a literal derivative-sign threshold
cannot recover real loading/unloading branches. Turning points are instead found on a Savitzky-Golay
smoothed strain signal via `scipy.signal.find_peaks` (prominence=0.05), which cleanly recovers ~10
cycles per experiment (visually confirmed). `loading_flag` is assigned from segments between
consecutive turning points, with a small dwell margin (3%) around each turning point flagged 0.

**Key findings:**

- **Resistance level is completely dominated by temperature, not strain.** `figures/R_vs_strain_all_temperatures_E1.png` / `_E2.png`: each temperature forms a near-flat horizontal band (E1: ~730k @33C -> ~1.41M @62C); the within-band spread from strain (full 0->0.5 range) is only ~1-3% of the baseline level. Quantitatively, `S_loading_mean` (dR/dstrain) ranges 11,068 (33C) to 64,955 (62C) for E1, against a baseline `B_mean` of 732k to 1.41M — i.e. even a full strain sweep moves resistance far less than temperature does. **This is the central mechanism**: any small error in temperature-compensation swamps the actual strain signal, which is why cross-temperature generalization is inherently harder than cross-temperature classification-style tasks.
- **B(T) and S(T) are smooth, monotonic, and well-behaved** (`figures/baseline_B_vs_temperature.png`, `sensitivity_S_vs_temperature.png`): both increase monotonically and convexly with T for both electrodes, consistent with the paper's exponential form.
- **43C lies cleanly between 33C and 52C**, confirmed at every granularity checked (dataset mean, and separately for each of the 2 experiments per dataset; `tables/mechanism_43C_between_check.csv` — `ds2_between_ds1_ds3 = True` in all 8 rows checked, both electrodes, both B and S_loading). **This directly answers Phase B's central diagnostic question: there is no geometric/physical anomaly at 43C** — it is not an outlier session, not a baseline-shifted dataset, not non-monotonic. Hypothesis **H5 (preprocessing/alignment/model issue) or a modeling-capacity issue is favored over H1-H4** (ordinary drift, sensitivity change, hysteresis change, session shift are all present as smooth, well-behaved trends, not sudden anomalies at 43C specifically).
- Hysteresis area also increases monotonically with temperature (`hysteresis_area_vs_temperature.png`), both electrodes.

---

## C. Original Paper Reproduction

Script: `scripts/phase_c_paper_reproduction.py`. Full results: `tables/paper_reproduction_results.csv`.
Architecture: 3 hidden Dense(tanh) + Dense(2, tanh) output, inputs (E1,E2) and outputs (Temperature,
Strain) both scaled to [-1,1] with train-only statistics. Trained on ds1+ds3+ds4 (33/52/62C); ds2 (43C)
held out entirely as the unseen-temperature test, matching the paper's stated protocol.

| Variant | within-pool test Strain R2 | within-pool test Temp R2 | **ds2 (43C) unseen Strain R2** | ds2 unseen Temp R2 |
|---|---|---|---|---|
| C1 random split | 0.903 | 1.000 | **-2.502** | -75,355 |
| C1 random split, 1/20 subsample | 0.870 | 1.000 | **-2.490** | -77,925 |
| C2 strict experiment-block split | 0.740 | -1985 (see note) | **-2.468** | -77,431 |

Note on Temperature R2: within a single held-out experiment/block, Temperature varies over only a
fraction of a degree, so its variance denominator in R2 is near-zero — a tiny absolute error produces
an enormous negative R2. This is a metric artifact, not evidence the model is unusually bad at
temperature; RMSE (in `paper_reproduction_results.csv`) is the meaningful number there.

**Conclusion: the paper's own architecture, faithfully reproduced (tanh, [-1,1] scaling, joint
T+Strain output), fails at 43C just as badly as the existing repo pipeline** (Strain R2 ~ -2.5,
consistent across random/strict split and with/without subsampling). `figures/paper_reproduction_C1_random_split_ds2_43C.png`
shows the failure mode visually: the model's Strain prediction **collapses to a near-constant value**
(~0.51, saturated) instead of tracking the true sawtooth strain pattern — a sign of tanh-layer
saturation / out-of-distribution collapse at the untrained 43C resistance level, not random noise.
This rules out "it's specific to this repo's code" as the explanation — the failure is architectural/
methodological, reproduced from scratch.

---

## D. Simple Baselines Before Deep Learning (superseded — see Phase D-fix below)

The original multi-model horse race (linear / ridge / HistGBR / Deep ANN, D1 same-temperature + D2 LOO)
has been archived to `history/phase_d_superseded_by_dfix/` to keep this project on a single, current
direction. Its one lasting finding — **a plain linear regression on (R1, R2, Temperature) reaches
R2=0.57 at 43C interpolation while Deep ANN scores -4.2 on the same strict split, proving the 43C
failure is a neural-network conditioning problem, not a data/domain impossibility** — is fully
re-confirmed with identical numbers and additional provenance/seed-stability detail in "Corrected Fair
Comparison (Phase D-fix)" immediately below. Do not cite the archived tables for the thesis; use
`tables/simple_baseline_results_corrected.csv` instead (see the authoritative-results ledger above).

---

## Corrected Fair Comparison (Phase D-fix)

Scripts: `scripts/windowing.py` (provenance-tracked window builder), `scripts/phase_dfix_windows.py`,
`scripts/phase_dfix_corrected_baselines.py`.

**Window construction fix.** `tables/window_provenance.csv` records every corrected window's source
dataset/Experiment/start/target index; windows are built independently per Experiment then
concatenated, so none cross an Experiment or dataset boundary by construction (90,694 valid windows
across ds1-4, window=20). `tables/window_corruption_old_approach.csv` quantifies what the *old* repo
approach actually corrupted: exactly 1 experiment boundary sits inside the old temporal-70% train
region per dataset, corrupting 20 windows each (~0.123-0.133% of that dataset's windows) — small in
count, confirming Phase A's bug #2 was real but minor in scale.

**Task separation.** All models compared below use Task A (nowcasting: `X[t] -> y[t]`), matching what
the original (uncorrected) Phase D already did — that script never built sequence windows, so it was
never affected by the window-crossing bug; this rerun re-confirms the same numbers under an explicit
label and produces the officially authoritative artifact.

**D1 — same-temperature, strict experiment-block holdout** (train on one whole Experiment, test on the
other; `r2_mean` over 3 seeds):

| Dataset | linear R1+R2 | linear R1+R2+T | ridge+interactions | Deep ANN |
|---|---|---|---|---|
| ds1 (33C) | 0.914 | 0.902 | 0.922 | 0.906 |
| ds2 (43C) | 0.925 | 0.903 | 0.909 | 0.774 |
| ds3 (52C) | 0.711 | 0.563 | 0.635 | 0.867 |
| ds4 (62C) | 0.749 | 0.756 | 0.762 | 0.765 |

**D2 — leave-one-temperature-out** (corrected, authoritative):

| Test dataset | kind | linear R1+R2 | **linear R1+R2+T** | ridge+interactions | Deep ANN (r2_std across seeds) |
|---|---|---|---|---|---|
| ds1 (33C) | extrapolation | -1.957 | -0.693 | -1.692 | -13.54 (std 13.82) |
| ds2 (43C) | interpolation | -0.002 | **+0.566** | +0.294 | -10.69 (std 5.39) |
| ds3 (52C) | interpolation | -0.025 | **+0.497** | +0.049 | -29.66 (std 16.75) |
| ds4 (62C) | extrapolation | -3.255 | -13.86 | -35.39 | -30.47 (std 33.95) |

**Confirmed (item 6): Linear+T still gives a positive R2 at 43C after all fixes (0.566) and at 52C
(0.497)** — identical to the pre-fix Phase D numbers, as expected since Phase D was never affected by
the windowing bug. New observation from the 3-seed std column: **Deep ANN's own LOO R2 is wildly
unstable across random seeds** (e.g. std=33.9 on ds4, larger than most of its mean values) — it isn't
just failing, it's failing *unpredictably*, which is itself evidence against relying on it as a
generalization baseline. `figures/loo_predictions_43C_corrected.png` shows the 4 models' predictions
at 43C side by side.

---

## Data Mechanism / B(T), S(T) Identification (Phase E)

Script: `scripts/phase_e_BT_ST.py`. Per-cycle table: `tables/B_S_by_cycle.csv` (156 rows). Bootstrap
(500 resamples, resampled by **cycle**, never by row): `tables/B_S_aggregated_bootstrap.csv`. Threshold
sensitivity: `tables/B_threshold_sensitivity.csv`. LOO backtests: `tables/B_temperature_model_LOO.csv`,
`tables/S_temperature_model_LOO.csv`. Loading/unloading separation: `tables/loading_unloading_separation_check.csv`.

**B(T) threshold sensitivity:** varying the zero-strain window from 0.01 to 0.10 strain changes the
estimated B by <0.3% for every dataset/electrode (e.g. ds1 E1: 732,153 at thr=0.01 vs 732,325 at
thr=0.10) — **B is not sensitive to this choice** in this dataset.

**Candidate temperature functions, LOO mean absolute relative error (fit on 3 temps, predict the 4th):**

| Candidate | B(T) LOO error | S(T) LOO error |
|---|---|---|
| **full exponential (a·e^(bT)+c)** | **0.49%** | **6.42%** |
| quadratic | 5.12% | 19.86% |
| reduced exponential (a·e^(bT)) | 8.78% | 7.71% |
| monotonic PCHIP | 6.60% | 25.66% |
| linear | 13.12% | 44.96% |
| constant | 21.55% | 84.27% |

**The full 3-parameter exponential — despite being exactly determined on only 3 training temperatures
(zero residual degrees of freedom, so it cannot be internally cross-checked during fitting) —
empirically gives the lowest LOO backtest error of every candidate, for both B and S, by a wide
margin.** This is a genuine, non-obvious result: the *theoretical* identifiability concern is real
(no internal goodness-of-fit signal is available when fitting 3 params to 3 points), but the
*practical* concern (does it extrapolate/interpolate accurately to a held-out temperature?) is not
borne out here — the functional form itself is evidently a very good match to this sensor's physics,
which is what makes the exactly-determined fit still generalize well. This validates the original
paper's choice of an exponential B(T)/S(T) form. Simpler forms (linear, constant) are cleanly worse and
markedly more stable/conservative, but materially less accurate.

**Monotonicity:** B(T) and S(T) are both strictly monotonic increasing for both electrodes across all
4 temperatures (`tables/monotonicity_check.csv`).

**43C between 33C/52C:** reconfirmed with this phase's bootstrap-CI values — `ds2_between=True` for
both B and S_loading, both electrodes (`tables/phaseE_43C_between_check.csv`), with tight, non-overlapping
95% CIs across temperatures (e.g. E1 B: 33C=[731.8k,732.5k], 43C=[822.4k,823.9k], 52C=[998.2k,999.0k] —
strict, well-separated ordering).

**Loading vs. unloading sensitivity separation:** mixed evidence — only 2 of 8 (dataset x electrode)
combinations show non-overlapping 95% CIs between S_loading and S_unloading (`ds2/E2`, `ds3/E1`); the
other 6 overlap. **Conclusion: a common S(T) is a reasonable default; separate loading/unloading S is
not consistently justified by this data** (it may matter for specific dataset/electrode combinations,
but not as a blanket rule) — `figures/loading_unloading_sensitivity.png`.

---

## Analytical Inverse Baseline (Phase F, v2 — corrected after review)

Script: `scripts/phase_f_analytical_inverse.py`. This section supersedes an earlier v1 pass; three
correctness/rigor problems were found on review and fixed (script docstring has full detail):

1. **F2 RMSE=29.9 bug (real bug, confirmed).** The `[-0.05, 0.55]` epsilon constraint only stopped bad
   candidates from *winning* the grid search — rows where every candidate was invalid still returned an
   unclipped garbage value from whatever sat at the grid's first index. Fixed with a hard clip on the
   final output plus an explicit `search_failed_rate` per fold (now in `analytical_inverse_results.csv`).
2. **"Oracle" was not a true oracle.** It calibrated B/S on the held-out dataset's first Experiment and
   tested on its second — still subject to Experiment-to-Experiment session shift, not pure forward-model
   error. Renamed to `session_calibrated_inverse`, and a genuine **`true_oracle_within_experiment`** was
   added: calibrates and tests on even/odd cycle halves of the *same* Experiment, isolating forward-model
   error with zero cross-Experiment or cross-temperature error.
3. **E1-only "preference" was post-hoc, i.e. model-selection leakage.** Replaced with
   `nested_select_variant()`: for each real LOO fold, each of the 3 training datasets is held out once
   *within the training pool* (fit on the other 2, evaluate on it) to score every variant using only
   training-side information; the lowest-nested-RMSE variant is applied to the real test set. The full
   post-hoc variant table is kept for reference but explicitly labeled diagnostic-only.

### F1 — known-T inverse, mandatory comparison (train-only selected variant)

| Held-out | kind | Linear+T | Deep ANN (+T) | Inverse (known-T, **train-selected variant**) | Inverse (unknown-T) | Session-calibrated | **True oracle** |
|---|---|---|---|---|---|---|---|
| ds1 33C | extrapolation | -0.730 | -494.7 | -22.06 (selected: E2_only) | -3.23 | 0.264 | **0.726** |
| ds2 43C | interpolation | **0.756** | -6.73 | 0.275 (selected: common_S) | -1.96 | -0.826 | **0.531** |
| ds3 52C | interpolation | **0.707** | -6.07 | -7.28 (selected: both_weighted) | -1.46 | 0.705 | **0.630** |
| ds4 62C | extrapolation | -13.95 | -30.89 | -1.55 (selected: E2_only) | -2.00 | -0.292 | **0.232** |

**The true oracle is positive and reasonably consistent in all 4 folds (0.23-0.73)** — a materially
cleaner and more interpretable signal than the earlier (session-contaminated) "oracle" numbers. This
confirms the reviewer's point directly: mixing in session shift was hiding how good the forward model
`R = B(T) + S(T)*epsilon` actually is at the *same* temperature.

**The train-only-selected variant performs far worse than the post-hoc-best variant would have** —
e.g. ds1: nested selection picks E2_only (real test R2 = -22.06), while the post-hoc-best E1_only
would have scored +0.906 (see diagnostic table below and `analytical_inverse_variant_selection_nested.csv`).
**This is itself an important, honest finding, not a disappointing implementation detail**: it shows
the apparent "E1 is better" pattern from the earlier pass was very likely overfitting to the specific
test set, and a principled train-only selection criterion does not currently recover that benefit —
electrode reliability at a genuinely new temperature does not transfer cleanly from cross-validation
within the known training temperatures (only 3 points, same sparsity problem as B(T)/S(T) extrapolation
itself). **Electrode selection should not be presented as a solved, reliable improvement at this
checkpoint.**

**Post-hoc diagnostic only (`analytical_inverse_F1_variants.csv`, NOT used for any decision):**

| Held-out | E1 only | E2 only | both-weighted | common-S |
|---|---|---|---|---|
| ds1 33C | 0.906 | -22.06 | -10.32 | -10.12 |
| ds2 43C | 0.806 | -0.072 | 0.255 | 0.275 |
| ds3 52C | -9.41 | -6.54 | -7.28 | -7.27 |
| ds4 62C | -10.78 | -1.55 | -0.027 | 0.070 |

### F2 — unknown-T joint inverse (fixed)

After the clipping fix, `search_failed_rate` (fraction of test rows where every grid candidate was
physically inadmissible) is 0% for ds1-ds3 and 13.5% for ds4 — confirming the fix targeted a real,
localized failure mode rather than masking a universal one. Recovered-strain R2 is still uniformly
negative (-1.46 to -3.23), now without the earlier nonsensical RMSE=29.9 outlier (ds4 RMSE is now a
physically sane 0.239). **Conclusion unchanged: joint T-strain estimation is not currently reliable**,
consistent with the Jacobian findings below — this is a genuine limitation, not an artifact of the fixed
bug.

### Jacobian / identifiability

Unchanged from v1 (uses the B(T)/S(T) fit only, not affected by the F1/F2/oracle fixes):
`tables/jacobian_condition_map.csv`, `figures/jacobian_condition_map.png`.

- **Condition number: median 12.1, min 5.3, max 7243.** Mildly ill-conditioned typically, with pockets
  of severe (>1000x) ill-conditioning at specific (T, strain) regions.
- **det(J) changes sign across the grid** — the system is invertible only *locally*, not globally;
  confirms the reviewer's "two equations, two unknowns is not sufficient" caution and explains F2's
  residual instability even after the clipping fix.

### Error decomposition (now three-way, per reviewer's point #2)

`tables/inverse_error_decomposition.csv`, `figures/inverse_error_decomposition.png` — separates:
**true-oracle RMSE** (forward-model form only) -> **session-calibrated RMSE** (+ Experiment-to-Experiment
session shift) -> **deployable RMSE** (+ B(T)/S(T) LOO-prediction error at the unseen temperature):

| Held-out | True oracle RMSE | Session-calibrated RMSE | Extra from session shift | Deployable RMSE | Extra from B/S LOO error |
|---|---|---|---|---|---|
| ds1 33C | 0.074 | 0.122 | +0.048 | 0.680 | **+0.559** |
| ds2 43C | 0.101 | 0.200 | **+0.099** | 0.126 | -0.074 |
| ds3 52C | 0.089 | 0.080 | -0.010 | 0.423 | **+0.343** |
| ds4 62C | 0.121 | 0.157 | +0.036 | 0.220 | +0.063 |

Reading this properly (unlike the merged v1 version): the **true-oracle RMSE is small and consistent
across all 4 folds (0.07-0.12)** — the static forward model itself is reasonably sound at a *known*
temperature with *no* session mismatch. **Session shift adds a real but generally modest amount**
(+0.01 to +0.10) except it can occasionally net negative (ds3, likely calibration-subset sampling
noise from the cycle-based split — small-sample caveat). **The dominant, large error source for the
hardest folds (ds1, ds3) is unambiguously B(T)/S(T) LOO-prediction error at the unseen temperature**
(+0.56, +0.34) — i.e. the deployable inverse's failure is real and comes from extrapolating B/S with
only 3 training temperatures, not from the forward-model shape or from session variation. This is a
cleaner, more defensible decomposition than v1's, and still motivates residual learning / calibration
as the productive next steps rather than a different forward-model form.

---

## Phase G — Minimal Calibration for Deployment

Scripts: `scripts/phase_g_calibration.py`, `scripts/phase_g_final_figure.py`. Model:
`R_i = B_i + S_i*epsilon + residual`, Temperature assumed known (no joint T-epsilon estimation here).
For every dataset both calibration directions are run (calibrate on Experiment A / test on B, and the
reverse) and reported separately before averaging. No electrode, calibration-level, `epsilon_ref`, or
fusion-weight choice ever uses the real test Experiment — all such choices are made on an internal
calib/validation split carved out of the calibration Experiment itself (first half of its cycles vs
second half).

**Levels evaluated:** G0 (zero target-domain data, B/S predicted from the other 3 temperatures via the
Phase E exponential fit — identical setup to Phase F's known-T inverse) -> G1 (zero-balance: B measured
directly from near-zero-strain samples in the calibration Experiment; budgets: first stable segment,
10, 50, 100, all available) -> G2 (zero-balance B + one reference-strain-point S, level chosen
internally) -> G3 (zero + {2,3,5} reference points, linear fit) -> G4 (one full loading-unloading
cycle: common S and branch-split S, same-session vs cross-session).

### Calibration curve (mean R2/RMSE across all 4 datasets x 2 directions, best electrode-fusion variant per level)

| Level | Mean R2 | Mean RMSE |
|---|---|---|
| G0 (zero calibration) | -1.445 | 0.181 |
| G1 zero-balance, first segment | 0.333 | 0.095 |
| G1 zero-balance, n=10 | 0.334 | 0.095 |
| G1 zero-balance, n=50 | 0.372 | 0.095 |
| G1 zero-balance, n=100 | 0.394 | 0.096 |
| **G1 zero-balance, all available** | **0.504** | **0.092** |
| G2 zero-balance + one ref point | 0.147 | 0.119 |
| G3 2-point linear | 0.567 | 0.086 |
| G3 3-point linear | 0.584 | 0.085 |
| **G3 5-point linear** | **0.593** | **0.085** |
| G4 one cycle, common S | 0.424 | 0.103 |
| G4 one cycle, branch-split S | -0.061 | 0.131 |

Full per-dataset/direction breakdown: `tables/calibration_results_all.csv`, `tables/calibration_summary.csv`.
`figures/calibration_curve_R2.png`, `figures/calibration_curve_RMSE.png`,
`figures/calibrated_predictions_all_temperatures.png`.

**G0 is unreliable and highly variable** (dataset mean R2 ranges from +0.58 (ds1) to -6.9 (ds3) —
matches Phase F's known-T inverse instability, as expected since it's the same computation).

**G1 (zero-balance alone) is the single biggest jump**: mean R2 goes from -1.45 to +0.33-0.50 depending
on budget, with **using all available near-zero-strain samples clearly best** (0.504) — more calibration
samples for B helps monotonically, no diminishing returns observed within the tested range. This
confirms the reviewer's "zero-balance is close to free and should be tried first" intuition directly.

**G2 (adding one reference-strain point for S) is *worse* than G1 alone in aggregate** (0.147 vs
0.504) — a genuine, non-obvious negative result. The internal-validation-only selection correctly
prefers a mid-strain reference over low/high extremes in most folds (`tables/reference_strain_selection.csv`
— low-strain references are occasionally catastrophic, e.g. ds4 A_to_B ref_low R2=-24.5), but even the
best single reference point is *noisier* than trusting the Phase E LOO-extrapolated S(T) (which already
has a low 6.4% mean LOO error, per Phase E) — replacing a good extrapolated estimate with one noisy
in-session point measurement is a net loss on average. **One-point sensitivity calibration should not
be recommended as-is.**

**G3 (multi-point static calibration) is the best deployable level overall** (0.567 -> 0.584 -> 0.593
for 2 -> 3 -> 5 points) — clear, and diminishing returns are already visible by 3 points (+0.017 from
2->3, +0.009 from 3->5). **3 reference points plus zero-balance is a reasonable practical recommendation** —
5 points buys very little extra.

**G4 (one full cycle) reaches 0.424 with a common (loading=unloading) S, but *branch-split* S from a
single cycle is actively harmful (-0.061)** — one cycle is too little data to reliably split S by
branch; this reinforces Phase E's own finding that loading/unloading separation is not consistently
justified with this data.

### Electrode selection and fusion — does calibration solve the Phase F problem?

`tables/electrode_fusion_calibration.csv`. **Yes, decisively — and this is the most important new
result of Phase G.** At G0, both electrodes are unreliable (mean R2: E1=-3.76, E2=-8.10). But from G1
onward, **Electrode 2 is consistently the more reliable electrode in every one of the 4 datasets**
(not just on average) — e.g. at G3 5-point: ds1 E2=0.812 vs E1=-0.197; ds2 E2=0.0001 vs E1=-0.992;
ds3 E2=0.876 vs E1=0.715; ds4 E2=0.677 vs E1=0.270. This is a sharp contrast with Phase F, where the
"best electrode" flip-flopped unpredictably across held-out temperatures (E2 in some folds, E1 in
others) when B/S had to be extrapolated across temperature with only 3 training points. **With even
minimal in-session calibration, electrode reliability becomes a stable, consistent property rather
than a fold-dependent artifact of cross-temperature extrapolation noise.**

The `constrained_linear_fusion` weight (selected only from the calibration-internal validation split,
never the real test) tracks `E2_only` almost exactly at every level and every dataset — it correctly
learns to put nearly all its weight on E2 without ever looking at the real test set. **Naive
equal-weight fusion is consistently worse than either the correctly-selected single electrode or the
learned fusion** (e.g. G3 5-point mean: equal=0.339 vs E2_only/fusion=0.591) — averaging two electrodes
of very different reliability is a mistake; a calibration-driven weight (or simply defaulting to E2) is
not.

### Session-shift analysis

`tables/session_shift_penalty.csv`, `figures/same_vs_cross_session_calibration.png` — same-session
(calibrate on part of an Experiment, test on a held-back part of the SAME Experiment) vs cross-session
(calibrate on Experiment 1, test on Experiment 2, and reverse), using the G4 one-cycle calibration:

| Dataset | A->B penalty | B->A penalty | Mean \|penalty\| |
|---|---|---|---|
| ds1 (33C) | +0.066 | -0.067 | 0.066 |
| ds2 (43C) | +0.104 | -0.085 | 0.094 |
| ds3 (52C) | +0.007 | -0.015 | 0.011 |
| ds4 (62C) | +0.040 | -0.048 | 0.044 |

The penalty **flips sign between directions for every dataset** — it is not "session 2 is always worse
than session 1," it is genuine, idiosyncratic Experiment-to-Experiment variation, consistent with Phase
F's session-calibrated-vs-true-oracle gap. **ds2 (43C) has the largest session-shift magnitude** of the
4 datasets (~0.09), which also explains why ds2's G3 cross-session result (R2~0.0001) lags far behind
its true-oracle same-experiment number from Phase F (R2=0.531) — ds2 appears to be a session-shift-heavy
dataset specifically, not just a hard interpolation case.

**Known, unresolved limitation — cause of ds2's session anomaly.** `Electrode2_resistance`'s
between-Experiment baseline std for ds2 is 5035 (`mechanism_between_experiment_variability.csv`), 3.5-25x
larger than the same quantity for ds1/ds3/ds4 (919/1399/201). Both of ds2's Experiments also show the
largest, same-signed within-session drift of any dataset (-0.43%/-0.38% for E1, -0.31%/-0.26% for E2,
vs mixed-sign, smaller drifts elsewhere). A per-step-jump check (`max abs single-step jump` ~75-85 ohm
vs total session drift ~3900-4100 ohm) shows the drift is smooth and continuous, not a discrete
recording error/contact glitch — so a simple data-entry mistake is unlikely. Beyond that, **the cause
cannot be determined from the data alone** (no experiment log, operator record, or mounting/timing
metadata is available to distinguish a genuine 43C-specific physical effect from a session-specific
setup difference, e.g. re-mounting or incomplete thermal equilibration before this particular pair of
runs). **This should be stated as an open, documented limitation in the thesis, not resolved into a
specific causal claim** — the diagnostic evidence rules out crude data-entry error but cannot adjudicate
between "real 43C material effect" and "this specific session's setup," and no further work in this
repository can resolve it without external experiment records.

### Output Summary (Phase G)

**A. G0 performance:** unreliable, mean R2=-1.45, ranges from +0.58 (ds1) to -6.9 (ds3).

**B. G1 zero-balance improvement:** large — mean R2 -1.45 -> 0.50 (using all available near-zero-strain
samples); monotonic improvement with more calibration samples, no observed diminishing returns in the
tested range (10/50/100/all).

**C. G2 one-point improvement:** **negative** — mean R2 drops to 0.15, worse than G1 alone. A single
reference-strain point is noisier than the Phase E LOO-extrapolated S(T); not recommended as-is.

**D. G3 multi-point diminishing returns:** clear and fast — 2pt: 0.567, 3pt: 0.584, 5pt: 0.593. Most of
the benefit is captured by 2-3 points; 5 points is only marginally better.

**E. G4 one-cycle improvement:** moderate with common S (0.424, between G1 and G3), but branch-split S
from one cycle is harmful (-0.061) — too little data to split reliably by loading/unloading.

**F. Best calibration level per temperature:** ds1 (33C): G1 first-segment/n10 (R2~0.86, cheapest);
ds2 (43C): none reach strong performance (G1 n50 best at ~-0.60 to G3 near 0, all mediocre — ds2 is the
hardest case, likely due to its large session-shift, see above); ds3 (52C): G3 3pt/5pt (R2~0.87-0.89);
ds4 (62C): G3 5pt (R2~0.68).

**G. Best single deployable calibration level overall:** **G3 with 3 reference points (plus
zero-balance)** — best mean performance-per-calibration-effort trade-off; 5 points adds little.

**H. Same-session vs cross-session penalty:** real but modest and direction-inconsistent (flips sign
per direction); largest for ds2 (~0.09-0.10 RMSE), smallest for ds3 (~0.01).

**I. Does calibration solve electrode selection/fusion?** **Yes.** Unlike Phase F's cross-temperature
nested selection (which picked an unreliable, fold-flipping electrode), in-session calibration reveals
Electrode 2 as consistently more reliable than Electrode 1 in all 4 datasets, and a
calibration-internal-validation-selected fusion weight recovers this automatically without ever seeing
the real test data.

**J. Is B recalibration alone sufficient?** Mostly — G1 (zero-balance only, keeping S from cross-temperature
extrapolation) already recovers most of the achievable gain; this is the single highest-leverage,
lowest-cost calibration step.

**K. Is S recalibration necessary?** Only with enough reference points — one point (G2) actively hurts;
2-3 points (G3) gives a real, if modest, additional gain over G1 alone. A single-point sensitivity
recalibration should not be used.

**L. Recommended practical deployment procedure:** (1) always perform zero-balance calibration (cheap,
large gain); (2) if 2-3 reference strain points can be collected, add multi-point G3 calibration for a
further modest gain; (3) prefer/weight Electrode 2 via a calibration-internal-validated fusion weight,
not equal-weight averaging or a temperature-cross-validated choice; (4) do not attempt single-point
sensitivity calibration or branch-split sensitivity from a single cycle.

**M. Recommended thesis claim:** *"A physically-grounded forward model (R = B(T) + S(T)*epsilon),
combined with minimal same-session calibration (zero-balance plus a small number of reference strain
points) and a calibration-validated electrode fusion weight, recovers most of the achievable accuracy
at a new temperature/session — outperforming both a purely cross-temperature-extrapolated inverse and
a deep ANN baseline, while remaining fully interpretable."* This is a defensible, demonstrated,
practically-deployable claim, stronger than either the original "best deep learning architecture" framing
or the earlier "analytical inverse E1-only" framing (now retracted).

**N. Is Phase H (residual learning) still necessary?** **Likely yes, but as a secondary refinement, not
the primary story.** Even the best G3 calibration (mean R2~0.59) does not close the gap to Phase F's
true-oracle ceiling (mean R2~0.53, computed differently but comparable order) uniformly across datasets
— ds2 in particular remains weak (R2~0) due to session shift that static B/S recalibration does not fully
correct. A residual model trained on top of the G3-calibrated inverse, using interpretable
branch/history features (not true strain derivative), remains a reasonable next step specifically to
address the *remaining* hysteresis/session-shift residual — but the calibration study itself, not
residual learning, is now the strongest and most defensible core contribution.

---

## Output Summary

**A. Corrected 43C Linear+T result:** R2 = 0.566 (D-fix, whole-dataset test) / 0.756 (Phase F,
second-experiment-only test, T-known setting) — consistently positive across both protocols.

**B. Corrected 43C Deep ANN result:** R2 = -10.69 (D-fix, 2 inputs, whole dataset) / -6.17 (Phase F,
3 inputs incl. T, second-experiment test) — consistently and badly negative regardless of protocol or
whether T is given as an input.

**C. Best B(T) model and its LOO error:** full exponential (a*e^(bT)+c), mean absolute relative LOO
error 0.49%. Theoretically non-identifiable with only 3 training points (zero residual d.f.) but
empirically the most accurate candidate by a wide margin — the functional form itself fits this
sensor well.

**D. Best S(T) model and its LOO error:** full exponential, mean absolute relative LOO error 6.42%
(next best: reduced exponential at 7.71%).

**E. Loading/unloading separate sensitivities:** not consistently needed — only 2 of 8
dataset x electrode combinations show non-overlapping bootstrap CIs between S_loading and S_unloading;
a common S(T) is a reasonable default.

**F. Known-temperature inverse R2/RMSE per held-out temperature (train-only-selected variant, no
test-set peeking):** ds1 33C: R2=-22.06 (E2_only), RMSE=0.680 | ds2 43C: R2=0.275 (common_S), RMSE=0.126
| ds3 52C: R2=-7.28 (both_weighted), RMSE=0.423 | ds4 62C: R2=-1.55 (E2_only), RMSE=0.220. **The
earlier "E1-only is best" claim was model-selection leakage** — a train-pool-only nested selection
picks a *different*, often worse-performing variant per fold, showing that electrode reliability does
not reliably transfer from the 3 known training temperatures to a genuinely new one (see
`analytical_inverse_variant_selection_nested.csv`). This is an honest negative result, not a fixable
implementation gap.

**G. Unknown-temperature inverse R2/RMSE and temperature error (post-fix):** strain RMSE now 0.23-0.24
(ds1, ds4) to 0.25-0.30 (ds2, ds3), all R2 negative (-1.46 to -3.23); `search_failed_rate` (fraction of
rows with no physically valid grid candidate) is 0% for ds1-3 and 13.5% for ds4. The earlier RMSE=29.9
outlier was a genuine clipping bug (fixed) — the corrected numbers are still uniformly poor, so
**F2 was already unreliable, not merely miscomputed**; the bug inflated one number but didn't manufacture
the underlying conclusion.

**H. Oracle inverse result (now two distinct numbers, per the review):**
- **True oracle** (same Experiment, cycle-split calibration, zero session/cross-temperature error):
  R2 = 0.726 (ds1), 0.531 (ds2), 0.630 (ds3), 0.232 (ds4); RMSE 0.07-0.12 — consistently positive,
  the cleanest evidence yet that the static forward model itself is sound at a *known* temperature.
- **Session-calibrated** (Experiment-to-Experiment shift included): R2 = 0.264, -0.826, 0.705, -0.292 —
  markedly worse and far less consistent than the true oracle in 3 of 4 folds, confirming
  Experiment-to-Experiment session shift is a real, separate error source that the earlier merged
  "oracle" number was obscuring.

**I. Error decomposition (three-way):** true-oracle RMSE (0.07-0.12) -> +session shift (mostly small,
+0.01 to +0.10, one fold slightly negative from small-sample calibration noise) -> +B(T)/S(T)
LOO-prediction error (the dominant term for ds1 and ds3: +0.56 and +0.34). **The forward model is sound;
session shift is a real but secondary factor; predicting B/S at a genuinely unseen temperature remains
the dominant, hardest problem.**

**J. Jacobian conditioning conclusion:** unchanged — median condition number ~12 (mild), but max ~7243
with sign-changing det(J) across the (T, strain) grid — the system is **only locally, not globally,
well-conditioned for joint T-strain inversion**. This explains F2's residual instability and is a
genuine, non-implementation-artifact limitation.

**K. Is the analytical inverse suitable as the thesis backbone?** **Yes for the forward model itself,
not yet for a specific deployable electrode configuration.** The true oracle result (H) is the strongest
finding in Phase F: the static `R=B(T)+S(T)*epsilon` form is sound and consistent (R2 0.23-0.73) with no
session or cross-temperature error. But: (a) a fair, train-only electrode-selection rule does *not*
reliably recover the earlier E1-only advantage — that specific configuration claim is retracted; (b)
joint T-strain estimation (F2) is unreliable given the Jacobian conditioning; (c) session shift (H, I)
is a real, separate error source on top of B/S-prediction error. **Revised recommendation: keep the
`R=B(T)+S(T)*epsilon` forward model as the backbone (justified by the true-oracle result), but treat
"how to combine/select electrodes" and "how to handle session shift" as open sub-problems to solve via
Phase G (calibration) rather than assume solved** — this is a more modest, more defensible claim than
the v1 "E1-only inverse rivals Linear+T" framing.

**L. Recommended next phase:** **Phase G (calibration-level study) is now the clear top priority**,
for two reasons the corrected Phase F newly supports: (1) it directly targets the dominant error source
identified in (I) — B(T)/S(T) LOO-prediction error at the unseen temperature — by measuring how much of
it a small amount of target-temperature data can recover; (2) it can also directly test whether a small
calibration sample resolves the electrode-selection non-transfer problem found in (F), e.g. by letting
electrode weighting be fit per-deployment from calibration data rather than chosen once from training
data. Residual learning (Phase H) should follow Phase G, using the *true-oracle* configuration (not a
session-calibrated or LOO one) as the interpretable baseline to correct, since (I) shows session shift
and B/S-prediction error are separate, addressable problems rather than irreducible model-form error.
Joint T-strain estimation (F2) remains lower priority given the identifiability findings, unless a
specific new idea addresses the conditioning problem directly.

---

## Phase H — Interpretable Residual Correction (v2, final decision run)

Script: `scripts/phase_h_residual_learning.py`. This supersedes an earlier v1 pass in place after two
review rounds found real issues to fix:

1. **Column-semantics check (done before this run, per the reviewer's request).** `Experiment` is a
   measurement **session** id — exactly 2 distinct values per dataset (e.g. ds1 = {3,4}), each with its
   own `Idx` restarting at 0. Sessions are not the same thing as loading cycles: each session contains
   ~10 loading/unloading cycles internally, found by turning-point detection on the strain signal
   (Phase B), not given as a raw column. `Experiment`/`Idx` are used only to sort rows and choose
   calibration vs. test sessions — confirmed by code audit that neither is ever passed as a model input.
2. **Leakage bug fix.** v1's `cycle_idx` feature was computed via `detect_turning_points(df["Strain"])`
   on the TEST dataframe too — i.e. it used the true strain signal (through its turning points) as an
   inference-time feature. Removed entirely (also required by the explicit rule against using cycle ID
   as a feature).
3. **Small GRU added**, and results are now reported **per calibration direction** (not averaged before
   the decision), per the reviewer's stricter final-decision protocol.

Baseline = Phase G "G3, 3-point" calibrated inverse with calibration-internal-validated electrode
fusion (same protocol as Phase G throughout — the real test session is never used to pick anything).
Residual target: `r = true_strain - eps_physical`, fit on the calibration session's own residual,
applied to the test session (cross-session). Features (all deployable: resistance/temperature/
physical-prediction only, never true strain, its derivative, or any session/cycle ID): R1, R2,
Temperature; smoothed dR1/dt, dR2/dt; rolling mean/std of R1, R2; running (causal) max/min of R1, R2;
branch (loading/unloading) inferred from resistance derivative sign; time since last turning point
(from resistance); the previous physical (G3) prediction.

**Per-dataset x direction RMSE and % improvement vs. G3** (`tables/residual_per_direction_rmse.csv`):

| Dataset | Direction | G3 RMSE | Ridge (% impr.) | MLP (% impr.) | GRU (% impr.) |
|---|---|---|---|---|---|
| ds1 (33C) | A→B | 0.069 | -19.4% | +25.9% | -31.8% |
| ds1 (33C) | B→A | 0.057 | -60.3% | +41.9% | +26.7% |
| ds2 (43C) | A→B | 0.161 | +58.2% | +55.2% | +50.1% |
| ds2 (43C) | B→A | 0.136 | +75.8% | +28.2% | +67.4% |
| ds3 (52C) | A→B | 0.045 | -131.0% | -22.0% | +40.3% |
| ds3 (52C) | B→A | 0.061 | -75.0% | -22.1% | +3.0% |
| ds4 (62C) | A→B | 0.093 | +61.6% | +68.5% | +46.3% |
| ds4 (62C) | B→A | 0.068 | +20.7% | +8.7% | +1.6% |

**Pre-registered decision rule** (keep a model only if: mean RMSE improvement ≥10%; improves in
**both** directions for **every** dataset; no dataset/direction degrades by more than 20%; ds2/43C
improves) — `tables/residual_decision_rule.csv`:

| Model | Mean % improvement | Improves both directions, every dataset | Worst-case change | Major degradation (>20%) | 43C improvement | **KEEP** |
|---|---|---|---|---|---|---|
| Ridge | -8.7% | No | -131.0% | Yes | +67.0% | **No** |
| Small MLP | +23.0% | No | -22.1% | Yes | +41.7% | **No** |
| Small GRU | +25.4% | No | -31.8% | Yes | +58.7% | **No** |

**=> Final decision: DROP all residual models. Use the Phase G "G3" calibrated physical inverse alone
as the final method.**

**Reading this honestly:** all 3 models show a real, substantial *mean* improvement (Ridge is the
exception, unstable overall) — MLP +23%, GRU +25% — and ds2 (43C, the session-shift-heavy dataset)
improves dramatically and consistently under every model (+28% to +76%), confirming Phase G's
prediction that a residual model can recover exactly the session-shift structure static calibration
misses. **But no single model is reliable everywhere**: ds3 is a consistent weak point for Ridge and
MLP (both directions degrade, -22% to -131%) while GRU fixes ds3 but then fails ds1's A→B direction
(-31.8%). Each candidate has at least one fold where it makes things meaningfully worse than
calibration alone — exactly the failure mode the pre-registered rule was designed to catch, and
exactly why the rule (not a post-hoc judgment call) is what decides this: **per the rule, none pass,
so none are kept.** This is an honest negative result, not a failure of the residual-learning idea in
principle — with only 4 datasets x 2 directions (8 evaluation folds) there isn't enough data to fit a
per-dataset or per-direction model-selection rule reliably (the same small-sample problem seen
throughout Phases E-G), so a single fixed residual model applied everywhere is the wrong level of
ambition for what this dataset can support.

`figures/final_prediction_all_temperatures.png` and `figures/residual_vs_loading_state.png` still show
the MLP comparison for illustration, but per the decision above, **the MLP/Ridge/GRU results are
reported as evidence for the decision, not as part of the final deployed method.**

---

## Phase H2 — Calibrated-Feature ANN (final combination test — PASSES)

Script: `scripts/phase_h2_calibrated_ann.py`. Key difference from Phase H: instead of feeding raw
`(R1, R2, T)` to an ANN (what the original repo's Deep ANN did) or adding an ANN-predicted residual on
top of the physics prediction (Phase H), this **physically normalizes the input first**, using the
same Phase G G3 calibration (`B_i`, `S_i` from zero-balance + 3 reference points):

```
z_i(t) = (R_i(t) - B_i) / S_i        # per-electrode physical strain estimate
[z1, z2, dz1, dz2, T] -> small ANN (Dense(8)->Dense(4)->Dense(1), L2=1e-2, early stopping) -> eps_ANN
eps_final = alpha*eps_ANN + (1-alpha)*eps_physics(G3)
```

`alpha` is selected per calibration session **only from that session's own internal train/val split**
(same protocol as Phase G/H — never the real test session), by grid search minimizing internal
validation RMSE.

**Per-dataset x direction results** (`tables/calibrated_ann_per_direction_rmse.csv`):

| Dataset | Direction | G3 RMSE | ANN-only (% impr.) | **Blended (% impr., alpha)** |
|---|---|---|---|---|
| ds1 (33C) | A→B | 0.069 | -104.5% | **+7.7% (alpha=0.10)** |
| ds1 (33C) | B→A | 0.057 | -156.4% | **+18.9% (alpha=0.15)** |
| ds2 (43C) | A→B | 0.161 | +23.7% | **+20.6% (alpha=0.80)** |
| ds2 (43C) | B→A | 0.136 | +32.1% | **+16.5% (alpha=0.25)** |
| ds3 (52C) | A→B | 0.045 | -0.9% | **+6.5% (alpha=0.65)** |
| ds3 (52C) | B→A | 0.061 | +28.6% | **+28.0% (alpha=0.50)** |
| ds4 (62C) | A→B | 0.093 | +15.8% | **+13.9% (alpha=0.65)** |
| ds4 (62C) | B→A | 0.068 | +11.4% | **+19.1% (alpha=0.85)** |

**Decision rule** (keep only if: improves every dataset in both directions; no fold degrades more than
10%; 43C improves; mean RMSE reduction ≥10%) — `tables/calibrated_ann_decision_rule.csv`:

| Model | Mean % improvement | Worst-case change | Every dataset improves, both directions | No fold degrades >10% | 43C improvement | **KEEP** |
|---|---|---|---|---|---|---|
| Calibrated ANN alone (alpha=1) | -18.8% | -156.4% | No | No | +27.9% | **No** |
| **Blended (calibration-selected alpha)** | **+16.4%** | **+6.5%** | **Yes** | **Yes** | **+18.6%** | **YES** |

**=> This passes.** Unlike Phase H's fixed additive residual model, the calibrated ANN **alone** is
just as unstable as before (catastrophic at ds1: -104% to -156%, same overfitting-to-calibration-noise
failure mode as Phase H's MLP). But **the alpha-blend rescues it**: `alpha` is chosen automatically,
per deployment, from calibration-internal data alone, and it does exactly what you'd want —
**ds1 (where the ANN fails) gets alpha=0.10-0.15 (mostly physics); ds3/ds4 (where the ANN helps) get
alpha=0.50-0.85 (mostly ANN)**. The blend is never worse than G3 alone in any of the 8 folds tested
(worst case is still +6.5%, not a degradation), and 43C improves by 18.6% — a real, reliable, and
*better-behaved* result than Phase H's fixed residual model.

**Important scope note:** the original repo's Deep ANN reached R2=0.85, but that was measured
**same-temperature** (train/test split within one temperature — an easy, in-domain task, see Phase B/D
discussion above). This experiment is evaluated cross-session under Phase G's harder, actual-deployment
protocol, so R2 in the 0.25-0.92 range here (mean R2 across folds ≈ 0.70) is **not directly comparable**
to that 0.85 and should not be read as "matching" it — the comparison that matters is against G3
(physics + calibration alone), which this blend reliably beats.

**Revised final method:** physics-based forward model + Phase G calibration (zero-balance + 3-point +
electrode fusion) **as the front-end**, with a **small ANN operating on the calibration-normalized
features** (`z1, z2, dz1, dz2, T`) as a **calibration-validated nonlinear correction**, blended via an
alpha weight chosen from calibration data alone. This reverses Phase H's "drop all residual models"
conclusion — the earlier rejection was specific to *adding an unconstrained residual model on raw
resistance-derived features*; normalizing the input first and gating the ANN's contribution through a
calibration-validated blend weight fixes exactly the reliability problem that got Phase H's models
rejected.

---

## Current Status / Next Decision

*(Checkpoint, not a final decision — this section summarizes where things stand and what's supportable
right now; it is not a declaration that the thesis direction is closed.)*

**Output Summary (superseded by Phase H2 — see below for the current decision):**

**A. Data-column semantics:** `Temperature`, `Electrode1_resistance`, `Electrode2_resistance` are
inputs; `Strain` is the target; `Experiment` identifies a measurement **session** (2 per dataset), each
containing ~10 loading/unloading cycles found by signal processing, not given as a column; `Idx` is
within-session time order. Neither `Experiment` nor `Idx` is ever used as a model feature (audited).

**B. G3 baseline results:** RMSE 0.045-0.161 across datasets/directions — matches Phase G.

**C-G. Phase H (fixed additive residual on raw resistance-derived features):** Ridge mean -8.7%
(unstable, -131% to +76%); MLP mean +23.0% (fails ds3, -22% both directions); GRU mean +25.4% (fixes
ds3, fails ds1 A→B, -31.8%). No model satisfied "improves every dataset in both directions with no
fold degrading >20%" — **Phase H's fixed residual models were rejected** (see Phase H section above for
full detail; this conclusion is superseded below, not because Phase H's numbers were wrong, but because
Phase H2 found a materially different, more reliable way to combine physics with an ANN).

**H2. Calibrated-feature ANN + calibration-validated blend (final combination test) — PASSES.**
Normalizing the input first (`z_i = (R_i-B_i)/S_i` via Phase G's own calibration) before the ANN, then
blending the ANN's prediction with the physics prediction using a weight `alpha` selected from
calibration-internal data alone, **passes the (even stricter) pre-registered decision rule**: mean RMSE
improvement +16.4%, worst-case fold change **+6.5% (never a degradation)**, every dataset improves in
both directions, and 43C improves by 18.6%. The blend weight adapts sensibly per fold (ds1, where the
raw ANN fails, gets alpha=0.10-0.15 i.e. mostly physics; ds3/ds4, where the ANN helps, get
alpha=0.50-0.85). See "Phase H2" above for the full table.

**I. Final recommended thesis method and title (revised after Phase H2):**

> **温度依存物理正規化とANN補正を統合した繊維型抵抗センサのひずみ推定**
> (Strain estimation for fiber-based resistive sensors combining temperature-dependent physical
> normalization with a calibration-validated ANN correction)

The method is: physics-based forward model + Phase G calibration (zero-balance + 3-point + electrode
fusion) **as a front-end normalization step**, feeding a small ANN whose contribution is blended with
the physics prediction via a weight chosen entirely from calibration data. This reverses Phase H's
"drop all deep learning" conclusion — **the earlier rejection was specific to adding an unconstrained
residual model on raw resistance-derived features; physically normalizing the input first, and gating
the ANN's influence through a calibration-validated blend, fixes exactly the reliability problem that
got Phase H's models rejected.** If, after your review, this result is not trusted enough to lead the
thesis (e.g. because of the still-small 8-fold evaluation base), the fallback title from Phase H
(`温度依存物理モデルと少数点校正に基づく繊維型抵抗センサのひずみ推定`, calibration-only, no ANN) remains
fully supported and is a safe, defensible alternative — Phase H2's ANN blend is the current best-tested
method, not the only acceptable one.

**What this project supports, end to end:** a physically-grounded forward model
(`R_i(t) = B_i(T) + S_i(T)*epsilon(t)`) with minimal same-session calibration (Phase G) is sound and
outperforms both a purely cross-temperature-extrapolated inverse (Phase F) and a deep ANN baseline
(consistently and badly negative in every LOO fold tried, Phases D-fix and F). On top of that, a small
ANN trained on calibration-normalized features, blended with the physics prediction via a
calibration-validated weight (Phase H2), reliably improves on calibration alone in every tested fold —
including recovering much of the session-shift weakness at 43C that static calibration alone could not
fix. The mechanism is fully explained throughout: temperature dominates raw resistance far more than
strain does (Phase B); B(T)/S(T) are smooth and well-modeled by an exponential (Phase E); the static
forward model is sound at a known temperature (Phase F true-oracle); the remaining gap is
B(T)/S(T)-extrapolation error (addressed by calibration) and session-to-session shift (now
substantially, though not perfectly, addressed by the calibrated-ANN blend).

**What this project does NOT support (do not claim these in the thesis):**
- Generalization to other fiber-based sensors, materials, or sensor geometries — only this dataset's
  2 electrodes across 4 temperatures were studied.
- Generalization to arbitrary unseen temperatures — only interpolation (ds2, ds3) and a narrow
  extrapolation range (ds1, ds4) were tested; nothing establishes behavior further outside 33-62C.
- Industrial deployment-readiness — the pipeline was validated offline on existing Experiment splits,
  not in a live deployment loop, and session-shift causes (data vs. physical vs. operator/setup) remain
  an open, undetermined limitation (see Phase G).
- That session shift is fully "solved" — Phase H2 substantially improves on it (never worse than
  calibration alone, +18.6% at 43C) but ds2 remains the weakest fold overall (R2 0.25-0.41); it is
  improved, not eliminated.
- That this beats the original repo's same-temperature Deep ANN (R2=0.85) — that number measures a
  different, easier, in-domain task (see Phase H2's scope note); the correct comparison is against G3
  physics+calibration alone, which this blend reliably beats.
- Uniformly high accuracy in all conditions — ds2 (43C) remains the weakest fold even after the
  Phase H2 improvement, and this should be reported plainly, not smoothed over.

**Defensible thesis claim:** *"Within this dataset, a physically-grounded model of temperature-strain
coupling, combined with minimal same-session calibration, substantially improves strain prediction at
temperatures/sessions not used to build the model. Feeding the calibration-normalized signal (rather
than raw resistance) into a small ANN, and blending its prediction with the physics prediction using a
weight selected entirely from calibration data, reliably improves further in every tested fold —
including partially recovering the session-shift weakness a purely static calibration could not fix —
without the instability seen when the same ANN is applied to raw, unnormalized features or as an
unconstrained additive residual."*

**Status:** this is the current checkpoint, not a locked/final decision — the direction can still be
revised pending your review. Historical/superseded artifacts have been removed from the working tree
(recoverable via git history, commits up to `889da86`, if ever needed) per the decision to keep this
project on a single, current direction.

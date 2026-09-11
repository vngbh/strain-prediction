# Research Report

## Title

Physics-normalized causal tree learning for cross-temperature and cross-session
strain prediction in an embedded resistive fiber sensor

Japanese working title:

> 物理正規化と因果的動的特徴を用いた木ベース機械学習による繊維型抵抗センサのひずみ推定

## Research Question

Can one globally trained machine-learning model be transferred to a different
temperature and measurement session after only a short physical calibration?

Raw resistance cannot answer this reliably because it responds to strain,
temperature, hysteresis, drift, and session-specific offsets at the same time.

## Final Method

For electrode `i`, a first-order sensor model is calibrated:

```text
R_i(t) = B_i + S_i * epsilon(t) + residual_i(t)
z_i(t) = (R_i(t) - B_i) / S_i
```

`B_i` is the zero-strain baseline and `S_i` is strain sensitivity. They are
estimated from four reference-level windows inside a fixed 125 s calibration
prefix. The levels are zero plus 5%, 50%, and 95% of the observed calibration
strain range. The zero window is `strain <= minimum + 0.02%`; the other windows
use `+/-0.03%`. The two normalized electrode estimates are fused using a weight
selected from these same calibration aggregates.

A `HistGradientBoostingRegressor` then predicts strain **directly** from:

- `z1`, `z2`, and the fused normalized estimate;
- derivatives and a loading/unloading proxy requiring no future samples;
- trailing rolling mean, standard deviation, minimum, and maximum.

Temperature is omitted from the final model. Under frozen LOTO, adding `T`
reduces mean improvement and causes one direction to degrade by 50.5%,
consistent with the limited extrapolation ability of decision trees.

The HGB model is trained globally on the non-target temperatures and then
frozen. At the target temperature, strain labels are used only for physical
calibration, never to train or tune HGB.

## Evaluation Protocol

Four temperatures are evaluated: 33, 43, 52, and 62 degC. Each has two
sessions. For every held-out temperature and direction (`A -> B`, `B -> A`):

1. Exclude both target-temperature sessions from HGB training.
2. Train HGB on both sessions at the other three temperatures.
3. Freeze HGB.
4. Use the fixed first 125 s of target session A to estimate `B`, `S`, and the
   electrode fusion weight at four reference levels.
5. Predict every sample of target session B; then reverse A and B.

This gives eight directional evaluations from four temperature groups, each
containing two sessions. Neither the eight directions nor the four groups are
independent sensor replicates: they share one specimen, experimental history,
and overlapping LOTO training sets. These are retrospective internal results,
not external confirmation. No additional measurement data are available, so the
thesis must keep that boundary explicit.

## Main Result

The final protocol averages samples around four reference levels within the
fixed 125 s calibration event (426-477 rows contribute to the four aggregates,
depending on direction). This requires a known proof load and a reference strain
measurement during calibration; it is not a calibration-free startup.

| Protocol | Mean RMSE | Mean R2 | Mean improvement | Worst fold | Folds improved |
|---|---:|---:|---:|---:|---:|
| Physics only, matched 125 s calibration | 0.0881% | - | baseline | - | - |
| **Frozen HGB: physics-normalized, causal, no `T`** | **0.0484%** | **0.859** | **37.6%** | **0.5%** | **8/8** |
| Frozen raw-resistance HGB with `T` | 0.1144% | 0.220 | -78.6% | -409.1% | 2/8 |
| Frozen raw HGB + target zero-offset + `T` | 0.0797% | 0.638 | -10.5% | -84.6% | 3/8 |

`0.0484%` strain is approximately `484 microstrain`, or 9.7% of the 0.5%
full-scale strain. The weakest-prediction fold is 43 degC A-to-B, with
`R2 = 0.430` and `RMSE = 0.112%` (about 1120 microstrain, 22.3% full scale).
The smallest improvement is only 0.5% at 33 degC B-to-A; therefore “8/8” is a
consistency description, not evidence of a large margin in every direction.

## Critical Ablations

All checks below use the **same final protocol**: frozen LOTO, fixed 125 s
calibration, and opposite-session testing.

| Check | Mean RMSE improvement | Worst fold | Interpretation |
|---|---:|---:|---|
| **Causal, no-`T`, direct HGB (final)** | **37.6%** | **+0.5%** | Only variant improving all directions |
| Residual-target HGB | 24.2% | -49.4% | Fails one direction |
| Centered derivative HGB | 35.2% | -6.2% | Uses future samples and fails one direction |
| Causal HGB with `T` | 33.2% | -50.5% | `T` harms one unseen-temperature fold |
| Ridge on the same features | 11.0% | -108.7% | Fails two temperature groups |
| Causal smoothing only | -6.7% | -13.7% | Improves no direction |
| Raw HGB with `T` | -78.6% | -409.1% | Fails all four temperature groups |
| Raw HGB + target zero-offset + `T` | -10.5% | -84.6% | Offset-only calibration remains insufficient |

The zero-offset comparator receives the same 125 s target calibration event but
uses only its no-load window. It is the requested stronger raw baseline.
Providing all four reference levels as an offset-and-scale transform would, in
substance, be the proposed physical normalization. The result supports
**calibrated transfer**, not label-free transfer.

## History-Length Sensitivity

The final causal derivative uses a trailing 21-sample mean. It requires no
future samples but has smoothing/group lag of roughly 10 samples (about 1 s).
The model also uses rolling history up to 201 samples (20.1 s).

| Available rolling history | Mean improvement | Worst direction | Groups improving both directions |
|---|---:|---:|---:|
| Local features only | 28.7% | -10.8% | 3/4 |
| Up to 21 samples | 27.1% | -5.1% | 2/4 |
| Up to 51 samples | 29.9% | -2.4% | 2/4 |
| Up to 101 samples | 32.0% | -1.8% | 2/4 |
| **Up to 201 samples** | **37.6%** | **+0.5%** | **4/4** |

This is a retrospective sensitivity analysis, not an independently tuned window
selection. It shows that long causal history, rather than smoothing alone, is
associated with reliable transfer in this corpus.

## Leakage And Error Audits

For all eight directions, three executable checks return exactly zero numerical
difference:

- features/predictions at time `t` are identical when computed from the full
  session or only the prefix ending at `t`;
- permuting every test-session strain label does not change predictions;
- permuting every calibration-session strain label after 125 s does not change
  calibration or predictions.

The original peak-detected “first cycle” selector failed the third audit because
its centered peak detector could inspect samples after the nominal boundary. It
was therefore replaced by the fixed 125 s prefix. The 125 s choice itself was
made retrospectively from observed cycle durations and is disclosed as such.

Prediction improvement is lowest near turning points (17.7%) and during stable
unloading (10.9%), versus 41.8% during stable loading. It remains positive in
the early, middle, and late thirds of sessions. Edge held-out temperatures
(33/62 degC) achieve mean `R2 = 0.910`; interior temperatures (43/52 degC)
achieve `R2 = 0.808`, driven mainly by the difficult 43 degC condition.

## What Is Actually Supported

The defensible contribution is not a new boosting algorithm and not the broad
idea of ML temperature compensation. It is the following application-method
result:

> Within this existing one-sensor dataset, a globally trained causal HGB model
> can transfer across the tested temperatures and sessions after a short,
> physically interpretable four-level calibration. Physical normalization is
> essential for reliable cross-temperature transfer among the tested variants,
> while dynamic nonlinear learning improves over the matched physical baseline
> in every evaluated direction.

The method is machine learning, not deep learning. Physics supplies a stable
coordinate system; HGB learns the remaining nonlinear and history-dependent
mapping. The model predicts strain directly, so it must not be described as
learning only a residual.

## Limitations

- One embedded sensor/specimen, four temperature groups, and two sessions per
  temperature; these are not independent sensor replicates.
- The method was selected after exploring this same corpus; the evidence is
  retrospective and cannot be called untouched external validation. The fixed
  HGB hyperparameters were inherited from development-stage analysis. They were
  not retuned on an outer target fold in this audit, but prior exposure to the
  corpus remains a meta-level model-selection limitation.
- Opposite directions from one temperature pair are not independent samples.
- Four-level calibration requires a known proof load and continuously observed
  reference strain during the fixed 125 s event. The source experiment used a
  bonded HBM LY11-6/350 strain gauge; its full uncertainty budget is unavailable.
- Dynamic-feature ablation shows that history matters, but does not by itself
  prove that all gain comes specifically from hysteresis.
- Accuracy has not been compared with an application-specific alarm threshold,
  so industrial SHM readiness is not claimed.
- With no possibility of additional measurements, generalization to another
  sensor, geometry, material, loading rate, or long-term aging remains outside
  the evidence available to this thesis.

## Reproduce

```bash
python3 src/frozen_protocol_audit.py
```

This command reproduces the final frozen-protocol ablations, window sensitivity,
causality/label-access audit, regional error analysis, per-temperature results,
and `results/figures/frozen_protocol_audit.png`.
Best- and lowest-R2 time traces are in
`results/figures/frozen_best_worst_predictions.png`.

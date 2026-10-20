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

`B_i` is the fitted low-strain intercept and `S_i` is strain sensitivity. They are
estimated from four reference-level windows inside a fixed 125 s calibration
prefix. The levels are zero plus 5%, 50%, and 95% of the observed calibration
strain range. The low-strain window is `strain <= minimum + 0.02%`; the other windows
use `+/-0.03%`. The two normalized electrode estimates are fused using a weight
selected from these same calibration aggregates. In every evaluated
calibration, this constrained selector assigned zero weight to electrode 1, so
the fused physics estimate is identical to electrode 2. Electrode 1 remains
available to HGB separately through `z1` and its temporal features.

A `HistGradientBoostingRegressor` then predicts strain **directly** from:

- `z1`, `z2`, and the fused normalized estimate;
- derivatives and a loading/unloading proxy requiring no future samples;
- trailing rolling mean, standard deviation, minimum, and maximum.

Temperature is omitted from the final model. Under frozen LOTO, adding `T`
reduces mean improvement and causes one direction to degrade by 50.5%,
consistent with the limited extrapolation ability of decision trees.

The HGB model is trained globally on the non-target temperatures and then
frozen. Within each fixed final-evaluation outer fold, target-temperature
strain labels are used only for physical calibration; they do not fit, update,
or tune HGB. This statement describes the data boundary of the final run, not
an independent model-selection history: the model family, calibration duration,
features, and window lengths were selected retrospectively after examining this
corpus.

## Evaluation Protocol

Four temperatures are evaluated: 33, 43, 52, and 62 degC. Each has two
sessions. For every held-out temperature and direction (`A -> B`, `B -> A`):

1. Exclude both target-temperature sessions from HGB training.
2. Train HGB on both sessions at the other three temperatures.
3. Freeze HGB.
4. Use the fixed first 125 s of target session A to estimate `B`, `S`, and the
   electrode fusion weight at four reference levels.
5. Predict every sample of target session B; then reverse A and B.

Source preprocessing mirrors this deployment transformation. Each of the six
source sessions estimates its own `B`, `S`, and fusion weight from that same
session's first 125 s. Those parameters normalize that whole session, including
its calibration prefix, whose rows are also HGB training rows; neither its
full-session labels nor the opposite session are used to estimate
its calibration parameters. Causal features are then constructed separately
within each session before the six source feature matrices and targets are
concatenated to train one global HGB.

At the beginning of a session, trailing statistics use all history currently
available (`min_periods=1`) until the requested window is full. The first causal
derivative value is set to zero. No initial rows are dropped and no future or
cross-session padding is introduced.

This gives eight directional evaluations from four temperature groups, each
containing two sessions. Neither the eight directions nor the four groups are
independent sensor replicates: they share one specimen, experimental history,
and overlapping LOTO training sets. These are retrospective internal results,
not external confirmation. No additional measurement data are available, so the
thesis must keep that boundary explicit.

## Configuration Selection Provenance

The final evaluation holds the outer target temperature out of HGB fitting, but
it does not erase prior researcher exposure to the full corpus. The distinction
is documented explicitly:

| Choice | Status in the final run | How it was chosen |
|---|---|---|
| HGB family and fixed hyperparameters | Frozen inside every outer fold | Inherited from development on this corpus |
| 125 s calibration prefix | Fixed for all directions | Chosen retrospectively from observed cycle durations |
| Four calibration bands | Fixed for all directions | Retained after development-stage calibration comparisons |
| Direct target, causal features, no `T` | Fixed for all directions | Compared retrospectively on this corpus |
| Rolling windows through 201 samples | Fixed for all directions | Supported by retrospective history-length sensitivity |

Consequently, the final table is a re-evaluation of one fixed configuration
under a strict deployment boundary, not an untouched estimate of the entire
model-selection procedure. A source-only nested audit remains possible as a
sensitivity analysis, but performing it now cannot undo the fact that the
researcher has already observed the corpus.

## Main Result

The final protocol averages samples around four reference levels within the
fixed 125 s calibration event (426-477 rows contribute to the four aggregates,
depending on direction). This requires a known proof load and a reference strain
measurement during calibration; it is not a calibration-free startup.

| Protocol | Mean RMSE | Mean R2 | Mean improvement | Worst fold | Folds improved |
|---|---:|---:|---:|---:|---:|
| Physics only, matched 125 s calibration | 0.0881% | - | baseline | - | - |
| **Frozen HGB: physics-normalized, causal, no `T`** | **0.0484%** | **0.859** | **37.6%** | **0.5%** | **8/8** |
| Frozen raw-resistance HGB, no `T` | 0.1156% | 0.203 | -80.8% | -414.7% | 2/8 |
| Frozen raw HGB + target low-strain offset, no `T` | 0.0821% | 0.631 | -16.1% | -82.0% | 3/8 |

`0.0484%` strain is approximately `484 microstrain`, or 9.7% of the 0.5%
full-scale strain. The weakest-prediction fold is 43 degC A-to-B, with
`R2 = 0.430` and `RMSE = 0.112%` (about 1120 microstrain, 22.3% full scale).
The smallest improvement is only 0.5% at 33 degC B-to-A; therefore “8/8” is a
consistency description, not evidence of a large margin in every direction. It
also depends on calibration bookkeeping and on the position of the calibration
window; see Calibration Sensitivity below.

The reported 37.6% is the arithmetic mean of the eight per-direction relative
improvements. The ratio of aggregate mean RMSEs is a different summary:
`1 - 0.0484 / 0.0881 = 45.1%`. Both are valid, but they answer different
questions; this report uses the former as the primary consistency-aware metric.
The distinction is especially important for raw HGB with a low-strain offset
and no `T`: its ratio-of-means reduction is `+6.9%`, but its mean
per-direction improvement is `-16.1%` and only 3/8 directions improve. A few
high-error folds dominate the aggregate RMSE, so this comparator is not a
reliable transfer method despite the favorable ratio of means.

## Critical Ablations

All checks below use the **same final protocol**: frozen LOTO, fixed 125 s
calibration, and opposite-session testing.

| Check | Mean RMSE improvement | Worst fold | Interpretation |
|---|---:|---:|---|
| **Causal, no-`T`, direct HGB (final)** | **37.6%** | **+0.5%** | Only variant improving all directions |
| Static normalized HGB (no dynamic features) | 5.1% | -91.0% | Normalization alone is insufficient for reliable transfer |
| Residual-target HGB | 24.2% | -49.4% | Fails one direction |
| Centered derivative HGB | 35.2% | -6.2% | Uses future samples and fails one direction |
| Causal HGB with `T` | 33.2% | -50.5% | `T` harms one unseen-temperature fold |
| Ridge on the same features | 11.0% | -108.7% | Fails two temperature groups |
| Causal smoothing only | -6.7% | -13.7% | Improves no direction |
| Raw HGB with `T` | -78.6% | -409.1% | Fails all four temperature groups |
| Raw HGB without `T` | -80.8% | -414.7% | Removing `T` does not rescue raw transfer |
| Raw HGB + low-strain offset with `T` | -10.5% | -84.6% | Offset-only calibration remains insufficient |
| Raw HGB + low-strain offset without `T` | -16.1% | -82.0% | Same conclusion after isolating `T` |

The strict static comparator uses the same physical normalization and HGB but
only `[z1, z2, zf]`; it removes derivatives, branch proxy, rolling statistics,
and temperature. It improves 6/8 directions, but its worst direction degrades
by 91.0%. Adding causal dynamics raises the mean per-direction improvement from
5.1% to 37.6% and the worst direction from -91.0% to +0.5%. This is direct
evidence that physical normalization and temporal description play distinct,
complementary roles under the final frozen protocol.

The low-strain-offset comparator receives the same 125 s target calibration
event but uses only its lowest-strain window. It is the requested stronger raw
baseline. Because force was not recorded, this window is not claimed to prove
a mechanically unloaded state.
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

## Calibration Sensitivity

These analyses keep the frozen protocol and change only how or when the target
calibration is taken. Scripts and outputs are in `research/probes/`; the
direction note is
`research/directions/2026-10-20-calibration-transfer-limits.md`.

**Bookkeeping.** The four calibration aggregates are assigned their nominal
strain levels, with exactly zero for the low band, although the low band's
measured strain is slightly above zero. Assigning each aggregate its measured
mean strain instead gives a mean per-direction improvement of 35.8%, a worst
direction of -5.5%, and 7/8 improved directions. The average gain survives;
the all-direction count does not.

**Window position.** Sliding the 125 s calibration window every 10 s over the
calibration session gives 813 dependent evaluations. Under nominal-level
calibration, the frozen dynamic HGB outperformed matched physics across all
sampled calibration positions in seven directions and 98/109 positions in the
eighth. This demonstrates a broadly persistent relative advantage within this
corpus, without establishing improved absolute accuracy at later positions.
Later windows were associated with higher HGB RMSE in most directions but lower
RMSE in one; window position, strain coverage, and low-band strain are strongly
dependent within sessions, so no general calibration-selection rule was
established.

**Static versus dynamic across windows.** Over the same window positions, the
dynamic HGB had lower RMSE than the static normalized HGB at every position in
seven directions and 96/109 positions in the eighth. Averaged over directions,
the median RMSE across positions was 0.0879% for physics, 0.0717% for static
HGB, and 0.0496% for dynamic HGB.

**What the temporal features represent.** Adding elapsed session time to the
static inputs did not recover the dynamic model's performance, and an explicit
causal loading-program state (branch, time since turning points, cycle count)
recovered only part of the gap. Loading-program tracking and sensor-state
mechanisms therefore remain unresolved.

## Leakage And Error Audits

For all eight directions, five executable checks pass, with exactly zero
numerical difference wherever a numerical comparison applies:

- features/predictions at time `t` are identical when computed from the full
  session or only the prefix ending at `t`;
- inference runs after the test-session `Strain` column is removed entirely;
- deleting the calibration-session tail after 125 s does not change calibration
  or predictions;
- strongly altering both resistance channels and temperature after time `t`
  does not change any feature or prediction through `t`;
- A-to-B and B-to-A at one held-out temperature reuse the same fitted HGB object,
  whose serialized fingerprint is unchanged by target calibration or inference.

These audits found no leakage in the tested data-access paths and boundaries.
They should not be interpreted as an absolute proof that no untested leakage
path can exist.

The original peak-detected “first cycle” selector failed the third audit because
its centered peak detector could inspect samples after the nominal boundary. It
was therefore replaced by the fixed 125 s prefix. The 125 s choice itself was
made retrospectively from observed cycle durations and is disclosed as such.

Prediction improvement is lowest near turning points (17.7%) and during stable
unloading (10.9%), versus 41.8% during stable loading. It remains positive in
the early, middle, and late thirds of sessions. Edge held-out temperatures
(33/62 degC) achieve mean `R2 = 0.910`; interior temperatures (43/52 degC)
achieve `R2 = 0.808`, driven mainly by the difficult 43 degC condition.

## Position Against Closest Work

The broad ideas of machine-learning sensor compensation, history-aware
hysteresis correction, and physics-guided learning are not new. The closest
individual studies define a narrower and more defensible position for this
work:

| Study | Closest overlap | Difference from this study |
|---|---|---|
| [Caglar et al. (2021)](https://doi.org/10.1002/admt.202000957) | Same embedded two-electrode fiber sensor; neural temperature-strain decoupling | Does not evaluate a frozen causal model with short target calibration and opposite-session transfer |
| [Koike et al. (2019)](https://doi.org/10.3390/s19194259) | Time-series information for sensor hysteresis compensation | Force/torque sensors rather than transfer across temperature and session in this fiber sensor |
| [Hannigan et al. (2021)](https://doi.org/10.3389/fmats.2021.639823) | Carbon-black fiber strain sensors, ML models, and signal history | Flexible wearable sensing with a different validation setting, not frozen target-temperature transfer |
| [Liu et al. (2024)](https://doi.org/10.3390/s24165394) | GRU-based temperature compensation for a piezoresistive sensor | Pressure sensor and recurrent compensation rather than physically normalized causal tree transfer |
| [Tian et al. (2026)](https://doi.org/10.1016/j.measurement.2025.119846) | Historical signals and cross-domain compensation for flexible piezoresistive sensing | Targets dynamic creep and loading-frequency transfer using FastDTW and domain adaptation |
| [Li et al. (2026)](https://doi.org/10.1038/s41467-026-72113-4) | Physics-guided ML for nonlinearity, hysteresis, cycling attenuation, and batch variation | Soft strain sensors, dedicated new experiments, sequential deep models, and robotic demonstrations |

The novelty claim is therefore not "physics plus ML" or "using history to
compensate a sensor." It is the evaluated combination of this thermally drawn
embedded fiber sensor, a fixed 125 s target-condition offset-and-sensitivity
calibration, causal history features, a globally trained HGB that remains frozen
at the target temperature, and matched deployment-boundary ablations. This is
an application-method contribution within the studied corpus, not a claim of a
new general learning algorithm.

## What Is Actually Supported

The defensible contribution is not a new boosting algorithm and not the broad
idea of ML temperature compensation. It is the following application-method
result:

> Within this existing one-sensor dataset, a globally trained causal HGB model
> can transfer across the tested temperatures and sessions after a short,
> physically interpretable four-level calibration. Among the tested variants,
> offset-and-sensitivity normalization gives the best mean cross-temperature
> transfer and the fewest failed directions, while dynamic nonlinear learning
> improves over the matched physical baseline in every evaluated direction.

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
- The strict static-feature ablation shows that causal history materially
  improves transfer, but does not prove that all of this gain comes specifically
  from hysteresis rather than drift or other time-dependent effects.
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

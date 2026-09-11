# Physics-Normalized Causal Tree Learning

Strain prediction for a resistive fiber sensor when temperature and measurement
session change, using a physically normalized and globally frozen tree model.

```text
other-temperature sessions -> train one causal HGB model -> freeze the model

new temperature/session -> fixed 125 s, four-level physical calibration
  -> z_i = (R_i - B_i) / S_i -> causal dynamic features
  -> frozen HGB -> strain estimate
```

The method is machine learning/AI, not deep learning. No additional data are
assumed or required by the current study.

## Main Result

Leave one complete temperature out of HGB training, calibrate physically on the
fixed first 125 seconds of one target session, and test on the opposite target
session:

| Method | Mean RMSE | Mean R2 | Mean improvement | Worst fold |
|---|---:|---:|---:|---:|
| Frozen causal HGB + 125 s calibration | **0.0484% strain** | **0.859** | **37.6%** | **+0.5%** |
| Frozen raw-resistance HGB with `T` | 0.1144% strain | 0.220 | -78.6% | -409.1% |
| Raw HGB + target zero-offset + `T` | 0.0797% strain | 0.638 | -10.5% | -84.6% |

The normalized method improves the matched physical baseline in all eight
directional evaluations from four temperature groups. All groups share one
sensor/specimen and are not independent sensor replicates. This is retrospective
internal leave-one-temperature-out evidence, not external validation.

## Why the Method Is Defensible

- Every major ablation uses the same frozen LOTO and fixed 125 s calibration.
- Smoothing alone makes the result worse (-6.7% mean).
- Ridge fails its worst direction (-108.7%).
- Residual HGB, centered features, and HGB with `T` each fail one direction.
- The final causal, no-`T`, direct HGB is the only tested variant that improves
  every direction.
- Raw HGB works within a temperature but fails when the held-out temperature
  changes. Among the tested variants, physical normalization is therefore the
  essential step for reliable transfer.
- HGB is frozen before target calibration and receives zero target-condition
  labels for model training. Target labels are used only during the known 125 s
  proof-load calibration event.
- Prefix-causality, test-label access, and post-calibration label-access audits
  all pass with exactly zero numerical prediction difference.

## Scope

The contribution is the integration and validation of physical calibration,
causal dynamic descriptors, and small-data tree learning for this thermally
drawn embedded fiber sensor. It is not a new boosting algorithm and does not
claim that ML temperature compensation in general is new.

Only the existing one-sensor dataset is available. Generalization to new
sensors, materials, loading rates, or long-term operation cannot be established
and is stated as a limitation rather than deferred to hypothetical future data.

## Files

```text
src/frozen_protocol_audit.py             # final method, ablations, and audits
results/tables/                          # reproducible numeric results
results/figures/                         # result figures
REPORT.md                                # academic interpretation and limitations
ENGLISH_EXPLANATION.md                   # simple-language explanation
```

Raw data and local progress-report documents are ignored by Git.

## Run

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/frozen_protocol_audit.py
```

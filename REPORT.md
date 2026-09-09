# Research Report

## Title

Physics-normalized tree boosting for strain prediction in resistive fiber sensors

Japanese working title:

> 物理正規化と木ベース機械学習補正による繊維型抵抗センサのひずみ推定

## Research Question

Can an embedded resistive fiber sensor estimate mechanical strain reliably when temperature and
measurement session change?

The practical problem is that resistance is affected by both strain and temperature/session drift.
A model trained directly on raw resistance can therefore confuse environmental or setup changes with
mechanical strain.

## Method

The method has two parts.

First, a small same-session calibration converts each electrode resistance into a strain-like signal:

```
z_i = (R_i - B_i) / S_i
```

`B_i` is obtained from near-zero-strain samples. `S_i` is obtained from three reference strain points.
The two electrodes are fused using a calibration-internal validation split.

Second, a tree-based machine-learning readout predicts strain from physically normalized and causal
time-series features:

- normalized estimates: `z1`, `z2`, fused `z`
- smoothed derivatives
- causal rolling statistics
- loading/unloading branch inferred from the resistance signal
- temperature

The main model is `HistGradientBoostingRegressor`. `ExtraTreesRegressor` is used as an independent
stability check.

## Evaluation

Evaluation uses four temperatures with two measurement sessions each:

| Dataset | Temperature | Evaluation |
|---|---:|---|
| dataset1 | 33 degC | session A -> B and B -> A |
| dataset2 | 43 degC | session A -> B and B -> A |
| dataset3 | 52 degC | session A -> B and B -> A |
| dataset4 | 62 degC | session A -> B and B -> A |

All model choices are made from the calibration session only. The test session is never used for
calibration, model selection, scaling, or blending.

The baseline is the calibrated physical estimate before the tree-based correction.

## Result

| Model | Mean RMSE improvement | Worst fold | 43 degC improvement | Stability |
|---|---:|---:|---:|---|
| HistGradientBoosting | 54.3% | 43.0% | 56.9% | KEEP in 4/4 seeds |
| ExtraTrees | 53.4% | 41.2% | 52.4% | KEEP in 4/4 seeds |

The result is not a deep-learning result. It is a small-data machine-learning result built on top of
physical normalization.

## Contribution

The contribution is the full method, not a new tree algorithm:

1. Show that raw black-box neural models are fragile under temperature/session shift.
2. Use physical calibration to remove the dominant resistance offset and sensitivity variation.
3. Learn only the remaining nonlinear correction from normalized deployable features.
4. Demonstrate that tree-based ML is more stable than the previous ANN correction for this small
   experimental dataset.

## Application Value

This matters for structural health monitoring of fiber-reinforced composites. If the sensor is used
outside one controlled laboratory session, temperature and setup drift can be misread as mechanical
strain. The proposed method reduces that failure mode while keeping the pipeline lightweight enough
for practical calibration and deployment.

## Limitations

- The result is validated on one sensor dataset with four main test temperatures.
- The independent experimental unit count is small: four temperature/session pairs, evaluated in two
  directions each.
- The method assumes a short same-session calibration is available.
- Generalization to other fiber materials, electrode compositions, geometries, load rates, and
  long-term aging requires new experiments.

## Reproduce

```bash
python3 src/physics_normalized_tree_boosting.py
```

Outputs:

- `results/tables/tree_boosting_results.csv`
- `results/tables/tree_boosting_per_direction_rmse.csv`
- `results/tables/tree_boosting_decision_rule.csv`
- `results/figures/tree_boosting_vs_physics.png`

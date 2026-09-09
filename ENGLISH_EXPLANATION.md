# Simple English Explanation

## What This Project Is About

This project predicts **strain**, which means how much a composite material is being stretched.

The sensor has two resistive electrodes embedded inside a fiber-reinforced composite material. When
the material is stretched, the electrical resistance changes. In a simple situation, we could say:

```text
resistance changes -> strain changes
```

But in the real experiment, resistance changes for more than one reason.

It changes because of:

- mechanical strain,
- temperature,
- measurement session differences,
- sensor drift,
- hysteresis between loading and unloading.

So the same resistance value can mean different strain values under different temperatures or
sessions. This is the main problem.

## Why Raw Resistance Is Not Enough

If we train an AI model directly on raw resistance, the model may learn the wrong relationship.

For example, the model may think:

```text
higher resistance = higher strain
```

But higher resistance may also come from higher temperature or session drift. This makes raw
resistance unstable for strain prediction when the temperature or session changes.

## The Main Idea

The project first uses a simple physical calibration:

```text
z = (R - B) / S
```

where:

- `R` is the measured resistance,
- `B` is the baseline resistance when strain is almost zero,
- `S` is the sensitivity of the sensor,
- `z` is a normalized strain-like signal.

This step does not make the sensor perfect. But it removes the biggest resistance offset and
sensitivity changes caused by temperature and session differences.

After this physical normalization, machine learning is used to learn the remaining errors.

The full flow is:

```text
raw resistance
-> physical calibration
-> normalized strain-like signal
-> causal dynamic features
-> tree-based machine learning correction
-> final strain prediction
```

## Why Use Machine Learning After Physics

The physical model handles the main part of the problem: baseline and sensitivity.

However, some effects are still difficult to write as a simple formula from this small dataset:

- hysteresis,
- low-strain nonlinearity,
- drift,
- session-dependent behavior.

Machine learning is used only for this remaining part. It does not replace physics. It improves the
prediction after physics has already organized the data into a better form.

In simple words:

```text
Physics makes the signal easier to understand.
Machine learning corrects what physics does not fully explain.
```

## Why Dynamic Features Are Important

The model should not look at each data point as an isolated point.

For this sensor, the recent signal history matters. The same resistance-like value can mean different
things during loading and unloading. This is called hysteresis.

So the project adds causal dynamic features, such as:

- whether the signal is increasing or decreasing,
- the loading/unloading branch,
- recent rolling average,
- recent rolling standard deviation,
- recent rolling minimum and maximum.

These features are not new measurements. They are calculated from the existing signal. They give the
machine learning model useful physical context about recent behavior.

In simple words:

```text
Static features tell the model where the signal is now.
Dynamic features tell the model how the signal got there.
```

## Why Tree-Based Machine Learning

This project uses tree-based machine learning, mainly `HistGradientBoostingRegressor`, with
`ExtraTreesRegressor` as a stability check.

This is machine learning / AI, but it is **not deep learning**.

Tree-based models are suitable here because:

- the experimental dataset is small,
- there are only a few independent measurement sessions,
- deep learning models can overfit easily,
- tree models are easier to train and explain,
- the physical normalization already does much of the hard work.

So the model is simple, but not weak. It is chosen because it fits the data situation better than a
large neural network.

## Current Prediction Performance

Compared with the calibrated physical baseline, the tree-based model improves prediction accuracy
across all tested temperature/session directions.

Main result:

```text
Mean RMSE improvement: 54.3%
Worst fold improvement: 43.0%
43 degC improvement: 56.9%
Seed stability: KEEP in 4/4 seeds
```

The dynamic feature ablation shows why the final model is stronger:

| Feature set | Mean RMSE improvement | Worst fold | Result |
|---|---:|---:|---|
| Static normalized features only | 14.6% | -13.4% | Fails robustness check |
| Local dynamics: derivatives + branch | 47.9% | 31.4% | Passes |
| Full dynamics: local dynamics + rolling statistics | 54.3% | 43.0% | Best |

This means physical normalization alone is not enough. A static tree model improves the average
result, but it can still fail on one session. Adding dynamic features makes the model much more
reliable.

The test R2 values range from about:

```text
0.697 to 0.980
```

The RMSE ranges from about:

```text
0.0207% strain to 0.0814% strain
```

Because full scale is 0.5% strain, this corresponds to approximately:

```text
207 microstrain to 814 microstrain
```

The hardest case is still the 43 degC A-to-B session test. This should be reported honestly. But the
model still improves over the baseline even in that difficult case.

## Why This Is Useful

This type of sensor can be embedded inside fiber-reinforced composite materials. These materials are
used in many engineering applications, such as:

- aircraft structures,
- automotive parts,
- wind turbine blades,
- sports equipment,
- civil engineering structures,
- smart composite materials.

In real use, the structure may experience different temperatures and different measurement
conditions. If the sensor cannot separate temperature effects from real mechanical strain, it may
give wrong information.

For example:

- it may report strain when the structure is only heating up,
- it may miss real strain because temperature drift hides it,
- it may require recalibration too often,
- it may give unreliable structural health monitoring signals.

This project helps make the sensor more reliable when temperature and session conditions change.

## Main Research Message

The main message is:

> Raw resistance is not stable enough for strain prediction under different temperatures and
> measurement sessions. By first using physical calibration and then applying a small tree-based
> machine learning correction with causal dynamic features, strain can be predicted more accurately
> and more reliably.

This is not a new deep learning algorithm. The contribution is the combination of physical
normalization, dynamic feature engineering, and small-data machine learning for this specific
embedded fiber sensor problem.

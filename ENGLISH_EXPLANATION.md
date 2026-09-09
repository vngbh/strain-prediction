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
> machine learning correction, strain can be predicted more accurately and more reliably.

This is not a new deep learning algorithm. The contribution is the combination of physical
normalization and small-data machine learning for this specific embedded fiber sensor problem.


# Simple English Explanation

## The Problem

This project predicts strain, which means how much a composite structure is
stretched. The sensor contains two electrical-resistance channels.

Stretching changes resistance, but temperature, measurement setup, sensor
history, and loading direction also change it. Therefore, the same resistance
can represent different strain values in different tests. An AI model that
memorizes raw resistance from one temperature may fail at another temperature.

## The Solution

The method gives physics and machine learning different jobs.

First, a short calibration changes resistance into a strain-like coordinate:

```text
z = (R - B) / S
```

`B` is a fitted low-strain intercept and `S` is sensor sensitivity. They are
measured from four reference-level windows during a fixed 125-second proof-load
calibration. Reference strain must be observed during this event.

Second, one globally trained `HistGradientBoostingRegressor` uses the normalized
signals and their recent history. Its features describe whether the signal is
rising or falling and its recent average, variation, minimum, and maximum. All
features use only the present and past, so the final method needs no future
samples. Its trailing smoother still introduces about one second of group lag.

In simple words:

```text
Physics removes the large temperature/session shift.
Machine learning interprets the smaller nonlinear and history effects.
```

## A Fair Deployment Test

For each test temperature, both sessions at that temperature are removed from
AI training. The AI learns from the other three temperatures and is then
frozen. At the new temperature, only the four-level physical calibration is
performed. The frozen AI never trains on target-temperature strain data.

The experiment is repeated in both session directions at 33, 43, 52, and
62 degC. This makes eight directional tests from four temperature groups. They
all use one sensor/specimen and are not independent sensor replicates.

## Accuracy

With fixed 125-second calibration, the final model achieves:

```text
Mean RMSE:                 0.0484% strain (about 484 microstrain)
Mean R2:                   0.859
Improvement over physics:  37.6% on average
Smallest improvement:      0.5%
Directional tests helped:  8 out of 8
```

The weakest test is 43 degC A-to-B, with `R2 = 0.430` and `RMSE = 0.112%`
strain. The method is
therefore promising, but it is not equally accurate in every condition.

The "8 out of 8" depends on a calibration bookkeeping detail. If the
calibration points use the strain actually measured in each band instead of
the nominal levels, 7 of 8 tests improve, although the average gain stays
similar.

## What Proves Each Part Matters

- Physical normalization without any dynamic features helps on average by only
  5.1% and makes its worst test 91.0% worse. Adding causal signal history raises
  the mean improvement to 37.6% and makes even the weakest direction 0.5%
  better. Physics and dynamics therefore do different jobs.
- Smoothing alone makes the average result 6.7% worse.
- Linear Ridge regression fails badly in its worst test (-108.7%).
- A nonlinear tree model with causal features improves all tests (with the
  standard calibration bookkeeping).
- Adding temperature makes one unseen-temperature direction 50.5% worse, so the
  final frozen tree deliberately omits it.
- A frozen model trained on raw resistance, even with temperature as an input,
  fails across temperatures: mean `R2 = 0.220`, with only 2 of 8 tests improved.
- Removing `T` does not rescue raw HGB: mean `R2 = 0.203`, and only 2 of 8
  directions improve.
- Giving raw HGB the same calibration event for a low-strain offset, without
  `T`, is still insufficient: mean `R2 = 0.631`, and only 3 of 8 directions
  improve.
- The physically normalized frozen model reaches mean `R2 = 0.859` and improves
  all 8 tests (7 of 8 if the calibration uses measured band strains).

This is the strongest evidence that physical normalization is useful for
transfer, while causal nonlinear dynamic learning contributes more than either
static normalized learning or simple smoothing.

## What The October 2026 Checks Added

- The advantage over physics does not depend on one lucky calibration moment:
  it persists when the 125-second calibration window is moved through the
  session.
- The signal-history features really help, but they are not simply "learning
  hysteresis". Part of their benefit matches knowing where the sensor is in
  the loading cycle; part matches a timing mismatch between the sensor and the
  reference gauge; the rest is unexplained.
- The sensor and the surface reference gauge appear offset in time by a few
  seconds at 33-62 degC, but this did not reappear in the one 21 degC session
  that was kept aside for a single confirmation test.
- The first loading cycle of every session looks more sensitive than later
  cycles, yet skipping it before calibration did not help.
- No untouched data are left, so these remain findings about this one sensor
  and loading program.

## Application

The sensor is embedded inside fiber-reinforced composite material. Such
materials are used in aircraft, vehicles, wind-turbine blades, sports equipment,
and civil structures.

Reliable strain estimation can support structural health monitoring. It can
reduce false alarms caused by heating, missed strain events hidden by drift,
and the need to retrain an AI model whenever the operating temperature changes.

The current data are not enough to claim industrial readiness or performance on
another sensor. The study shows what can be supported by this one existing
sensor dataset; no additional measurements are assumed.

The executable audits found no leakage in the data-access paths and boundaries
they tested. This is strong internal evidence, not an absolute proof covering
every conceivable implementation error.

## Main Claim

> In this dataset, a globally trained causal tree model can transfer across the
> tested temperatures and sessions after a short physical calibration. Physics
> normalization gives the best mean transfer and the fewest failed directions
> among the tested variants, and nonlinear dynamic machine learning improves the
> matched physical baseline in every evaluated direction under the standard
> calibration bookkeeping.

This is machine learning, not deep learning. The contribution is the complete
physics-plus-ML method and its careful validation, not a new tree algorithm.

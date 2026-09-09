# Physics-Normalized Tree Boosting for Fiber Sensor Strain Prediction

This project predicts mechanical strain from two resistive electrodes embedded in a fiber-reinforced
composite sensor. The current research direction is intentionally narrow:

```
raw resistance + temperature
  -> same-session physical calibration
  -> physical normalization z_i = (R_i - B_i) / S_i
  -> tree-based machine-learning correction
  -> strain estimate
```

The learning model is **machine learning / AI**, but **not deep learning**. The main model is
`HistGradientBoostingRegressor`; `ExtraTreesRegressor` is used as a stability check.

The aim is not to win a model zoo comparison. The aim is to make an embedded resistive fiber sensor
usable when temperature and measurement session change, so that temperature drift or setup drift is
not misread as mechanical strain.

## Current Result

Against the calibrated-physics baseline, evaluated on ds1-ds4 with both cross-session directions:

| Model | Mean RMSE improvement | Worst fold | 43 degC improvement | Stability |
|---|---:|---:|---:|---|
| HistGradientBoosting | 54.3% | 43.0% | 56.9% | KEEP in 4/4 seeds |
| ExtraTrees | 53.4% | 41.2% | 52.4% | KEEP in 4/4 seeds |

The ANN/CNN/LSTM/CWT exploration is no longer the active direction. The prior ANN-fusion report is
kept in `to_know_how/` as background, but this cleaned project focuses on the tree-boosted method.

## Files

```
src/physics_normalized_tree_boosting.py   # main method and evaluation
results/tables/                           # result CSVs
results/figures/                          # result figure
REPORT.md                                 # concise research report
to_know_how/                              # source paper, lab files, raw CSVs, original progress report
```

## Data

The raw columns are:

| Column | Meaning |
|---|---|
| `Temperature` | Ambient temperature, degC |
| `Strain` | Target strain, in percent; full scale is 0.5% = 5000 microstrain |
| `Electrode1_resistance` | Resistance of electrode 1 |
| `Electrode2_resistance` | Resistance of electrode 2 |
| `Experiment` | Measurement session id |
| `Idx` | Time index |

Primary evaluation uses `to_know_how/files/dataset1.csv` to `dataset4.csv`:

| Dataset | Temperature | Sessions |
|---|---:|---:|
| dataset1 | 33 degC | 2 |
| dataset2 | 43 degC | 2 |
| dataset3 | 52 degC | 2 |
| dataset4 | 62 degC | 2 |

`dataset0.csv` is the 21 degC reference-temperature file from the source material, but it is not part
of the current cross-session evaluation.

## Run

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/physics_normalized_tree_boosting.py
```

Outputs are written to `results/tables/` and `results/figures/`.

## Thesis Claim

Within this dataset, a same-session physical calibration step turns raw resistance into a normalized
strain-like signal. A tree-based ML readout trained on those normalized and causal time-series
features improves cross-session strain prediction more reliably than the prior ANN correction, while
remaining easier to train, explain, and deploy on small experimental data.

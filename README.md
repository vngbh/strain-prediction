# Strain Prediction from Fiber-Based Sensors

Predict mechanical **strain** from electrical resistance signals of fiber-based sensors, with a focus
on generalization across unseen temperatures. Prior deep-learning experiments (ANN, LSTM, CNN, CWT,
Optuna) motivated the current direction but have been removed from the working tree to keep the
project on a single direction — that code is still recoverable from git history (commits up to
`889da86`) if it's ever needed again.

---

## Current direction

A physically-grounded forward model,

```
R_i(t) = B_i(T) + S_i(T) * epsilon(t) + residual
```

combined with minimal same-session calibration and a calibration-validated electrode fusion weight,
rather than a black-box deep-learning architecture. Full methodology, findings, and the authoritative
results ledger are in **[`research_diagnosis/REPORT.md`](research_diagnosis/REPORT.md)**.

Summary of what's been established so far (see REPORT.md for full detail and caveats):
- The repo's original deep-learning pipeline had a few confirmed bugs (test-set leakage into Optuna,
  minor window-boundary crossing) but none of them explain the catastrophic failure at unseen
  temperatures — that failure reproduces even in a from-scratch paper-style model and in plain linear
  regression baselines.
- `B(T)` and `S(T)` (baseline resistance and strain-sensitivity, per electrode) are smooth, monotonic,
  and well described by an exponential form — the physical model itself is sound.
- A pure cross-temperature-extrapolated inverse is unreliable (electrode reliability doesn't transfer
  across temperature), but **minimal in-session calibration (zero-balance + a few reference strain
  points) recovers most of the achievable accuracy** and makes electrode selection stable — this is the
  current core result.
- Session-to-session (Experiment-to-Experiment) variation is a real, separate error source on top of
  temperature; residual learning on top of the calibrated inverse is the planned next step (Phase H).

## Dataset

5 CSV files collected at different temperatures (not included in this repository):

| Column | Description |
|--------|-------------|
| `Electrode1_resistance` | Resistance of electrode 1 (Ω) |
| `Electrode2_resistance` | Resistance of electrode 2 (Ω) |
| `Temperature` | Ambient temperature (°C) |
| `Strain` | Target — mechanical strain |
| `Experiment` | Experiment ID |
| `Idx` | Time index |

| Dataset | Samples | Temperature |
|---------|---------|-------------|
| dataset0 | 11,776 | ~21°C (excluded — single experiment, sanity-check only) |
| dataset1 | 23,053 | ~33°C |
| dataset2 | 23,227 | ~43°C |
| dataset3 | 21,497 | ~52°C |
| dataset4 | 23,077 | ~62°C |

---

## Project structure

```
research_diagnosis/
├── REPORT.md              # Full findings, authoritative results ledger, decision log
├── scripts/                # One script per diagnostic phase (B, C, D-fix, E, F, G, ...)
├── tables/                 # CSV outputs — see REPORT.md's ledger for which are authoritative
├── figures/                # PNG outputs referenced from REPORT.md
└── logs/                   # Package versions for reproducibility

data/            # Raw sensor CSVs (not committed)
other/           # Original course/lab materials (assignment brief, mentor slides, etc.)
```

---

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib tensorflow keras
# exact versions used are recorded in research_diagnosis/logs/package_versions.txt
```

## Run

Each script under `research_diagnosis/scripts/` is self-contained and documents its own inputs/outputs
in its docstring, e.g.:

```bash
cd /path/to/strain-prediction
.venv/bin/python3 research_diagnosis/scripts/phase_g_calibration.py
```

See `research_diagnosis/REPORT.md` for what each phase script produces and how the results fit together.

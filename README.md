# Strain Prediction from Fiber-Based Sensors

Predict mechanical **strain** from electrical resistance signals of fiber-based sensors using Deep Learning. This project improves upon a baseline ANN by systematically exploring feature engineering, advanced architectures, and cross-temperature generalization.

---

## Background

Fiber-based strain sensors measure mechanical deformation by detecting changes in electrode resistance. The challenge: sensor behavior shifts across different temperatures (thermal drift), making it difficult to build a model that generalizes beyond its training conditions.

**Task:** Regression — given electrode resistance readings (and temperature), predict strain value.

---

## Dataset

5 CSV files collected at different temperatures. Each contains:

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
| dataset0 | 11,776 | ~21°C |
| dataset1 | 23,053 | ~33°C |
| dataset2 | 23,227 | ~43°C |
| dataset3 | 21,497 | ~52°C |
| dataset4 | 23,077 | ~62°C |

> Dataset is not included in this repository.

---

## Project Structure

```
code/
├── utils/
│   ├── data_loader.py       # Load datasets, feature definitions
│   └── metrics.py           # MSE, MAE, RMSE, R² evaluation
│
├── 01_eda.py                # Phase 1: Exploratory Data Analysis
├── 02_baseline.py           # Phase 2: Reproduce original ANN
├── 03_feature_engineering.py  # Phase 3: Temperature + derived features
│
├── requirements.txt
├── results/                 # Auto-generated outputs (not committed)
└── docs/
    ├── project_structure.md
    ├── experiment_log.md    # Results log — updated each phase
    └── phases/              # Detailed notes per phase
```

---

## Setup

```bash
# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
cd code

# Phase 1 — EDA
python 01_eda.py

# Phase 2 — Baseline
python 02_baseline.py

# Phase 3 — Feature Engineering
python 03_feature_engineering.py
```

Results are saved to `code/results/`.

---

## Results

### Baseline ANN — R² per dataset

| Dataset | Temp | R² |
|---------|------|----|
| dataset0 | 21°C | 0.9790 |
| dataset1 | 33°C | 0.2896 |
| dataset2 | 43°C | 0.6170 |
| dataset3 | 52°C | 0.9086 |
| dataset4 | 62°C | 0.7554 |

**Key finding:** Baseline is highly inconsistent across temperatures — motivates the research.

### Feature Engineering — R² by feature set

| Feature Set | ds0 (21°C) | ds1 (33°C) | ds2 (43°C) | ds3 (52°C) | ds4 (62°C) |
|-------------|-----------|-----------|-----------|-----------|-----------|
| baseline (E1+E2) | 0.996 | 0.940 | 0.313 | 0.742 | 0.777 |
| +Temperature | 0.993 | 0.841 | 0.597 | **0.864** | 0.725 |
| +delta_R +R_ratio | 0.996 | 0.934 | **0.897** | 0.805 | 0.775 |
| all features | 0.987 | **0.941** | 0.888 | 0.819 | 0.003 ⚠️ |

`delta_R + R_ratio` fixes dataset2 (0.31→0.90), but "all features" collapses on dataset4 — Dense(4) is too small. Next: upgrade architecture.

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Exploratory Data Analysis | ✅ Done |
| 2 | Baseline ANN (reproduce mentor model) | ✅ Done |
| 3 | Feature Engineering (Temperature, ΔR, R_ratio) | ✅ Done |
| 4 | Deep ANN + Dropout + BatchNorm | 🔲 Todo |
| 5 | LSTM / GRU (time-series) | 🔲 Todo |
| 6 | CNN-1D | 🔲 Todo |
| 7 | CNN + LSTM Hybrid | 🔲 Todo |
| 8 | Cross-temperature Generalization | 🔲 Todo |
| 9 | Hyperparameter Tuning (Optuna) | 🔲 Todo |

---

## Tech Stack

- Python 3.13
- TensorFlow / Keras
- scikit-learn
- pandas, numpy, matplotlib, seaborn

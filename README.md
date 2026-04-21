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

_Evaluated on ds1–ds4 only. 3 random seeds, scaler fit on train only, each change isolated._

### Phase 2 — Baseline (mean R² ± std, 3 seeds)

| Dataset | Temp | R² mean | ±std |
|---------|------|---------|------|
| ds1 | 33°C | 0.628 | ±0.445 |
| ds2 | 43°C | 0.342 | ±0.391 |
| ds3 | 52°C | 0.856 | ±0.030 |
| ds4 | 62°C | 0.720 | ±0.060 |
| **Mean** | | **0.637** | |

High std on ds1–ds2 = unstable, seed-sensitive — motivates the research.

### Phase 3 — Feature effect (mentor arch fixed, 3 seeds)

| Feature Set | ds1 | ds2 | ds3 | ds4 | mean |
|-------------|-----|-----|-----|-----|------|
| baseline (E1+E2) | 0.707 | 0.483 | 0.893 | 0.641 | 0.681 |
| **+delta_R +R_ratio** | **0.900** | **0.808** | 0.762 | 0.716 | **0.797** |
| all 5 features | 0.714 | 0.692 | 0.656 | 0.743 | 0.701 |

### Phase 4 — Architecture effect (2×2 ablation, 3 seeds)

| Cell | Architecture | Features | mean R² |
|------|-------------|---------|---------|
| A (baseline) | Dense(4) | E1, E2 | 0.541 |
| B | Dense(4) | all 5 | 0.685 — feature effect: +0.144 |
| **C (best)** | **Dense(64→32→16)** | **E1, E2** | **0.846** — arch effect: +0.305 |
| D | Dense(64→32→16) | all 5 | 0.772 |

**Architecture is the main driver (+0.305 vs +0.144 for features).**

### All Phases — Final Comparison (mean R²)

| Model | ds1 | ds2 | ds3 | ds4 | Mean |
|-------|-----|-----|-----|-----|------|
| Mentor baseline | 0.628 | 0.342 | 0.856 | 0.720 | 0.637 |
| ANN P4 (Cell C) | 0.908 | 0.835 | **0.910** | **0.731** | **0.846** |
| LSTM P5 | **0.937** | **0.859** | 0.843 | 0.663 | 0.826 |
| CNN-1D P6 | 0.900 | 0.782 | 0.838 | 0.647 | 0.792 |
| CNN+LSTM P7 | 0.897 | 0.856 | 0.885 | 0.575 | 0.803 |

**Best model overall: ANN P4 Cell C (mean R²=0.846).** LSTM P5 wins on ds1/ds2 (higher temperatures with complex patterns). Hybrid did not outperform standalone models.

### Phase 8 — Cross-temperature Generalization

**Key finding: both models fail completely on unseen temperatures (R² < 0).**

| Test DS | ANN R² (cross) | LSTM R² (cross) | Within-dataset ANN |
|---------|---------------|-----------------|-------------------|
| ds1 (33°C) | -14.38 | -40.33 | 0.908 |
| ds2 (43°C) | -0.72 | -0.54 | 0.835 |
| ds3 (52°C) | 0.013 | 0.010 | 0.910 |
| ds4 (62°C) | -2.12 | -8.01 | 0.731 |

Extrapolation (train 33–52°C → test 62°C): ANN R²=0.05, LSTM R²=-7.25 — both **❌ fail**.

Thermal drift between temperatures creates entirely different resistance scales. MinMaxScaler trained on other temperatures cannot normalize unseen ones. This confirms that per-temperature calibration or domain adaptation is required for deployment.

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Exploratory Data Analysis | ✅ Done |
| 2 | Baseline ANN (reproduce mentor model) | ✅ Done |
| 3 | Feature Engineering (Temperature, ΔR, R_ratio) | ✅ Done |
| 4 | Deep ANN + Dropout + BatchNorm | ✅ Done |
| 5 | LSTM (time-series, window=20) | ✅ Done |
| 6 | CNN-1D | ✅ Done |
| 7 | CNN+LSTM Hybrid | ✅ Done |
| 8 | Cross-temperature Generalization | ✅ Done |
| 9 | Hyperparameter Tuning (Optuna) | 🔲 Todo |

---

## Tech Stack

- Python 3.13
- TensorFlow / Keras
- scikit-learn
- pandas, numpy, matplotlib, seaborn

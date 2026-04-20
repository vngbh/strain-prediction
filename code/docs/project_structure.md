# Project Structure

_Cập nhật: 2026-04-21 — Phase 3 hoàn thành_

## Tổng quan

Dự án dự đoán **Strain** (biến dạng cơ học) từ tín hiệu điện trở của cảm biến fiber, sử dụng Deep Learning. Mục tiêu nâng cấp từ ANN cơ bản lên model xịn hơn, có khả năng generalize qua các mức nhiệt độ khác nhau.

**Input:** Electrode1_resistance, Electrode2_resistance (+ Temperature từ Phase 3 trở đi)
**Output:** Strain

---

## Cấu trúc thư mục

```
Lab/
├── code/                        ← toàn bộ source code
│   ├── utils/
│   │   ├── data_loader.py       ← load dataset, định nghĩa features
│   │   └── metrics.py           ← hàm evaluate (MSE, MAE, RMSE, R²)
│   │
│   ├── 01_eda.py                ← Phase 1: Exploratory Data Analysis
│   ├── 02_baseline.py           ← Phase 2: Reproduce ANN gốc của mentor
│   ├── 03_feature_engineering.py ← Phase 3: Temperature + derived features
│   │
│   ├── requirements.txt         ← dependencies
│   ├── results/                 ← output tự động (không commit)
│   │   ├── phase1_eda/
│   │   └── phase2_baseline/
│   │
│   └── docs/                    ← tài liệu dự án (commit cùng code)
│       ├── project_structure.md ← file này
│       ├── experiment_log.md    ← log kết quả từng phase
│       └── phases/
│           ├── phase1_eda.md
│           └── phase2_baseline.md
│
├── data/                        ← dataset (KHÔNG commit — bí mật)
│   ├── dataset0.csv             ← ~11k samples, 21°C
│   ├── dataset1.csv             ← ~23k samples, 33°C
│   ├── dataset2.csv             ← ~23k samples, 43°C
│   ├── dataset3.csv             ← ~21k samples, 52°C
│   └── dataset4.csv             ← ~23k samples, 62°C
│
├── .gitignore
└── CLAUDE.md
```

---

## Dataset

| File | Samples | Nhiệt độ | Experiments |
|------|---------|----------|-------------|
| dataset0.csv | 11,776 | ~21°C | [2] |
| dataset1.csv | 23,053 | ~33°C | [3, 4] |
| dataset2.csv | 23,227 | ~43°C | [5, 6] |
| dataset3.csv | 21,497 | ~52°C | [7, 8] |
| dataset4.csv | 23,077 | ~62°C | [9, 10] |

Columns: `Temperature`, `Strain`, `Electrode1_resistance`, `Electrode2_resistance`, `Experiment`, `Idx`

---

## Roadmap

| Phase | File | Trạng thái |
|-------|------|------------|
| 1 — EDA | `01_eda.py` | ✅ Done |
| 2 — Baseline ANN | `02_baseline.py` | ✅ Done |
| 3 — Feature Engineering | `03_feature_engineering.py` | ✅ Done |
| 4 — Deep ANN + Regularization | `04_models/ann_deep.py` | 🔲 Todo |
| 5 — LSTM / GRU | `04_models/lstm.py` | 🔲 Todo |
| 6 — CNN-1D | `04_models/cnn1d.py` | 🔲 Todo |
| 7 — CNN + LSTM Hybrid | `04_models/cnn_lstm.py` | 🔲 Todo |
| 8 — Cross-temperature Eval | `05_cross_dataset_eval.py` | 🔲 Todo |
| 9 — Hyperparameter Tuning | `06_hyperparameter_tuning.py` | 🔲 Todo |

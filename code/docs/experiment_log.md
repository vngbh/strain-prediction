# Experiment Log

_Mỗi lần chạy xong một phase, cập nhật kết quả vào đây._

---

## Quy tắc thí nghiệm (bất biến)

- **Dataset:** chỉ dùng ds1–ds4. ds0 bị loại (1 experiment, likely sample).
- **Scaler:** fit trên X_train, transform X_test riêng — không leak.
- **So sánh:** thay đổi 1 yếu tố mỗi lần (feature OR architecture, không cả 2).
- **Overfitting:** mỗi model phải có plot train vs val loss.
- **Seed:** cố định SEED=0, báo cáo nếu kết quả nhạy cảm với seed.

---

## Phase 1 — EDA (2026-04-21)

**Mục tiêu:** Hiểu cấu trúc data, phân phối Strain, ảnh hưởng của Temperature.

**Kết quả chính:**
- 5 datasets, tổng ~102k samples, nhiệt độ từ 21°C → 62°C
- Temperature gần như cố định trong mỗi dataset (dao động < 0.2°C)
- Strain range tương đồng giữa các dataset: [~0, ~0.5]
- Correlation của Electrode resistance với Strain rất thấp khi gộp tất cả datasets (~0.02) — vì mỗi dataset có scale resistance khác nhau → cần normalize per-dataset hoặc thêm derived features
- **Temperature không được dùng trong model gốc** — đây là điểm cải tiến rõ ràng

**Plots:** `results/phase1_eda/`

---

## Phase 2 — Baseline ANN (2026-04-21, revised)

**Mục tiêu:** Reproduce model gốc của mentor, lấy metrics làm baseline so sánh.

**Model:** Input(2) → Dense(4, relu) → Dense(1) | Adam(lr=1e-4), 50 epochs, batch=32
**Data:** ds1–ds4, shuffle=False, scaler fit on X_train only, 3 seeds (0/42/123)

| Dataset | Temp | R² mean | ±std | RMSE mean | Overfit? |
|---------|------|---------|------|-----------|----------|
| ds1 | 33°C | 0.628 | 0.445 | 0.069 | no |
| ds2 | 43°C | 0.342 | 0.391 | 0.110 | no |
| ds3 | 52°C | 0.856 | 0.030 | 0.056 | ⚠️ yes |
| ds4 | 62°C | 0.720 | 0.060 | 0.073 | ⚠️ yes |
| **Mean** | | **0.637** | | | |

**Nhận xét:**
- R² không ổn định, std cao trên ds1–ds2 (±0.4!) → model nhạy với seed
- ds3, ds4 có overfitting nhẹ (val > train loss)
- Đây là baseline tham chiếu cho mọi so sánh tiếp theo

**Plots:** `results/phase2_baseline/`

---

## Phase 3 — Feature Engineering (2026-04-21, revised)

**Mục tiêu:** Đo tác động features **thuần túy** — cùng kiến trúc mentor, chỉ đổi features.
**Data:** ds1–ds4, 3 seeds, scaler fit on X_train only.

| Feature Set | ds1 (±std) | ds2 (±std) | ds3 (±std) | ds4 (±std) | mean |
|-------------|-----------|-----------|-----------|-----------|------|
| baseline (E1+E2) | 0.707±0.16 | 0.483±0.28 | 0.893±0.03 | 0.641±0.13 | 0.681 |
| +Temperature | 0.817±0.05 | 0.397±0.48 | 0.897±0.04 | 0.697±0.04 | 0.702 |
| **+delta_R +R_ratio** | **0.900±0.05** | **0.808±0.09** | 0.762±0.07 | 0.716±0.03 | **0.797** |
| all 5 features | 0.714±0.22 | 0.692±0.14 | 0.656±0.02 | 0.743±0.02 | 0.701 |

**Insight:**
- `+delta_R +R_ratio` là feature set tốt nhất với mentor arch (mean 0.797)
- `+Temperature` không giúp nhiều — nhiệt độ cố định trong mỗi dataset nên không có thông tin mới
- `all 5 features` không tốt hơn `+delta_R +R_ratio` — Dense(4) không đủ capacity
- std cao trên ds2 (±0.28–0.48) → model vẫn nhạy với seed khi dùng mentor arch

---

## Phase 4 — Deep ANN Ablation (2026-04-21, revised)

**Thiết kế:** 2×2 ablation — đổi từng yếu tố riêng biệt.
**Data:** ds1–ds4, 3 seeds, scaler fit on X_train only.

| Cell | Architecture | Features | ds1 | ds2 | ds3 | ds4 | mean |
|------|-------------|---------|-----|-----|-----|-----|------|
| A | mentor (Dense 4) | E1, E2 | 0.879 | 0.111 | 0.580 | 0.593 | 0.541 |
| B | mentor (Dense 4) | all 5 | 0.724 | 0.554 | 0.723 | 0.737 | 0.685 |
| **C** | **medium (64→32→16)** | **E1, E2** | **0.908** | **0.835** | **0.910** | **0.731** | **0.846** |
| D | medium (64→32→16) | all 5 | 0.825 | 0.827 | 0.832 | 0.602 | 0.772 |

**Phân tích attribution:**
- Feature effect (A→B): +0.144
- **Architecture effect (A→C): +0.305** ← yếu tố chính
- C (medium + E1,E2) tốt hơn D (medium + all 5): thêm features không giúp thêm khi arch đã đủ mạnh
- **Best: Cell C**, mean R² **0.846 vs baseline 0.637**

**Overfitting:** val_loss > train_loss ở tất cả cells — cần regularization ở phase sau.

---

## Phase 5 — LSTM (2026-04-21)

**Thiết kế:** Sliding window=20, LSTM(64)→LSTM(32)→Dense(16)→Dense(1), 3 seeds, TimeSeriesSplit CV.

**R² mean ± std:**

| Dataset | ANN (Cell C) | LSTM | Winner |
|---------|-------------|------|--------|
| ds1 | 0.915 ±0.011 | **0.937** ±0.004 | LSTM ✅ |
| ds2 | 0.501 ±0.261 | **0.859** ±0.038 | LSTM ✅ |
| ds3 | **0.913** ±0.006 | 0.843 ±0.023 | ANN |
| ds4 | **0.741** ±0.021 | 0.663 ±0.019 | ANN |
| Mean | 0.768 | **0.825** | LSTM ✅ |

**Diagnostics:**
- ds2 ANN overfit nghiêm trọng (gap train/test=0.453) → LSTM fix: gap=0.086
- LSTM ổn định hơn nhiều trên ds2: std 0.261→0.038
- ds4 LSTM borderline overfit (gap=0.194) — cần cải thiện
- TimeSeriesSplit CV: ds1/ds3 stable, ds2/ds4 unstable (std>0.09)

**Kết luận:** LSTM tốt hơn overall (+0.058), đặc biệt fix được ds2. Nhưng ds3/ds4 vẫn prefer ANN → hybrid model (Phase 7) là hướng tiếp theo.

---

## Phase 6 — CNN-1D (2026-04-21)

| Dataset | ANN | LSTM | CNN-1D | Best |
|---------|-----|------|--------|------|
| ds1 | 0.908 | **0.937** | 0.900 ±0.021 | LSTM |
| ds2 | 0.835 | **0.859** | 0.782 ±0.119 | LSTM |
| ds3 | **0.910** | 0.843 | 0.838 ±0.004 | ANN |
| ds4 | **0.731** | 0.663 | 0.647 ±0.168 | ANN |
| Mean | **0.846** | 0.826 | 0.792 | ANN |

**Kết luận:** CNN-1D yếu nhất — overfit nặng trên ds2/ds4 (gap ~0.33–0.36), CV unstable.
Temporal dependencies trong data này là **long-range** → LSTM phù hợp hơn CNN-1D standalone.
CNN-1D sẽ được dùng làm feature extractor trong hybrid model (Phase 7).

---

## Phase 7 — CNN+LSTM Hybrid (2026-04-21)

| Dataset | Mentor | ANN P4 | LSTM P5 | CNN P6 | Hybrid P7 |
|---------|--------|--------|---------|--------|-----------|
| ds1 | 0.628 | 0.908 | **0.937** | 0.900 | 0.897 ±0.011 |
| ds2 | 0.342 | 0.835 | **0.859** | 0.782 | 0.856 ±0.047 |
| ds3 | 0.856 | **0.910** | 0.843 | 0.838 | 0.885 ±0.021 |
| ds4 | 0.720 | **0.731** | 0.663 | 0.647 | 0.575 ±0.136 |
| **Mean** | 0.637 | **0.846** | 0.826 | 0.792 | 0.803 |

**Kết luận:** Hybrid không thắng. ds4 collapse nặng (overfit gap=0.459). ANN P4 vẫn là best overall.

---

## Tổng kết tất cả phases

### Best model per dataset

| Dataset | Best Model | R² |
|---------|-----------|-----|
| ds1 (33°C) | LSTM P5 | **0.937** |
| ds2 (43°C) | LSTM P5 | **0.859** |
| ds3 (52°C) | ANN P4 | **0.910** |
| ds4 (62°C) | ANN P4 | **0.731** |
| **Overall** | **ANN P4 (mean 0.846)** | |

### Ranking by mean R²

| Rank | Model | Mean R² | vs Mentor |
|------|-------|---------|----------|
| 1 | ANN P4 (Cell C) | **0.846** | +0.209 |
| 2 | LSTM P5 | 0.826 | +0.189 |
| 3 | Hybrid P7 | 0.803 | +0.166 |
| 4 | CNN-1D P6 | 0.792 | +0.155 |
| — | Mentor baseline | 0.637 | — |

---

_Phase 8 (Cross-temperature generalization) và Phase 9 (Hyperparameter tuning) tiếp theo._

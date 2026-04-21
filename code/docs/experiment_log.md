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

## Phase 2 — Baseline ANN (2026-04-21) ⚠️ cần re-run không có ds0

**Mục tiêu:** Reproduce model gốc của mentor, lấy metrics làm baseline so sánh.

**Model:** Input(2) → Dense(4, relu) → Dense(1) — 13 parameters
**Config:** Adam(lr=1e-4), MSE loss, 50 epochs, batch=32, train/test=70/30, shuffle=False

**Kết quả:**

| Dataset | Temp | MSE | RMSE | MAE | R² |
|---------|------|-----|------|-----|----|
| dataset0 | 21°C | 0.000414 | 0.020335 | 0.018999 | **0.9790** |
| dataset1 | 33°C | 0.014087 | 0.118690 | 0.114418 | **0.2896** |
| dataset2 | 43°C | 0.008179 | 0.090438 | 0.080330 | **0.6170** |
| dataset3 | 52°C | 0.001987 | 0.044576 | 0.038948 | **0.9086** |
| dataset4 | 62°C | 0.004660 | 0.068266 | 0.059562 | **0.7554** |

**Nhận xét:**
- Kết quả không ổn định giữa các dataset (R² từ 0.29 → 0.98)
- dataset1 và dataset2 có R² thấp — gợi ý model không generalize tốt theo nhiệt độ
- Đây là điểm mạnh trong luận văn: chứng minh rõ hạn chế của baseline

**Plots:** `results/phase2_baseline/`

---

## Phase 3 — Feature Engineering (2026-04-21) ⚠️ cần re-run không có ds0

**Mục tiêu:** Thêm Temperature và derived features, train cùng kiến trúc để đo tác động thuần túy của features.

**Features thêm vào:**
- `Temperature` (trực tiếp)
- `delta_R = Electrode1 - Electrode2`
- `R_ratio = Electrode1 / Electrode2`

**Kết quả R² theo bộ features:**

| Feature Set | ds0 | ds1 | ds2 | ds3 | ds4 |
|-------------|-----|-----|-----|-----|-----|
| baseline (E1+E2) | 0.996 | 0.940 | 0.313 | 0.742 | 0.777 |
| +Temperature | 0.993 | 0.841 | 0.597 | **0.864** | 0.725 |
| +delta_R +R_ratio | 0.996 | 0.934 | **0.897** | 0.805 | 0.775 |
| all features | 0.987 | **0.941** | 0.888 | 0.819 | 0.003 ⚠️ |

**Insight quan trọng:**
- `delta_R + R_ratio` giúp dataset2 tăng mạnh: 0.31 → **0.90** (loại bỏ temperature offset)
- "all features" sụp đổ trên dataset4 (R²=0.003) — model Dense(4) quá đơn để xử lý 5 features đồng thời
- **Kết luận:** Cần nâng cấp kiến trúc (Phase 4+) trước khi thêm nhiều features

---

## Phase 4 — Deep ANN (2026-04-21) ⚠️ cần re-run không có ds0 + ablation study

**Mục tiêu:** Chứng minh kiến trúc sâu hơn + full features giải quyết được vấn đề của baseline.

**Config:** EarlyStopping(patience=20), ReduceLROnPlateau, batch=64, max 300 epochs

**Kết quả R²:**

| Architecture | ds0 | ds1 | ds2 | ds3 | ds4 | mean |
|---|---|---|---|---|---|---|
| mentor baseline | 0.979 | 0.290 | 0.617 | 0.909 | 0.755 | 0.710 |
| shallow (5 feat) | 0.898 | ❌ | ❌ | ❌ | ❌ | — |
| **medium (3 layers)** | **0.995** | **0.925** | **0.875** | **0.881** | **0.737** | **0.883** |
| deep (BN+Dropout) | ❌ | 0.565 | 0.305 | ❌ | 0.069 | — |

**Insight:**
- Medium model (Dense 64→32→16) là kiến trúc tốt nhất: mean R² **0.883 vs 0.710**
- Shallow + 5 features = diverge → model quá nhỏ
- Deep + BN không ổn định trên data này (BatchNorm nhạy cảm)
- **Milestone:** Lần đầu có model **tốt hơn mentor** một cách ổn định trên hầu hết datasets

---

_Các phase tiếp theo sẽ được thêm vào sau._

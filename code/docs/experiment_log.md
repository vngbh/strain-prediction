# Experiment Log

_Mỗi lần chạy xong một phase, cập nhật kết quả vào đây._

---

## Phase 1 — EDA (2026-04-21)

**Mục tiêu:** Hiểu cấu trúc data, phân phối Strain, ảnh hưởng của Temperature.

**Kết quả chính:**
- 5 datasets, tổng ~102k samples, nhiệt độ từ 21°C → 62°C
- Temperature gần như cố định trong mỗi dataset (dao động < 0.2°C)
- Strain range tương đồng giữa các dataset: [~0, ~0.5]
- Electrode resistance thay đổi rõ theo Strain (tương quan cao)
- **Temperature không được dùng trong model gốc** — đây là điểm cải tiến rõ ràng

**Plots:** `results/phase1_eda/`

---

## Phase 2 — Baseline ANN (2026-04-21)

**Mục tiêu:** Reproduce model gốc của mentor, lấy metrics làm baseline so sánh.

**Model:** Input(2) → Dense(4, relu) → Dense(1)
**Config:** Adam(lr=1e-4), MSE loss, 50 epochs, batch=32, train/test=70/30

**Kết quả:** _(chạy `02_baseline.py` để điền số vào đây)_

| Dataset | Temp | MSE | RMSE | MAE | R² |
|---------|------|-----|------|-----|----|
| dataset0 | 21°C | — | — | — | — |
| dataset1 | 33°C | — | — | — | — |
| dataset2 | 43°C | — | — | — | — |
| dataset3 | 52°C | — | — | — | — |
| dataset4 | 62°C | — | — | — | — |

**Plots:** `results/phase2_baseline/`

---

_Các phase tiếp theo sẽ được thêm vào sau._

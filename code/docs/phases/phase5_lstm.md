# Phase 5 — LSTM

## Thiết kế

- Sliding window size = 20 steps
- Input shape: (20, 2) — chuỗi 20 bước thời gian, 2 features (E1, E2)
- Architecture: LSTM(64, return_seq=True) → LSTM(32) → Dense(16) → Dense(1)
- So sánh fair với Phase 4 Cell C (cùng features, cùng split)
- Diagnostics built-in: overfitting check, residual corr, TimeSeriesSplit CV
- 3 seeds

## Chạy

```bash
cd /Users/vngbh/Lab/code
PYTHONPATH=. .venv/bin/python 05_models/lstm.py
```

## Kết quả — R² (mean ± std, 3 seeds)

| Dataset | ANN (Cell C) | LSTM | Δ R² | Winner |
|---------|-------------|------|------|--------|
| ds1 (33°C) | 0.915 ±0.011 | **0.937** ±0.004 | +0.021 | LSTM ✅ |
| ds2 (43°C) | 0.501 ±0.261 | **0.859** ±0.038 | +0.358 | LSTM ✅ |
| ds3 (52°C) | **0.913** ±0.006 | 0.843 ±0.023 | -0.070 | ANN |
| ds4 (62°C) | **0.741** ±0.021 | 0.663 ±0.019 | -0.078 | ANN |
| **Mean** | 0.768 | **0.825** | **+0.058** | **LSTM ✅** |

## Diagnostics

### Overfitting (seed=0, gap = train R² - test R²)

| Dataset | ANN gap | LSTM gap | |
|---------|---------|---------|--|
| ds1 | 0.070 ✅ | 0.051 ✅ | cả hai tốt |
| ds2 | **0.453 ⚠️** | **0.086 ✅** | ANN overfit nặng, LSTM fix được |
| ds3 | 0.035 ✅ | 0.059 ✅ | cả hai tốt |
| ds4 | 0.144 ⚠️ | 0.194 ⚠️ | cả hai borderline |

### Residual Correlation (seed=0)

| Dataset | ANN | LSTM | Cải thiện? |
|---------|-----|------|-----------|
| ds1 | 0.169 | **0.131** | ✅ |
| ds2 | 0.436 | 0.714 | ❌ tăng |
| ds3 | 0.276 | 0.469 | ❌ tăng |
| ds4 | 0.177 | 0.327 | ❌ tăng |

**Lưu ý:** LSTM có residual corr cao hơn nhưng R² cũng cao hơn — LSTM capture được global pattern nhưng introduce autocorrelation trong residuals. Đây là behaviour bình thường với sequence models.

### TimeSeriesSplit CV (5 folds, LSTM)

| Dataset | CV R² | ±std | Status |
|---------|-------|------|--------|
| ds1 | 0.930 | 0.033 | ✅ stable |
| ds2 | 0.810 | 0.130 | ⚠️ unstable |
| ds3 | 0.924 | 0.016 | ✅ stable |
| ds4 | 0.765 | 0.092 | ⚠️ unstable |

## Phân tích

**LSTM thắng overall** nhưng không phải ở mọi dataset:

- **ds1, ds2:** LSTM rõ ràng tốt hơn. ds2 đặc biệt ấn tượng: ANN overfit nặng (gap=0.45!), LSTM không overfit (gap=0.086). Kết quả ds2 tăng từ 0.50→0.86.
- **ds3, ds4:** ANN thắng nhẹ. Có thể do temporal patterns ở nhiệt độ cao đơn giản hơn, hoặc LSTM cần window lớn hơn.
- **ds2 std giảm mạnh:** ANN ±0.261 → LSTM ±0.038 — LSTM ổn định hơn nhiều trên dataset khó nhất.

**Finding cho luận văn:**
> LSTM giải quyết vấn đề overfitting nghiêm trọng của ANN trên ds2, nâng R² từ 0.50 lên 0.86. Tuy nhiên, với ds3/ds4 (nhiệt độ cao), ANN vẫn tốt hơn — gợi ý rằng một hybrid model (Phase 7) có thể tận dụng được điểm mạnh của cả hai.

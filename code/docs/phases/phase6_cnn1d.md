# Phase 6 — CNN-1D

## Thiết kế

- Sliding window = 20 (giống Phase 5)
- Architecture: Conv1D(64,k=3) → Conv1D(32,k=3) → GlobalAvgPool → Dense(16) → Dense(1)
- Cùng features, cùng split, cùng callbacks với Phase 5

## Kết quả — 3-way comparison

| Dataset | ANN (P4) | LSTM (P5) | CNN-1D (P6) | ±std | Best |
|---------|----------|-----------|-------------|------|------|
| ds1 | 0.908 | **0.937** | 0.900 | 0.021 | LSTM |
| ds2 | 0.835 | **0.859** | 0.782 | 0.119 | LSTM |
| ds3 | **0.910** | 0.843 | 0.838 | 0.004 | ANN |
| ds4 | **0.731** | 0.663 | 0.647 | 0.168 | ANN |
| **Mean** | **0.846** | 0.826 | 0.792 | | **ANN** |

## Diagnostics

### Overfitting (seed=0)

| Dataset | Gap | Status |
|---------|-----|--------|
| ds1 | 0.079 | ✅ |
| ds2 | **0.360** | ⚠️ overfit nặng |
| ds3 | 0.082 | ✅ |
| ds4 | **0.328** | ⚠️ overfit nặng |

### TimeSeriesSplit CV

| Dataset | CV R² | ±std | Status |
|---------|-------|------|--------|
| ds1 | 0.883 | 0.051 | ⚠️ |
| ds2 | 0.745 | 0.171 | ❌ rất unstable |
| ds3 | 0.856 | 0.079 | ⚠️ |
| ds4 | 0.754 | 0.100 | ⚠️ |

## Phân tích

CNN-1D là model yếu nhất trong 3 phases vì:

1. **Overfit ds2 và ds4** (gap ~0.33–0.36) — Conv1D không có cơ chế regularization nội tại như LSTM's gating
2. **CV không ổn định** — std cao (0.10–0.17) trên hầu hết datasets
3. **GlobalAveragePooling mất thông tin vị trí** — pooling trung bình toàn bộ sequence có thể bỏ sót pattern quan trọng ở cuối chuỗi

**Tuy nhiên:** CNN-1D là component tốt khi ghép với LSTM trong hybrid model (Phase 7) — CNN extract local features, LSTM capture long-range dependencies.

## Kết luận cho luận văn

CNN-1D standalone không phù hợp với dataset này. Đây là bằng chứng rằng temporal dependencies trong tín hiệu cảm biến có tính **long-range** (LSTM xử lý tốt hơn) hơn là **local** (CNN-1D).

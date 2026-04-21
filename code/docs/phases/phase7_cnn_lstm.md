# Phase 7 — CNN+LSTM Hybrid

## Thiết kế

- Architecture: Conv1D(64,k=3) → Conv1D(32,k=3) → LSTM(64) → Dense(16) → Dense(1)
- CNN giữ sequence structure (không dùng GlobalAvgPool) → LSTM nhận CNN features
- Cùng window=20, cùng features (E1+E2), 3 seeds, TimeSeriesSplit CV

## Kết quả — Full Comparison

| Dataset | Mentor | ANN P4 | LSTM P5 | CNN P6 | **Hybrid P7** | ±std |
|---------|--------|--------|---------|--------|---------------|------|
| ds1 (33°C) | 0.628 | 0.908 | **0.937** | 0.900 | 0.897 | 0.011 |
| ds2 (43°C) | 0.342 | 0.835 | **0.859** | 0.782 | 0.856 | 0.047 |
| ds3 (52°C) | 0.856 | **0.910** | 0.843 | 0.838 | 0.885 | 0.021 |
| ds4 (62°C) | 0.720 | **0.731** | 0.663 | 0.647 | 0.575 | 0.136 |
| **Mean** | 0.637 | **0.846** | 0.826 | 0.792 | 0.803 | |

## Diagnostics

### Overfitting (seed=0)

| Dataset | Gap | Status |
|---------|-----|--------|
| ds1 | 0.097 | ✅ |
| ds2 | **0.158** | ⚠️ |
| ds3 | 0.028 | ✅ |
| ds4 | **0.459** | ⚠️ overfit nặng |

### TimeSeriesSplit CV (5 folds)

| Dataset | CV R² | ±std | Status |
|---------|-------|------|--------|
| ds1 | 0.912 | 0.073 | ⚠️ |
| ds2 | 0.869 | 0.083 | ⚠️ |
| ds3 | 0.885 | 0.051 | ⚠️ |
| ds4 | 0.763 | 0.110 | ⚠️ |

## Phân tích

**Hybrid không thắng như kỳ vọng.** ANN P4 vẫn là model tốt nhất overall.

**Lý do ds4 collapse (R²=0.575, gap=0.459):**
- ds4 (62°C) là dataset khó nhất — thermal drift cao nhất
- Hybrid có nhiều parameters hơn → overfit mạnh hơn trên dataset này
- Với window=20, model không đủ context để generalize

**Tại sao Hybrid không beat standalone?**
- CNN+LSTM tăng complexity nhưng không tăng thông tin
- window=20 có thể quá ngắn cho CNN để extract features có ý nghĩa trước khi đưa vào LSTM
- Regularization (Dropout, L2) chưa được thêm vào

## Bảng tổng hợp cuối — Best model per dataset

| Dataset | Temp | Best Model | R² |
|---------|------|-----------|-----|
| ds1 | 33°C | LSTM P5 | **0.937** |
| ds2 | 43°C | LSTM P5 | **0.859** |
| ds3 | 52°C | ANN P4 | **0.910** |
| ds4 | 62°C | ANN P4 | **0.731** |
| **Overall** | | **ANN P4 (mean 0.846)** | |

## Kết luận cho luận văn

> CNN+LSTM Hybrid, dù có nền tảng lý thuyết vững, không cải thiện được kết quả so với standalone ANN và LSTM trên dataset này. Điều này cho thấy với window=20 và data từ cảm biến fiber, temporal dependencies được capture tốt hơn bởi LSTM đơn thuần (ds1/ds2) hoặc ANN với features thủ công (ds3/ds4). Kết quả này tự thân là một đóng góp khoa học: đặt ra câu hỏi về điều kiện để Hybrid model thực sự có lợi thế.

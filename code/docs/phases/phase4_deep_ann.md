# Phase 4 — Deep ANN

## Mục tiêu

Chứng minh rằng **nâng cấp kiến trúc** (không chỉ features) là yếu tố quyết định.
Dùng full feature set (E1, E2, Temp, ΔR, R_ratio) + 3 độ sâu khác nhau.

## Kiến trúc thử nghiệm

| Model | Layers | Params | LR |
|-------|--------|--------|----|
| shallow (baseline arch) | Dense(4) → Dense(1) | ~25 | 1e-4 |
| medium (3 layers) | Dense(64)→Dense(32)→Dense(16)→Dense(1) | ~3k | 1e-3 |
| deep (BN + Dropout) | Dense(128)→BN→Drop(0.1)→Dense(64)→BN→Drop(0.1)→Dense(32)→Dense(16)→Dense(1) | ~15k | 5e-4 |

**Config chung:** EarlyStopping(patience=20), ReduceLROnPlateau(patience=10), batch=64, max 300 epochs

## Chạy

```bash
cd /Users/vngbh/Lab/code
PYTHONPATH=. .venv/bin/python 04_models/ann_deep.py
```

## Kết quả

| Architecture | ds0 | ds1 | ds2 | ds3 | ds4 | mean |
|---|---|---|---|---|---|---|
| mentor baseline | 0.979 | 0.290 | 0.617 | 0.909 | 0.755 | 0.710 |
| shallow (5 features) | 0.898 | -32.7 | -5.4 | -30.6 | -11.6 | — |
| **medium (3 layers)** | **0.995** | **0.925** | **0.875** | **0.881** | **0.737** | **0.883** |
| deep (BN + Dropout) | -0.36 | 0.565 | 0.305 | -1.11 | 0.069 | — |

## Insight

- **Medium model** là kiến trúc tốt nhất Phase 4: mean R² **0.883 vs 0.710** (baseline)
- Shallow + 5 features = diverge hoàn toàn → kiến trúc nhỏ không xử lý được nhiều features
- Deep + BN vẫn không ổn định (hit early stopping ở epoch 20) → BN nhạy cảm với data này
- **Kết luận thesis:** Tăng depth vừa phải (medium) giúp tổng quát hóa tốt hơn; quá sâu cần tune thêm

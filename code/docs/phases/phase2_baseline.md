# Phase 2 — Baseline ANN

## Mục tiêu

Reproduce chính xác model gốc của mentor để có điểm so sánh (baseline) rõ ràng.
**Không được thay đổi gì** — mục đích là chứng minh điểm xuất phát.

## Kiến trúc

```
Input(2)  →  Dense(4, ReLU)  →  Dense(1)
```

- Input: `Electrode1_resistance`, `Electrode2_resistance` (đã normalize MinMax)
- Không dùng `Temperature`
- Optimizer: Adam(lr=1e-4)
- Loss: MSE
- Epochs: 50, Batch: 32
- Train/Test split: 70/30, shuffle=False (giữ thứ tự thời gian)

## Chạy

```bash
cd /Users/vngbh/Lab/code
python 02_baseline.py
```

## Output

- `results/phase2_baseline/baseline_metrics.json` — metrics đầy đủ
- `results/phase2_baseline/loss_curve_dataset{i}.png` — loss curve
- `results/phase2_baseline/prediction_dataset{i}.png` — ground truth vs predicted

## Hạn chế của baseline (sẽ giải quyết ở các phase sau)

| Hạn chế | Giải pháp |
|---------|-----------|
| Không dùng Temperature | Phase 3: thêm Temperature làm input |
| Kiến trúc quá đơn (13 params) | Phase 4: Deep ANN với Dropout, BatchNorm |
| Không khai thác time-series | Phase 5-6: LSTM, CNN-1D |
| Train riêng từng dataset | Phase 8: Cross-dataset evaluation |

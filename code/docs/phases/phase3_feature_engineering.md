# Phase 3 — Feature Engineering

## Mục tiêu

Đo tác động **thuần túy** của việc thêm features bằng cách giữ nguyên kiến trúc model (Dense(4)).
Cô lập được: tác động của features vs tác động của kiến trúc.

## Features thêm vào

| Feature | Công thức | Lý do |
|---------|-----------|-------|
| `Temperature` | raw | Cảm biến fiber bị ảnh hưởng bởi nhiệt (thermal drift) |
| `delta_R` | E1 - E2 | Chênh lệch điện trở — loại bỏ noise chung |
| `R_ratio` | E1 / E2 | Tỉ lệ — bất biến với temperature offset |

## Các bộ feature được thử

1. `baseline` — chỉ E1, E2 (giống model gốc)
2. `+Temperature` — thêm nhiệt độ
3. `+delta_R +R_ratio` — thêm derived features
4. `all features` — tất cả

## Chạy

```bash
cd /Users/vngbh/Lab/code
.venv/bin/python 03_feature_engineering.py
```

## Output

- `results/phase3_features/feature_metrics.json`
- `results/phase3_features/feature_comparison.png`

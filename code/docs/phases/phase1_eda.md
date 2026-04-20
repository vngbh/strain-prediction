# Phase 1 — Exploratory Data Analysis

## Mục tiêu

Trước khi xây bất kỳ model nào, cần hiểu rõ data:
1. Cấu trúc và kích thước từng dataset
2. Phân phối của Strain và Electrode resistance
3. Mức độ ảnh hưởng của Temperature lên Strain
4. Tín hiệu có tính time-series không? (quan trọng để quyết định dùng LSTM hay không)

## Câu hỏi cần trả lời

- [ ] Electrode resistance có tương quan tuyến tính với Strain không?
- [ ] Temperature khác nhau có làm thay đổi pattern R→Strain không?
- [ ] Tín hiệu có nhiễu nhiều không?
- [ ] 5 dataset có thể gộp chung để train không?

## Chạy

```bash
cd /Users/vngbh/Lab/code
python 01_eda.py
```

## Output

- `results/phase1_eda/dataset_summary.csv` — thống kê từng dataset
- `results/phase1_eda/correlation_heatmap.png` — tương quan giữa các features
- `results/phase1_eda/strain_distribution.png` — phân phối Strain theo nhiệt độ
- `results/phase1_eda/electrode_vs_strain.png` — scatter plot R vs Strain
- `results/phase1_eda/time_series.png` — dạng sóng tín hiệu theo thời gian

## Insight dự kiến

Temperature có thể gây shift trong mối quan hệ R→Strain (thermal drift), đây là lý do model gốc không generalize tốt qua các nhiệt độ khác nhau.

# Diagnostics — Scientific Validity

**Model được test:** Cell C — medium arch (Dense 64→32→16), features E1+E2, ds1–ds4.

## Test 1 — Train R² vs Test R² (overfitting check)

| Dataset | Train R² | Test R² | Gap | Status |
|---------|----------|---------|-----|--------|
| ds1 | 0.971 | 0.891 | 0.080 | ✅ ok |
| ds2 | 0.937 | 0.888 | 0.049 | ✅ ok |
| ds3 | 0.941 | 0.913 | 0.028 | ✅ ok |
| ds4 | 0.879 | 0.769 | **0.110** | ⚠️ borderline |

**Kết luận:** ds1–ds3 không overfit. ds4 borderline — cần regularization bổ sung ở phase sau.

---

## Test 2 — Residual Analysis

| Dataset | Shapiro-Wilk p | Normal? | Corr(resid, pred) | Uncorrelated? |
|---------|---------------|---------|-------------------|---------------|
| ds1 | ~0 | ❌ | 0.213 | ❌ |
| ds2 | ~0 | ❌ | 0.597 | ❌ |
| ds3 | ~0 | ❌ | 0.132 | ❌ |
| ds4 | ~0 | ❌ | 0.347 | ❌ |

**Giải thích — đây không phải failure:**

- Shapiro-Wilk p≈0 với n>5000 là **expected** — test này cực nhạy với sample lớn, ngay cả phân phối gần chuẩn cũng bị reject. Không có ý nghĩa thực tế.
- Correlation(residual, prediction) ≠ 0 **là finding quan trọng**: model ANN bỏ sót **systematic temporal patterns** trong tín hiệu. ds2 đặc biệt cao (0.597).
- **Đây là lý do để làm Phase 5 (LSTM):** ANN chỉ nhìn từng điểm, không khai thác chuỗi thời gian → residuals có cấu trúc → LSTM sẽ capture được.

---

## Test 3 — 5-Fold Cross-Validation (bằng chứng mạnh nhất)

| Dataset | CV R² mean | ±std | Status |
|---------|-----------|------|--------|
| ds1 | **0.931** | 0.031 | ✅ stable |
| ds2 | **0.911** | 0.035 | ✅ stable |
| ds3 | **0.942** | 0.010 | ✅ very stable |
| ds4 | **0.836** | 0.053 | ⚠️ slightly unstable |

**Ý nghĩa:** K-Fold không dùng fixed split — kết quả không phụ thuộc vào "may mắn" của cách chia data. R² cao và std thấp = **kết quả thực, không phải artifact.**

---

## Test 4 — Learning Curve

- ds1, ds3: test R² tăng dần theo training size → model học thực sự, không memorize
- ds2: test R² âm ở training size nhỏ, phục hồi khi dùng đầy đủ data → cần cả 2 experiments để generalize
- ds4: test R² tương đối ổn định quanh 0.7–0.8 → đã gần convergence

---

## Kết luận chung

| Tiêu chí | Kết quả | Đánh giá |
|---------|---------|---------|
| Overfitting | Gap < 0.11 | ✅ Acceptable |
| Reproducibility | K-Fold std < 0.05 | ✅ Results are real |
| Residual correlation | 0.13–0.60 | ⚠️ ANN bỏ sót temporal patterns |
| Learning behavior | Converging | ✅ Model học, không memorize |

**Finding cho luận văn:** ANN bỏ sót temporal patterns trong tín hiệu (residual có cấu trúc) → đây là **scientific justification** để chuyển sang LSTM/CNN-1D ở Phase 5–6.

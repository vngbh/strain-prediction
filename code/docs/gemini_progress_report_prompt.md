# Prompt tạo báo cáo tiến độ nghiên cứu trên Google Docs

## Cách sử dụng

Hãy đọc toàn bộ prompt tiếng Việt bên dưới, sau đó tạo một tài liệu báo cáo tiến độ **hoàn toàn bằng tiếng Nhật** trực tiếp trên Google Docs. Không tự ý thay đổi số liệu nghiên cứu và không tự suy đoán thông tin chưa được cung cấp.

---

## PROMPT DÀNH CHO GEMINI

Bạn là trợ lý viết báo cáo nghiên cứu khoa học chuyên ngành Machine Learning và cảm biến. Hãy đọc dữ liệu và yêu cầu bằng tiếng Việt dưới đây, sau đó tạo một **báo cáo tiến độ nghiên cứu hoàn toàn bằng tiếng Nhật** trên Google Docs.

Toàn bộ nội dung trong tài liệu cuối phải bằng tiếng Nhật tự nhiên, bao gồm tiêu đề, mục lục, nội dung phân tích, bảng, chú thích hình, kết luận và phụ lục. Chỉ giữ nguyên tiếng Anh đối với tên mô hình, thư viện, file, biến, thuật ngữ hoặc ký hiệu kỹ thuật cần thiết như ANN, LSTM, CNN, Optuna, TensorFlow, R², RMSE và `Electrode1_resistance`. Không để sót câu hoặc đoạn văn tiếng Việt trong tài liệu cuối.

### 1. Yêu cầu về tài liệu

- Loại tài liệu: Báo cáo tiến độ gửi giảng viên hướng dẫn.
- Ngôn ngữ đầu ra: Tiếng Nhật.
- Văn phong: tiếng Nhật học thuật, tự nhiên, rõ ràng và khách quan, phù hợp với báo cáo tiến độ của sinh viên đại học gửi giảng viên hướng dẫn.
- Sử dụng nhất quán thể văn viết trang trọng 「である調」.
- Không dịch từng chữ một cách máy móc. Hãy diễn đạt tự nhiên đối với người đọc Nhật Bản nhưng không thay đổi ý nghĩa khoa học.
- Độ dài mong muốn: khoảng 8-12 trang A4, chưa tính phụ lục.
- Không viết như quảng cáo hoặc phóng đại kết quả.
- Không bịa tài liệu tham khảo, số liệu, phương pháp hoặc kết luận.
- Giữ nguyên các giá trị R², độ lệch chuẩn và thông tin thí nghiệm được cung cấp.
- Giải thích thuật ngữ kỹ thuật ngắn gọn khi xuất hiện lần đầu.
- Phân biệt rõ:
  - Phase 1-9: đã hoàn thành và được ghi nhận chính thức.
  - Phase 10: đã chạy thử và có kết quả, nhưng đang ở trạng thái thử nghiệm, chưa hoàn tất tài liệu hóa.
- Dùng bảng để trình bày kết quả thay vì liệt kê quá nhiều số liệu trong đoạn văn.
- Không dùng emoji.
- Không sử dụng lời khẳng định nhân quả nếu thiết kế thí nghiệm chỉ cho phép kết luận về tương quan hoặc mức cải thiện thực nghiệm.
- Cuối tài liệu phải có phần công việc tiếp theo và kế hoạch dự kiến.

### 2. Định dạng Google Docs

Thiết lập tài liệu theo phong cách báo cáo học thuật:

- Khổ giấy A4.
- Font tiếng Nhật ưu tiên `Noto Serif JP`, `Yu Mincho` hoặc một font Mincho tương đương có sẵn trên Google Docs; cỡ chữ 12-13.
- Tiêu đề chính cỡ 16, in đậm, căn giữa.
- Heading 1 cỡ 15, Heading 2 cỡ 14.
- Giãn dòng 1.5.
- Căn đều hai lề cho nội dung.
- Lề trang tiêu chuẩn khoảng 2-2.5 cm.
- Đánh số bảng và hình bằng tiếng Nhật theo thứ tự: 表1, 表2, 図1, 図2...
- Mỗi bảng và hình phải có tên mô tả.
- Tạo mục lục tự động bằng Heading.
- Tạo số trang.
- Công thức và ký hiệu như R², ΔR phải hiển thị đúng.

### 3. Phần mở đầu tài liệu

Không tạo trang bìa hành chính và không yêu cầu bất kỳ thông tin cá nhân nào như tên trường, khoa, họ tên sinh viên, mã số sinh viên hoặc tên giảng viên.

Mở đầu trực tiếp bằng:

- Tiêu đề chính: **研究進捗報告書**
- Tiêu đề đề tài: **深層学習を用いた繊維型センサの電気抵抗信号からのひずみ予測**
- Phụ đề tiếng Anh, cỡ chữ nhỏ hơn: **Strain Prediction from Fiber-Based Sensor Signals Using Deep Learning**

Sau phần tiêu đề, tạo mục lục tự động rồi bắt đầu nội dung báo cáo.

### 4. Cấu trúc báo cáo bắt buộc

Viết báo cáo theo cấu trúc sau và sử dụng đúng các tiêu đề tiếng Nhật tương ứng trong Google Docs:

1. 研究背景
2. 研究目的および研究課題
3. データセットと問題設定
4. 研究方法および実験手順
5. 研究進捗および実験結果
6. モデルの総合比較
7. 結果の分析と主な知見
8. 現時点での課題と限界
9. 今後の研究計画
10. 進捗のまとめ
11. 付録

### 5. Bối cảnh nghiên cứu

Đề tài nghiên cứu việc dự đoán **strain**, tức biến dạng cơ học, từ tín hiệu điện trở của cảm biến sợi. Cảm biến ghi nhận thay đổi điện trở trên hai điện cực khi chịu biến dạng.

Thách thức quan trọng là đặc tính của cảm biến thay đổi theo nhiệt độ. Hiện tượng trôi nhiệt làm cho cùng một mức strain có thể tạo ra thang điện trở khác nhau tại các nhiệt độ khác nhau. Vì vậy, mô hình đạt kết quả tốt trong một điều kiện nhiệt độ chưa chắc hoạt động tốt ở nhiệt độ chưa xuất hiện trong dữ liệu huấn luyện.

Bài toán được xây dựng dưới dạng hồi quy:

- Đầu vào chính: `Electrode1_resistance`, `Electrode2_resistance`.
- Các đặc trưng đã thử nghiệm thêm: `Temperature`, `delta_R`, `R_ratio`.
- Đầu ra: `Strain`.

Mục tiêu của dự án:

- Xây dựng baseline dựa trên mô hình ANN ban đầu.
- Đánh giá tác động riêng của feature engineering và kiến trúc mạng.
- So sánh ANN, LSTM, CNN-1D và CNN-LSTM.
- Tối ưu siêu tham số bằng Optuna.
- Kiểm tra khả năng tổng quát hóa giữa các mức nhiệt độ.
- Xác định hạn chế chính và hướng nghiên cứu tiếp theo.

### 6. Dữ liệu

Dự án có 5 tập dữ liệu CSV:

| Dataset | Số mẫu | Nhiệt độ gần đúng | Trạng thái sử dụng |
|---|---:|---:|---|
| dataset0 | 11.776 | 21°C | Loại khỏi đánh giá chính vì chỉ có một experiment, có khả năng là dữ liệu mẫu |
| dataset1 | 23.053 | 33°C | Sử dụng |
| dataset2 | 23.227 | 43°C | Sử dụng |
| dataset3 | 21.497 | 52°C | Sử dụng |
| dataset4 | 23.077 | 62°C | Sử dụng |

Các cột dữ liệu:

- `Electrode1_resistance`
- `Electrode2_resistance`
- `Temperature`
- `Strain`
- `Experiment`
- `Idx`

Nhiệt độ gần như cố định trong từng dataset, với dao động nhỏ hơn khoảng 0,2°C. Khoảng giá trị strain giữa các dataset tương đối tương đồng, xấp xỉ từ 0 đến 0,5. Tuy nhiên, thang điện trở khác nhau đáng kể giữa các nhiệt độ.

### 7. Nguyên tắc bảo đảm tính hợp lệ của thí nghiệm

Hãy trình bày rõ các nguyên tắc sau trong phần phương pháp:

- Chỉ dùng dataset1-dataset4 cho đánh giá chính.
- Chia dữ liệu theo thứ tự thời gian với `shuffle=False`.
- Fit MinMaxScaler chỉ trên tập train, sau đó dùng scaler đó để transform tập test nhằm tránh data leakage.
- Khi so sánh, cố gắng chỉ thay đổi một yếu tố tại một thời điểm.
- Báo cáo mean và standard deviation trên ba seed: 0, 42 và 123 khi phù hợp.
- Dùng EarlyStopping và theo dõi train loss so với validation loss.
- Không chỉ báo cáo lần chạy tốt nhất.
- Các trường hợp R² âm hoặc mô hình thất bại vẫn phải được ghi nhận.
- Metric chính là hệ số xác định R²; ngoài ra có sử dụng MSE, MAE và RMSE ở một số phase.

Giải thích ngắn gọn rằng:

- R² càng gần 1 thì khả năng giải thích biến thiên của mô hình càng tốt.
- R² gần 0 tương đương với dự đoán bằng giá trị trung bình.
- R² âm cho thấy mô hình tệ hơn cách dự đoán bằng giá trị trung bình trên tập kiểm tra.

### 8. Tiến độ và kết quả theo từng phase

#### Phase 1 - Exploratory Data Analysis

Mục tiêu:

- Khảo sát cấu trúc, phân phối và mối quan hệ giữa các biến.

Kết quả:

- Có khoảng 102.000 mẫu trên toàn bộ 5 dataset.
- Nhiệt độ gần như cố định bên trong từng dataset.
- Correlation giữa điện trở và strain rất thấp khi gộp tất cả dataset, khoảng 0,02.
- Nguyên nhân quan sát được là mỗi nhiệt độ tạo ra một thang điện trở khác nhau.
- Kết quả EDA cho thấy cần chuẩn hóa theo dữ liệu huấn luyện và khảo sát ảnh hưởng của thermal drift.

#### Phase 2 - Baseline ANN

Kiến trúc:

`Input(2) -> Dense(4, ReLU) -> Dense(1)`

Thiết lập chính:

- Adam, learning rate 1e-4.
- 50 epochs, batch size 32.
- Ba seed.

Kết quả:

| Dataset | R² mean | Standard deviation |
|---|---:|---:|
| ds1 | 0,628 | 0,445 |
| ds2 | 0,342 | 0,391 |
| ds3 | 0,856 | 0,030 |
| ds4 | 0,720 | 0,060 |
| Trung bình | 0,637 | - |

Nhận xét:

- Baseline không ổn định, đặc biệt trên ds1 và ds2.
- Độ lệch chuẩn cao cho thấy mô hình nhạy với seed.

#### Phase 3 - Feature Engineering

Mục tiêu:

- Giữ nguyên kiến trúc baseline và chỉ thay đổi tập đặc trưng.

| Tập đặc trưng | ds1 | ds2 | ds3 | ds4 | Mean R² |
|---|---:|---:|---:|---:|---:|
| E1 + E2 | 0,707 | 0,483 | 0,893 | 0,641 | 0,681 |
| E1 + E2 + Temperature | 0,817 | 0,397 | 0,897 | 0,697 | 0,702 |
| E1 + E2 + delta_R + R_ratio | 0,900 | 0,808 | 0,762 | 0,716 | 0,797 |
| Toàn bộ 5 đặc trưng | 0,714 | 0,692 | 0,656 | 0,743 | 0,701 |

Kết luận:

- `delta_R` và `R_ratio` giúp baseline rõ rệt.
- Temperature không cung cấp nhiều thông tin khi gần như cố định trong từng dataset.
- Thêm tất cả đặc trưng không tự động làm kết quả tốt hơn.

#### Phase 4 - Deep ANN và ablation

Thiết kế 2x2 nhằm tách tác động của feature và architecture:

| Cell | Kiến trúc | Đặc trưng | Mean R² |
|---|---|---|---:|
| A | Dense(4) | E1, E2 | 0,541 |
| B | Dense(4) | Toàn bộ 5 đặc trưng | 0,685 |
| C | Dense(64 -> 32 -> 16) | E1, E2 | 0,846 |
| D | Dense(64 -> 32 -> 16) | Toàn bộ 5 đặc trưng | 0,772 |

Phân tích:

- Feature effect từ A sang B: +0,144 R².
- Architecture effect từ A sang C: +0,305 R².
- Cell C là cấu hình tốt nhất.
- Kết quả cho thấy kiến trúc là yếu tố cải thiện chính trong phạm vi thí nghiệm này.

#### Phase 5 - LSTM

Kiến trúc:

`LSTM(64, return_sequences=True) -> LSTM(32) -> Dense(16) -> Dense(1)`

Thiết lập:

- Sliding window 20 bước.
- Ba seed.
- Có kiểm tra TimeSeriesSplit.

| Dataset | ANN Phase 4 | LSTM Phase 5 |
|---|---:|---:|
| ds1 | 0,908 | 0,937 |
| ds2 | 0,835 | 0,859 |
| ds3 | 0,910 | 0,843 |
| ds4 | 0,731 | 0,663 |
| Mean | 0,846 | 0,826 |

Nhận xét:

- LSTM tốt hơn trên ds1 và ds2.
- LSTM giúp giảm tính bất ổn trên ds2.
- ANN vẫn tốt hơn trên ds3, ds4 và trung bình toàn bộ.

#### Phase 6 - CNN-1D

| Dataset | CNN-1D R² |
|---|---:|
| ds1 | 0,900 |
| ds2 | 0,782 |
| ds3 | 0,838 |
| ds4 | 0,647 |
| Mean | 0,792 |

Kết luận:

- CNN-1D không vượt ANN hoặc LSTM.
- Mô hình có dấu hiệu overfitting và bất ổn trên ds2, ds4.

#### Phase 7 - CNN-LSTM Hybrid

| Dataset | CNN-LSTM R² |
|---|---:|
| ds1 | 0,897 |
| ds2 | 0,856 |
| ds3 | 0,885 |
| ds4 | 0,575 |
| Mean | 0,803 |

Kết luận:

- Kiến trúc hybrid không vượt các mô hình đơn lẻ.
- ds4 giảm mạnh và có khoảng cách train-test lớn.
- Kết quả cho thấy tăng độ phức tạp không bảo đảm tăng độ chính xác.

#### Phase 8 - Cross-temperature Generalization

Mục tiêu:

- Kiểm tra khả năng dự đoán ở nhiệt độ chưa xuất hiện trong training.

Thiết kế:

- Leave-One-Dataset-Out: train trên ba nhiệt độ, test trên nhiệt độ còn lại.
- Extrapolation: train trên 33-52°C và test trên 62°C.

Kết quả leave-one-out:

| Dataset test | ANN R² | LSTM R² |
|---|---:|---:|
| ds1, 33°C | -14,376 | -40,335 |
| ds2, 43°C | -0,719 | -0,542 |
| ds3, 52°C | 0,013 | 0,010 |
| ds4, 62°C | -2,120 | -8,008 |

Kết quả extrapolation 33-52°C sang 62°C:

- ANN: R² mean = 0,050.
- LSTM: R² mean = -7,251.

Kết luận:

- Cả ANN và LSTM đều không tổng quát tốt sang nhiệt độ chưa thấy.
- Thermal drift là nút thắt chính của bài toán.
- MinMaxScaler học trên các nhiệt độ train không đủ để xử lý thang điện trở tại nhiệt độ mới.
- Đây là một kết quả nghiên cứu quan trọng, dù kết quả metric không cao.

#### Phase 9 - Hyperparameter Tuning bằng Optuna

Không gian tìm kiếm:

- 1-4 hidden layers.
- Units: 16, 32, 64 hoặc 128.
- Dropout: 0-0,4.
- Learning rate: 1e-4 đến 1e-2.
- Batch size: 32, 64 hoặc 128.
- 50 trials cho mỗi dataset.
- Search dùng seed 0; đánh giá cuối dùng ba seed.

| Dataset | ANN Phase 4 | Optuna Phase 9 | Standard deviation | Thay đổi |
|---|---:|---:|---:|---:|
| ds1 | 0,908 | 0,791 | 0,036 | -0,117 |
| ds2 | 0,835 | 0,897 | 0,035 | +0,062 |
| ds3 | 0,910 | 0,932 | 0,005 | +0,022 |
| ds4 | 0,731 | 0,779 | 0,006 | +0,048 |
| Mean | 0,846 | 0,850 | - | +0,004 |

Kết luận:

- Optuna đạt mean R² cao nhất là 0,850.
- Mức tăng trung bình chỉ là +0,004 so với ANN Phase 4.
- Hiệu suất ds1 giảm do cấu hình search trên một seed không ổn định khi đánh giá lại trên ba seed.
- Phase 4 đã ở gần vùng siêu tham số tốt.

#### Phase 10 - CWT và CNN-2D, trạng thái thử nghiệm

Ý tưởng:

- Chuyển cửa sổ tín hiệu một chiều thành scalogram hai chiều bằng Continuous Wavelet Transform.
- Dùng Morlet wavelet, 32 scales và window 64.
- Đưa scalogram của hai kênh điện trở vào CNN-2D.

Kết quả thử nghiệm:

| Dataset | CWT + CNN-2D R² mean | Standard deviation | CNN-1D | Thay đổi |
|---|---:|---:|---:|---:|
| ds1 | 0,855 | 0,087 | 0,900 | -0,045 |
| ds2 | 0,491 | 0,065 | 0,782 | -0,291 |
| ds3 | 0,853 | 0,043 | 0,838 | +0,015 |
| ds4 | 0,596 | 0,017 | 0,647 | -0,051 |
| Mean | 0,699 | - | 0,792 | -0,093 |

Kết luận tạm thời:

- CWT + CNN-2D chỉ cải thiện nhẹ trên ds3.
- Kết quả trung bình thấp hơn CNN-1D.
- Cách biểu diễn time-frequency hiện tại chưa tạo ra lợi ích tổng thể.
- Phase này chưa được xem là kết quả chính thức cuối cùng và cần phân tích thêm trước khi quyết định tiếp tục hoặc dừng hướng CWT.

### 9. Bảng so sánh tổng hợp bắt buộc

Tạo một bảng tổng hợp:

| Xếp hạng | Mô hình | Mean R² | Chênh lệch so với baseline |
|---:|---|---:|---:|
| 1 | Optuna Phase 9 | 0,850 | +0,213 |
| 2 | ANN Phase 4, Cell C | 0,846 | +0,209 |
| 3 | LSTM Phase 5 | 0,826 | +0,189 |
| 4 | CNN-LSTM Phase 7 | 0,803 | +0,166 |
| 5 | CNN-1D Phase 6 | 0,792 | +0,155 |
| 6 | CWT + CNN-2D Phase 10, thử nghiệm | 0,699 | +0,062 |
| - | Mentor baseline Phase 2 | 0,637 | - |

Đồng thời tạo bảng mô hình tốt nhất theo từng dataset:

| Dataset | Mô hình tốt nhất | R² |
|---|---|---:|
| ds1, 33°C | LSTM Phase 5 | 0,937 |
| ds2, 43°C | Optuna Phase 9 | 0,897 |
| ds3, 52°C | Optuna Phase 9 | 0,932 |
| ds4, 62°C | Optuna Phase 9 | 0,779 |

### 10. Các phát hiện nghiên cứu cần nhấn mạnh

Hãy phân tích, không chỉ lặp lại bảng số liệu:

1. Trong ablation Phase 4, cải thiện kiến trúc đóng góp lớn hơn feature engineering: +0,305 so với +0,144 R².
2. LSTM hữu ích khi dữ liệu có phụ thuộc thời gian và giúp ds2 ổn định hơn, nhưng không thắng ANN trên toàn bộ dataset.
3. Mô hình phức tạp hơn không mặc nhiên tốt hơn: CNN-LSTM và CWT-CNN đều không vượt ANN.
4. Hyperparameter tuning chỉ tăng trung bình +0,004, cho thấy ANN Phase 4 đã gần tối ưu trong không gian tìm kiếm hiện tại.
5. Vấn đề quan trọng nhất không còn là nâng R² trong cùng một nhiệt độ, mà là khả năng generalization qua nhiệt độ.
6. R² âm trong Phase 8 không nên bị che giấu; đây là bằng chứng thực nghiệm về ảnh hưởng nghiêm trọng của thermal drift.
7. Để triển khai thực tế, có thể cần hiệu chuẩn theo nhiệt độ, fine-tuning bằng một lượng nhỏ dữ liệu tại nhiệt độ mới hoặc domain adaptation.

### 11. Hạn chế cần trình bày trung thực

- Mỗi dataset gần như chỉ đại diện cho một mức nhiệt độ.
- Số lượng experiment độc lập trong mỗi dataset còn hạn chế.
- Kết quả tốt trong cùng dataset chưa chứng minh khả năng hoạt động ở môi trường mới.
- Optuna search dùng một seed nên có nguy cơ search overfitting.
- Phase 10 chưa có phân tích đầy đủ về lựa chọn wavelet, scales và window size.
- Chưa triển khai phương pháp domain adaptation.
- Chưa có đánh giá trên một bộ dữ liệu ngoài hoàn toàn độc lập.
- Chưa phân tích đầy đủ chi phí tính toán và khả năng chạy thời gian thực.

### 12. Công việc tiếp theo

Đề xuất kế hoạch theo thứ tự ưu tiên:

1. Hoàn thiện kiểm tra overfitting bằng cách báo cáo train R², test R² và learning curve cho các model chính.
2. Chuẩn hóa lại tài liệu và mã nguồn của Phase 10; quyết định giữ Phase 10 như kết quả âm hay tiếp tục tuning.
3. Xây dựng temperature-conditioned model với nhiệt độ là điều kiện đầu vào theo cách phù hợp hơn việc chỉ thêm một feature gần như hằng số.
4. Thử nghiệm fine-tuning với một lượng nhỏ mẫu calibration tại nhiệt độ mới.
5. Nghiên cứu domain adaptation, ví dụ Domain-Adversarial Neural Network.
6. So sánh với các phương pháp ML truyền thống như Random Forest, XGBoost hoặc SVR nếu cần một baseline bổ sung.
7. Đánh giá độ ổn định trên nhiều seed và nhiều cách chia dữ liệu hơn.
8. Chuẩn bị biểu đồ, bảng và phần mô tả phương pháp cho báo cáo cuối kỳ.

Không khẳng định chắc chắn các phương pháp tương lai sẽ cải thiện kết quả. Hãy trình bày đây là các giả thuyết và hướng cần kiểm chứng.

### 13. Hình ảnh đề xuất

Trong báo cáo, chèn các vị trí giữ chỗ và chú thích sau. Nếu có quyền truy cập file dự án, sử dụng hình tương ứng; nếu không, giữ nguyên placeholder để tôi bổ sung:

Các câu chú thích được cung cấp bên dưới bằng tiếng Việt chỉ là nội dung nguồn. Trong Google Docs, phải dịch toàn bộ chú thích hình sang tiếng Nhật và đánh số theo dạng `図1`, `図2`...

- `[CHÈN HÌNH: code/results/phase1_eda/correlation_heatmap.png]`
  - Chú thích: Ma trận tương quan giữa các biến trong dữ liệu cảm biến.
- `[CHÈN HÌNH: code/results/phase4_deep_ann/ablation_r2.png]`
  - Chú thích: So sánh kết quả ablation giữa tác động của đặc trưng và kiến trúc ANN.
- `[CHÈN HÌNH: code/results/phase7_cnn_lstm/all_phases_comparison.png]`
  - Chú thích: So sánh hiệu suất các kiến trúc từ Phase 2 đến Phase 7.
- `[CHÈN HÌNH: code/results/phase8_cross_temp/cross_temp_results.png]`
  - Chú thích: Kết quả đánh giá khả năng tổng quát hóa qua nhiệt độ.
- `[CHÈN HÌNH: code/results/phase9_optuna/optuna_vs_p4.png]`
  - Chú thích: So sánh ANN Phase 4 với mô hình được tối ưu bằng Optuna.
- `[CHÈN HÌNH: code/results/phase10_cwt_cnn/example_scalogram.png]`
  - Chú thích: Minh họa chuyển đổi tín hiệu điện trở sang scalogram bằng CWT.
- `[CHÈN HÌNH: code/results/phase10_cwt_cnn/cwt_vs_cnn1d.png]`
  - Chú thích: So sánh CWT + CNN-2D với CNN-1D.

Không mô tả nội dung chi tiết của hình nếu chưa thực sự quan sát được hình.

### 14. Kết luận tiến độ mong muốn

Phần kết luận cần thể hiện cân bằng các điểm sau:

- Dự án đã hoàn thành pipeline nghiên cứu từ EDA, baseline, feature engineering, thử nghiệm kiến trúc, tuning đến đánh giá cross-temperature.
- Trong điều kiện train và test cùng dataset, kết quả đã tăng từ baseline mean R² 0,637 lên 0,850.
- ANN sâu vẫn là lựa chọn cân bằng tốt giữa hiệu quả và độ phức tạp.
- LSTM có lợi thế trên một số dataset nhưng không tốt nhất tổng thể.
- Phase 10 cho kết quả thử nghiệm chưa tích cực.
- Thermal drift và khả năng tổng quát hóa sang nhiệt độ mới là vấn đề nghiên cứu trọng tâm cần giải quyết tiếp theo.
- Tiến độ hiện tại đủ để xác định rõ mô hình tham chiếu, giới hạn phương pháp hiện tại và hướng nghiên cứu kế tiếp.

### 15. Phụ lục

Tạo phụ lục ngắn gồm:

- Danh sách các phase và trạng thái.
- Danh sách công nghệ:
  - Python 3.13
  - TensorFlow/Keras
  - scikit-learn
  - pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Optuna
  - PyWavelets
- Cấu trúc mã nguồn chính:
  - `01_eda.py`
  - `02_baseline.py`
  - `03_feature_engineering.py`
  - `04_models/ann_deep.py`
  - `05_models/lstm.py`
  - `06_models/cnn1d.py`
  - `07_models/cnn_lstm.py`
  - `08_cross_temperature.py`
  - `09_hyperparameter_tuning.py`
  - `10_cwt_cnn.py`

### 16. Kiểm tra trước khi hoàn thành

Trước khi kết thúc, hãy tự kiểm tra:

- Mọi con số có đúng với dữ liệu đầu vào không?
- Phase 10 có được ghi rõ là thử nghiệm không?
- Có vô tình tuyên bố model generalize tốt qua nhiệt độ không?
- Có giải thích đúng ý nghĩa của R² âm không?
- Có phân biệt kết quả trong cùng dataset và cross-temperature không?
- Bảng xếp hạng có nhất quán với số liệu từng phase không?
- Báo cáo có phần hạn chế và kế hoạch tiếp theo không?
- Toàn bộ văn bản trong tài liệu cuối đã được viết bằng tiếng Nhật chưa?
- Có còn sót tiêu đề, bảng, chú thích hình hoặc đoạn văn tiếng Việt nào không?
- Văn phong 「である調」 có được sử dụng nhất quán không?
- Tài liệu có vô tình thêm tên trường, tên sinh viên, mã số sinh viên hoặc tên giảng viên không?

Sau khi kiểm tra, hãy hoàn thiện Google Docs và bảo đảm cấu trúc Heading cho phép tạo mục lục tự động.

# Dự báo US AQI tại Hà Nội

## 1. Tiêu đề và Giới thiệu (Title & Introduction)

Dự báo chất lượng không khí (nồng độ bụi mịn PM2.5) tại Hà Nội và quy đổi sang chỉ số US AQI chuẩn EPA.

- **Bài toán đang giải quyết:** Chuỗi thời gian đa biến (Multivariate Time Series Forecasting) dưới dạng Hồi quy (Regression), kết hợp phân loại ngầm định các mức độ cảnh báo chất lượng không khí.
- **Mục tiêu kinh doanh/nghiên cứu:** Cảnh báo mức độ ô nhiễm không khí tại 3 khung thời gian trong tương lai (t+1h, t+24h, t+72h) để hỗ trợ người dân và cơ quan đưa ra các quyết định sinh hoạt và lập kế hoạch dài hạn. Dự án áp dụng kiến trúc MLOps Level 1 với hệ thống quản lý siêu dữ liệu (metadata) nhằm tự động hóa quy trình.

## 2. Cấu trúc thư mục (Project Structure)

```text
PM25_NMKHDL/
├── data/
│   ├── raw/                            # Data Lake chứa dữ liệu gốc dạng Parquet (phân mảnh theo year/month)
│   └── features/                       # Dữ liệu bảng đặc trưng (features_targets.parquet)
├── metadata/                           # Các file theo dõi trạng thái cập nhật (data_freshness.json)
├── models/                             # Các mô hình XGBoost (.json) và SARIMA (.pkl) đã huấn luyện
├── reports/
│   └── figures/                        # Kết xuất biểu đồ phân tích (EDA, SHAP, kết quả Backtest)
├── src/
│   ├── pipelines/                      # Các luồng xử lý độc lập
│   │   ├── ingestion_pipeline.py       # Tải dữ liệu Delta từ API
│   │   ├── feature_pipeline.py         # Kỹ nghệ đặc trưng
│   │   ├── training_pipeline.py        # Huấn luyện mô hình và đánh giá
│   │   └── prediction_pipeline.py      # Module suy luận (Inference)
│   └── ui/
│       └── app_streamlit.py            # Giao diện web Dashboard
├── requirements.txt                    # Danh sách thư viện cần thiết
└── README.md                           # Tài liệu giới thiệu dự án
```

## 3. Môi trường và Cài đặt (Installation/Setup)

Khuyến nghị sử dụng Python 3.10 trở lên.

```bash
# Tạo môi trường ảo (Khuyến nghị)
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

## 4. Dữ liệu (Dataset)

Dự án **không** lưu trữ file dữ liệu lớn trực tiếp mà sử dụng Ingestion Pipeline để tự động tải và cập nhật.

- **Nguồn dữ liệu:**
  - Ô nhiễm không khí thực đo (PM2.5, PM10, CO): Trích xuất từ API của OpenAQ (Trạm đo tại Đại sứ quán Mỹ - Hà Nội).
  - Dữ liệu thời tiết (Nhiệt độ, độ ẩm, tốc độ gió, lượng mưa): Lấy từ API lịch sử (Archive) của Open-Meteo.
- **Quy trình tiền xử lý (Preprocessing):**
  - Chuẩn hóa Timezone về `Asia/Ho_Chi_Minh`.
  - Xử lý giá trị khuyết thiếu bằng nội suy tuyến tính (Linear Interpolation) cho khoảng trống ngắn và Forward Fill cho khoảng trống dài.
  - Khởi tạo các biến cờ (Missing flags) và gạt bỏ các giá trị ngoại lai (Extreme outliers) do lỗi cảm biến.
- **Cách tải dữ liệu:** Chạy lệnh `python -m src.pipelines.ingestion_pipeline` để hệ thống tự động kéo dữ liệu lịch sử và các điểm dữ liệu mới nhất.

## 5. Mô hình và Phương pháp (Methodology/Modeling)

Dự án sử dụng phương pháp **Direct Strategy** (Huấn luyện các bộ mô hình độc lập cho từng khung thời gian dự báo) để triệt tiêu hoàn toàn sai số lũy kế.

- **Kiến trúc mô hình:**
  - **XGBoost (XGBRegressor):** Thuật toán chính giúp nắm bắt các mối quan hệ phi tuyến tính phức tạp giữa nồng độ chất ô nhiễm và biến đổi thời tiết.
  - **SARIMA:** Mô hình thống kê cơ sở, hỗ trợ nắm bắt mạnh yếu tố chu kỳ/mùa vụ tuyến tính.
- **Kỹ nghệ đặc trưng (Feature Engineering - 35 biến):**
  - **Biến Độ trễ (Lag):** Trạng thái PM2.5 ở 1h, 24h, 72h trước đó.
  - **Biến Trượt (Rolling statistics):** Trung bình và độ lệch chuẩn của nồng độ PM2.5 trong các cửa sổ thời gian.
  - **Biến Chu kỳ (Cyclical):** Mã hóa Giờ, Tháng thành dạng `sin/cos` để bảo toàn tính tuần hoàn.
- **Giải thích mô hình (Explainable AI - XAI):**
  - Tích hợp công cụ **SHAP** để bóc tách độ quan trọng của các đặc trưng (Summary Plot) và lý giải cụ thể nguyên nhân tại thời điểm ô nhiễm nghiêm trọng (Waterfall Plot).

## 6. Đánh giá và Kết quả (Evaluation & Results)

Chiến lược kiểm định sử dụng Walk-forward Spliting (Train: 70%, Valid: 10%, Test: 20%). Mô hình được đánh giá khắt khe qua 5 kịch bản (K1 - K5). Dưới đây là kết quả của mô hình XGBoost:

| Khung thời gian (Horizon) | MAE (µg/m³) | RMSE (µg/m³) | F1-Macro (Cảnh báo AQI) |
| ------------------------- | ----------- | ------------ | ----------------------- |
| t+1h                      | 9.73        | 13.66        | 0.492                   |
| t+24h                     | 19.39       | 27.67        | 0.238                   |
| t+72h                     | 21.45       | 29.62        | 0.182                   |

_(Lưu ý: Các hình ảnh trực quan biểu đồ SHAP và Confusion Matrix được tạo tự động sau quá trình huấn luyện và lưu tại thư mục `reports/figures/`)_.

## 7. Hướng dẫn sử dụng (Usage)

### 7.1. Chạy Ứng dụng Web (Dashboard)

Để quan sát kết quả trực tiếp thông qua Bảng điều khiển (Inference):

```bash
streamlit run src/ui/app_streamlit.py
```

### 7.2. Luồng Thực thi (Pipeline Architecture)

Dự án được thiết kế theo kiến trúc **Module Separation**, nghĩa là các file xử lý lõi (như `src/modeling/train.py` hay `src/visualization/eda.py`) chỉ đóng vai trò là thư viện chứa hàm (module), **không thể tự chạy độc lập**.

Để toàn bộ hệ thống từ đầu đến cuối được vận hành, bạn chỉ cần chạy **3 Pipeline điều phối chính** theo đúng thứ tự sau:

```bash
# Pipeline 1: Tải và cập nhật dữ liệu Delta mới nhất từ API (Ingestion)
python -m src.pipelines.ingestion_pipeline

# Pipeline 2: Làm sạch dữ liệu và tính toán lại 35 đặc trưng thời gian (Feature Engineering)
# Lưu ý: Chỉ chạy khi Pipeline 1 có kéo được dữ liệu mới.
python -m src.pipelines.feature_pipeline

# Pipeline 3: Tự động gọi lần lượt Huấn luyện (Train), Đánh giá (Evaluate),
# Vẽ biểu đồ (EDA) và Phân tích SHAP (Explainability).
python -m src.pipelines.training_pipeline
```

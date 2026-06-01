# Dự báo US AQI tại Hà Nội

> Dự báo nồng độ PM2.5 tại Hà Nội với Machine Learning, chuyển đổi sang chỉ số **US AQI** chuẩn EPA, hỗ trợ 3 khung thời gian: **t+1h / t+24h / t+72h**.

---

## Cấu trúc project

```
PM25_NMKHDL/
├── data/
│   ├── raw/                            # Dữ liệu gốc từ API
│   ├── processed/                      # Dữ liệu sau ETL & timezone
│   └── features/
│       └── features_targets.parquet   # Cache features + targets (tần suất giờ)
│
├── models/                             # Mô hình đã huấn luyện
│   ├── xgb_t1.json                    #  XGBoost → t+1h
│   ├── xgb_t24.json                   #  XGBoost → t+24h
│   ├── xgb_t72.json                   #  XGBoost → t+72h
│   ├── sarima_t1.pkl                  #  SARIMA  → t+1h
│   ├── sarima_t24.pkl                 #  SARIMA  → t+24h
│   ├── sarima_t72.pkl                 #  SARIMA  → t+72h
│   └── best_model.joblib              #  Alias mô hình tốt nhất (dùng cho dashboard)
│
├── reports/
│   └── figures/                        # Biểu đồ EDA, SHAP, Confusion Matrix
│
├── src/
│   ├── config.py                       #  Hằng số, features, hyperparameters
│   ├── aqi_formula.py                  #  Chuyển đổi PM2.5 → US AQI (chuẩn EPA)
│   ├── main.py                         #  Pipeline runner (7 bước end-to-end)
│   ├── data/
│   │   └── load_data.py               # Tải dữ liệu thô từ file/API
│   ├── etl/
│   │   ├── preprocess.py              # Missing values, outliers, timezone
│   │   └── build_features.py          # Lag, rolling, cyclical features
│   ├── modeling/
│   │   ├── train.py                   # Huấn luyện, CV, lưu mô hình
│   │   └── evaluate.py                # 5 kịch bản backtesting (K1–K5)
│   ├── explainability/
│   │   └── explain.py                # SHAP Summary + Waterfall
│   ├── visualization/
│   │   └── eda.py                    # 11 biểu đồ EDA (time series, ACF/PACF...)
│   └── ui/
│       └── app_streamlit.py          # Dashboard Streamlit
│
├── requirements.txt
└── README.md
```

---

## 🚀 Cách chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy toàn bộ pipeline (7 bước)

```bash
python -m src.main
```

Pipeline tự động thực hiện:

1. **Nạp dữ liệu** — đọc `features_targets.parquet` (cache) hoặc chạy ETL đầy đủ
2. **Tiền xử lý** — xử lý missing values, outliers, timezone `Asia/Ho_Chi_Minh`
3. **Feature Engineering** — Lag, rolling, cyclical sin/cos, multi-horizon targets
4. **Huấn luyện & Lưu** — XGBoost + SARIMA cho 3 horizons (t+1 / t+24 / t+72)
5. **Backtesting 5 kịch bản** — K1 → K5 (xem bên dưới)
6. **EDA Plots** — 11 biểu đồ vào `reports/figures/`
7. **SHAP Explainability** — Summary Plot + Waterfall Plot

### 3. Chạy Dashboard Streamlit

```bash
streamlit run src/ui/app_streamlit.py
```

Dashboard hiển thị:

- **AQI hiện tại** với badge màu chuẩn EPA (Good → Hazardous)
- **Dự báo đa khung thời gian** t+1h / t+24h / t+72h (chọn qua sidebar)
- **SHAP Summary & Waterfall** — giải thích yếu tố ảnh hưởng nhất
- **Heatmap bản đồ** — phân bố AQI theo khu vực Hà Nội

---

## Mô hình & Chiến lược

### Chiến lược Multi-Horizon (Direct Method)

Thay vì dự báo tuần tự (accumulation error), dự án dùng **Direct Strategy**: 3 mô hình độc lập.

| Mô hình | Horizon | File lưu                |
| ------- | ------- | ----------------------- |
| XGBoost | t+1h    | `models/xgb_t1.json`    |
| XGBoost | t+24h   | `models/xgb_t24.json`   |
| XGBoost | t+72h   | `models/xgb_t72.json`   |
| SARIMA  | t+1h    | `models/sarima_t1.pkl`  |
| SARIMA  | t+24h   | `models/sarima_t24.pkl` |
| SARIMA  | t+72h   | `models/sarima_t72.pkl` |

### Post-processing: nồng độ → US AQI

Mô hình dự báo **nồng độ PM2.5 (µg/m³)**, sau đó `aqi_formula.py` chuyển sang US AQI:

| AQI     | Mức                     | Màu          |
| ------- | ----------------------- | ------------ |
| 0–50    | Good                    | 🟢 `#00E400` |
| 51–100  | Moderate                | 🟡 `#FFFF00` |
| 101–150 | Unhealthy for Sensitive | 🟠 `#FF7E00` |
| 151–200 | Unhealthy               | 🔴 `#FF0000` |
| 201–300 | Very Unhealthy          | 🟣 `#8F3F97` |
| 301–500 | Hazardous               | 🟤 `#7E0023` |

---

## Features (35 đặc trưng đầu vào)

| Nhóm                   | Đặc trưng                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| **Ô nhiễm thực đo**    | `pm2_5`, `pm10`, `co` (từ OpenAQ)                                                           |
| **Thời tiết**          | `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `precipitation` (từ Open-Meteo) |
| **Lag features**       | `pm2_5_lag_{1,2,3,6,12,24,48,72}h`                                                          |
| **Rolling statistics** | `pm2_5_roll_{mean,std}_{3,6,12,24}h` (causal: shift-before-roll)                            |
| **Cyclical time**      | `hour_sin/cos`, `month_sin/cos`, `day_of_week_sin/cos`                                      |
| **Missing flags**      | `pm2_5_missing_flag`, `temperature_2m_missing_flag`...                                      |

---

## Tài liệu

| Tài liệu                               | Nội dung                            |
| -------------------------------------- | ----------------------------------- |
| [`reports/figures/`](reports/figures/) | Biểu đồ EDA, SHAP, Confusion Matrix |

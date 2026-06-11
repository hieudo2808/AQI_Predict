# SUPERSEDED

This legacy report is retained for traceability only. It was created before the
corrected operational/oracle benchmark and before champion registry export.
Do not use the XGBoost or old baseline conclusions below as the final project
conclusion. The current source of truth is `reports/model_selection/`.

# Báo cáo model và so sánh thực nghiệm

Ngày tạo: 2026-06-08 18:35:20

## Kết luận nhanh

- Project hiện chạy bài toán dự báo `pm2_5` theo chuỗi thời gian đa biến, tần suất giờ.
- Model sản phẩm trong pipeline/UI là `XGBoost` (`xgboost.XGBRegressor`) theo direct strategy: mỗi horizon có một file `models/xgb_t{h}.json`.
- `best_model.joblib` hiện là bản dump của `XGBoost_t24` để phục vụ explainability/SHAP, không phải kết quả chọn lại tự động từ toàn bộ model.
- Base model/baseline trong code là `Naive Persistence` và `Seasonal Naive (7d)`. Trên hold-out hiện tại, baseline theo tuần đang là đối thủ rất mạnh.
- Với số liệu chạy lại trong script này, model thắng MAE theo từng horizon là:

|   horizon | model               |   MAE |   RMSE |     R2 |   F1_macro |
|----------:|:--------------------|------:|-------:|-------:|-----------:|
|         1 | Random Forest       |  3.47 |   6.24 | 0.9109 |     0.5692 |
|        24 | Seasonal Naive (7d) | 10.54 |  15.91 | 0.421  |     0.3865 |
|        48 | Seasonal Naive (7d) | 10.55 |  15.92 | 0.4206 |     0.3865 |
|        72 | Seasonal Naive (7d) | 10.56 |  15.93 | 0.42   |     0.3866 |

## Nguồn kiểm tra

- Dữ liệu: `data/features/features_targets.parquet`
- Code chia dữ liệu: `src/modeling/train.py::prepare_data`
- Code chạy model: `src/modeling/train.py::run_all_models`
- Code metric AQI-level: `src/modeling/evaluate.py::evaluate_classification_metrics`
- Script tạo báo cáo: archived/removed because the report is superseded.
- Output số liệu thô: `reports/archive/model_comparison_metrics.csv`

## Model đang được project sử dụng

UI (`src/ui/app_streamlit.py`) gọi `load_horizon_model(horizon)` và ưu tiên nạp:

1. `models/xgb_t{horizon}.json` bằng `xgboost.XGBRegressor().load_model(...)`
2. `models/sarima_t{horizon}.pkl` chỉ là fallback nếu thiếu XGBoost

Training pipeline (`src/pipelines/training_pipeline.py`) cũng nạp `xgb_t{h}.json` để backtesting K1-K5. Vì vậy model vận hành chính là `XGBoost`, không phải SARIMAX.

Metadata các file XGBoost đang có:

|   horizon | path                | exists   |   n_features |   n_boosted_rounds |
|----------:|:--------------------|:---------|-------------:|-------------------:|
|         1 | models\xgb_t1.json  | True     |           30 |                500 |
|        24 | models\xgb_t24.json | True     |           30 |                500 |
|        48 | models\xgb_t48.json | True     |           30 |                500 |
|        72 | models\xgb_t72.json | True     |           30 |                500 |

`best_model.joblib`:

| path                     | exists   | class        |   n_features |   n_boosted_rounds |
|:-------------------------|:---------|:-------------|-------------:|-------------------:|
| models\best_model.joblib | True     | XGBRegressor |           30 |                500 |

## Base model là gì?

Trong project này nên hiểu "base model" theo nghĩa baseline so sánh:

- `Naive Persistence`: dự báo tương lai bằng giá trị hiện tại/quá khứ gần nhất theo horizon.
- `Seasonal Naive (7d)`: dự báo bằng giá trị cùng giờ của chu kỳ tuần trước.

Đây là baseline thực sự trong `src/modeling/train.py`. XGBoost là primary/production model, không phải baseline.

Kết quả baseline:

|   horizon | model               |   rank_mae |   MAE |   RMSE |      R2 |   F1_macro |
|----------:|:--------------------|-----------:|------:|-------:|--------:|-----------:|
|         1 | Naive Persistence   |          6 |  3.6  |   6.67 |  0.8983 |     0.6153 |
|         1 | Seasonal Naive (7d) |          8 | 10.55 |  15.91 |  0.4211 |     0.3861 |
|        24 | Seasonal Naive (7d) |          1 | 10.54 |  15.91 |  0.421  |     0.3865 |
|        24 | Naive Persistence   |          6 | 12.66 |  17.83 |  0.273  |     0.3353 |
|        48 | Seasonal Naive (7d) |          1 | 10.55 |  15.92 |  0.4206 |     0.3865 |
|        48 | Naive Persistence   |          3 | 15.07 |  20.87 |  0.0043 |     0.2668 |
|        72 | Seasonal Naive (7d) |          1 | 10.56 |  15.93 |  0.42   |     0.3866 |
|        72 | Naive Persistence   |          3 | 16.91 |  23.78 | -0.2921 |     0.2415 |

## So sánh tại horizon mặc định t+24

`DEFAULT_HORIZON = 24` trong `src/config.py`, nên đây là horizon đang được lưu vào `best_model.joblib`.

|   rank_mae | model               |   MAE |   RMSE |     R2 |   F1_macro |   test_rows |
|-----------:|:--------------------|------:|-------:|-------:|-----------:|------------:|
|          1 | Seasonal Naive (7d) | 10.54 |  15.91 | 0.421  |     0.3865 |        4228 |
|          2 | Random Forest       | 11.47 |  15.33 | 0.4629 |     0.3188 |        4228 |
|          3 | Linear Regression   | 12.18 |  16.11 | 0.4069 |     0.3216 |        4228 |
|          4 | XGBoost             | 12.29 |  16.22 | 0.3985 |     0.306  |        4228 |
|          5 | LightGBM            | 12.33 |  16.26 | 0.3952 |     0.3014 |        4228 |
|          6 | Naive Persistence   | 12.66 |  17.83 | 0.273  |     0.3353 |        4228 |
|          7 | SARIMAX             | 13.97 |  18.06 | 0.2543 |     0.2621 |        4228 |
|          8 | Prophet             | 15.69 |  20.64 | 0.0254 |     0.2186 |        4228 |

Nhận xét: ở t+24, `Seasonal Naive (7d)` có MAE tốt nhất trong lần chạy này. `XGBoost` không thắng hold-out theo MAE, dù vẫn là model sản phẩm mà pipeline/UI đang dùng.

## XGBoost qua các horizon

|   horizon |   rank_mae |   MAE |   RMSE |      R2 |   F1_macro |
|----------:|-----------:|------:|-------:|--------:|-----------:|
|         1 |          3 |  3.51 |   6.31 |  0.9091 |     0.5726 |
|        24 |          4 | 12.29 |  16.22 |  0.3985 |     0.306  |
|        48 |          7 | 16.53 |  21.07 | -0.0151 |     0.2283 |
|        72 |          6 | 18.17 |  23.08 | -0.2178 |     0.222  |

## Bảng so sánh đầy đủ

|   horizon |   rank_mae | model               |   MAE |   RMSE |      R2 |   F1_macro |   test_rows |
|----------:|-----------:|:--------------------|------:|-------:|--------:|-----------:|------------:|
|         1 |          1 | Random Forest       |  3.47 |   6.24 |  0.9109 |     0.5692 |        4232 |
|         1 |          2 | LightGBM            |  3.49 |   6.3  |  0.9093 |     0.5673 |        4232 |
|         1 |          3 | XGBoost             |  3.51 |   6.31 |  0.9091 |     0.5726 |        4232 |
|         1 |          4 | Linear Regression   |  3.59 |   6.48 |  0.9039 |     0.6172 |        4232 |
|         1 |          4 | SARIMAX             |  3.59 |   6.49 |  0.9037 |     0.6168 |        4232 |
|         1 |          6 | Prophet             |  3.6  |   6.48 |  0.9039 |     0.6186 |        4232 |
|         1 |          6 | Naive Persistence   |  3.6  |   6.67 |  0.8983 |     0.6153 |        4232 |
|         1 |          8 | Seasonal Naive (7d) | 10.55 |  15.91 |  0.4211 |     0.3861 |        4232 |
|        24 |          1 | Seasonal Naive (7d) | 10.54 |  15.91 |  0.421  |     0.3865 |        4228 |
|        24 |          2 | Random Forest       | 11.47 |  15.33 |  0.4629 |     0.3188 |        4228 |
|        24 |          3 | Linear Regression   | 12.18 |  16.11 |  0.4069 |     0.3216 |        4228 |
|        24 |          4 | XGBoost             | 12.29 |  16.22 |  0.3985 |     0.306  |        4228 |
|        24 |          5 | LightGBM            | 12.33 |  16.26 |  0.3952 |     0.3014 |        4228 |
|        24 |          6 | Naive Persistence   | 12.66 |  17.83 |  0.273  |     0.3353 |        4228 |
|        24 |          7 | SARIMAX             | 13.97 |  18.06 |  0.2543 |     0.2621 |        4228 |
|        24 |          8 | Prophet             | 15.69 |  20.64 |  0.0254 |     0.2186 |        4228 |
|        48 |          1 | Seasonal Naive (7d) | 10.55 |  15.92 |  0.4206 |     0.3865 |        4223 |
|        48 |          2 | Linear Regression   | 14.05 |  17.93 |  0.2655 |     0.2776 |        4223 |
|        48 |          3 | Naive Persistence   | 15.07 |  20.87 |  0.0043 |     0.2668 |        4223 |
|        48 |          4 | SARIMAX             | 16.3  |  20.07 |  0.0797 |     0.228  |        4223 |
|        48 |          5 | LightGBM            | 16.37 |  20.79 |  0.0118 |     0.2318 |        4223 |
|        48 |          6 | Random Forest       | 16.47 |  20.93 | -0.0009 |     0.2308 |        4223 |
|        48 |          7 | XGBoost             | 16.53 |  21.07 | -0.0151 |     0.2283 |        4223 |
|        48 |          8 | Prophet             | 24.04 |  29.25 | -0.9552 |     0.0925 |        4223 |
|        72 |          1 | Seasonal Naive (7d) | 10.56 |  15.93 |  0.42   |     0.3866 |        4218 |
|        72 |          2 | Linear Regression   | 14.72 |  18.74 |  0.1972 |     0.2599 |        4218 |
|        72 |          3 | Naive Persistence   | 16.91 |  23.78 | -0.2921 |     0.2415 |        4218 |
|        72 |          4 | SARIMAX             | 17.51 |  21.4  | -0.0467 |     0.2037 |        4218 |
|        72 |          5 | LightGBM            | 17.78 |  22.59 | -0.1663 |     0.2263 |        4218 |
|        72 |          6 | XGBoost             | 18.17 |  23.08 | -0.2178 |     0.222  |        4218 |
|        72 |          7 | Random Forest       | 19.56 |  25.23 | -0.4551 |     0.2188 |        4218 |
|        72 |          8 | Prophet             | 23.97 |  29.78 | -1.0265 |     0.0932 |        4218 |

## Dữ liệu và split

- Số dòng feature table: `21,161`
- Số cột feature table: `50`
- Feature dùng train: `30`
- Weather features được shift bằng Perfect Prognosis: `temperature_2m, relative_humidity_2m, wind_speed_10m, pressure_msl, precipitation`
- Horizon đang cấu hình trong code: `[1, 24, 48, 72]`
- Target trong code: `pm2_5`

Tóm tắt target:

| stat   |     pm2_5 |   target_pm2_5_t1 |   target_pm2_5_t24 |   target_pm2_5_t48 |   target_pm2_5_t72 |
|:-------|----------:|------------------:|-------------------:|-------------------:|-------------------:|
| count  | 21161     |         21161     |          21138     |          21114     |          21090     |
| mean   |    37.596 |            37.595 |             37.588 |             37.583 |             37.582 |
| std    |    25.831 |            25.831 |             25.843 |             25.857 |             25.871 |
| min    |     0.4   |             0.4   |              0.4   |              0.4   |              0.4   |
| 50%    |    33     |            33     |             33     |             33     |             33     |
| max    |   383.6   |           383.6   |            383.6   |            383.6   |            383.6   |

## Hyperparameter model chính

```json
{
  "Random Forest": {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": 42
  },
  "XGBoost": {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0
  },
  "LightGBM": {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1
  },
  "Prophet": {
    "yearly_seasonality": true,
    "weekly_seasonality": true,
    "daily_seasonality": false
  },
  "SARIMAX": {
    "order": "(1, 0, 1)",
    "seasonal_order": "(0, 0, 0, 0)",
    "use_exog": true
  }
}
```

## Lưu ý kỹ thuật

- `configs/config.yaml` đang mô tả target `us_aqi`, frequency `daily`, horizons `[1, 3, 7]`. Code thực thi trong `src/config.py` lại dùng target `pm2_5`, frequency giờ ngầm định, horizons `[1, 24, 48, 72]`. Khi trình bày project, nên lấy `src/config.py` và pipeline hiện tại làm nguồn đúng.
- `README.md` có bảng kết quả XGBoost cũ. Báo cáo này là kết quả chạy lại trực tiếp trên dữ liệu `features_targets.parquet` hiện có.
- Việc Seasonal Naive thắng một số horizon cho thấy PM2.5 trong tập này có chu kỳ tuần mạnh. Nếu muốn giữ XGBoost làm model sản phẩm, nên bổ sung bước chọn model theo validation/test hoặc ensemble với baseline tuần.

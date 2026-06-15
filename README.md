# PM2.5 Hanoi Forecasting

Dự án dự báo PM2.5 theo giờ cho một trạm tại Hà Nội, quy đổi kết quả sang US AQI và hiển thị trên dashboard Streamlit. Luồng chuẩn của project là chạy đầy đủ dữ liệu và mô hình có thể chạy trong môi trường, so sánh bằng cùng một split thời gian, chọn champion theo `operational` leaderboard, rồi đưa champion đó ra giao diện.

Kết luận "model mạnh nhất" trong project này nghĩa là model có kết quả tốt nhất trên test holdout theo rule đã định: MAE thấp nhất theo từng horizon; nếu chênh trong 1% thì chọn model đơn giản hơn hoặc nhanh hơn. `oracle_weather` chỉ là upper-bound vì dùng actual future weather, không được dùng làm model chạy app.

## Flow Từ Đầu

Người dùng chưa có gì nên chạy theo thứ tự này:

| Bước | Lệnh chính                       | Kết quả                                      |
| ---- | -------------------------------- | -------------------------------------------- |
| 0    | Tạo môi trường và cài dependency | Python env chạy được toàn bộ pipeline        |
| 1    | Cấu hình `.env`                  | Có key để tải dữ liệu và chạy TabPFN nếu cần |
| 2    | Ingestion                        | Có raw PM2.5/weather data                    |
| 3    | Feature pipeline                 | Có `features_targets.parquet`                |
| 4    | Full model selection             | Có leaderboard và champion artifacts         |
| 5    | Dashboard                        | Giao diện dùng champion model để dự đoán     |
| 6    | Tests                            | Xác nhận codebase không lỗi                  |

## Bước 0: Tạo Môi Trường

Chạy một lần trên máy mới:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements-modeling.txt
```

`requirements-modeling.txt` đã bao gồm `requirements.txt`, nên đây là môi trường đầy đủ để chạy benchmark khoa học dữ liệu. File `requirements.txt` chỉ dùng cho môi trường chỉ chạy app, không phải flow chính để chọn model tốt nhất.

Kết quả mong đợi: import được các package chính; model nào không tương thích với Python/GPU/license hiện tại sẽ được pipeline ghi rõ trong model card.

Kiểm tra môi trường:

```powershell
.\.venv\Scripts\python.exe -m src.pipelines.model_selection_pipeline --preflight
```

## Bước 1: Cấu Hình Key

Tạo file `.env` ở thư mục gốc nếu chưa có:

```text
PURPLEAIR_READ_KEY=your_purpleair_key_here
TABPFN_TOKEN=your_priorlabs_key_here
TABPFN_MODEL_CACHE_DIR=models/model_selection/tabpfn_cache
```

`PURPLEAIR_READ_KEY` dùng để tải dữ liệu PM2.5 từ PurpleAir. Open-Meteo không cần key. `TABPFN_TOKEN` dùng khi chạy TabPFN; nếu không có hoặc TabPFN không chạy được, pipeline vẫn chạy các model còn lại và ghi lý do skip trong `reports/model_selection/model_cards.md`.

## Bước 2: Tải Dữ Liệu Raw

```powershell
.\.venv\Scripts\python.exe -m src.pipelines.ingestion_pipeline
```

Lệnh này đọc `metadata/data_freshness.json`, tải thêm dữ liệu PM2.5/PM10 từ PurpleAir và dữ liệu thời tiết từ Open-Meteo.

Kết quả sinh ra:

- `data/raw/aq/`
- `data/raw/weather/`
- `metadata/data_freshness.json`

Nếu chạy từ máy hoàn toàn mới, pipeline sẽ bắt đầu từ mốc mặc định `2024-01-01T00:00:00`.

## Bước 3: Tạo Feature Và Target

```powershell
.\.venv\Scripts\python.exe -m src.pipelines.feature_pipeline
```

Lệnh này merge air-quality data với weather data, xử lý missing values, tạo lag/rolling/calendar features và sinh target cho các horizon `t+1h`, `t+24h`, `t+48h`, `t+72h`.

Kết quả sinh ra:

- `data/features/features_targets.parquet`
- `metadata/feature_metadata.json`

File `features_targets.parquet` là đầu vào duy nhất cho bước chọn model.

## Bước 4: Chạy Full Model Selection

Đây là lệnh chính để so sánh toàn bộ model có thể chạy được trong môi trường:

```powershell
.\.venv\Scripts\python.exe -m src.pipelines.model_selection_pipeline --mode both --run-neuralforecast --export-champions
```

Lệnh này chạy:

- Corrected baselines: Persistence, SeasonalNaive, RollingMean.
- Tabular models: Ridge, ElasticNet, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost.
- Foundation/tabular model: TabPFN nếu dependency/token/môi trường cho phép.
- Sequential/time-series models: NeuralForecast models nếu dependency tương thích.
- `operational` mode để chọn champion thật cho app.
- `oracle_weather` mode để đo upper-bound, không dùng làm deployment champion.

Nguyên tắc benchmark: không loại model chỉ vì train lâu hoặc inference nặng. Runtime được ghi lại để phân tích trade-off, nhưng champion trước hết được chọn theo metric trên holdout test. TabPFN dùng toàn bộ training window hợp lệ như các model khác; project không tự cap số dòng train hoặc ép số estimator thấp, đồng thời bật `ignore_pretraining_limits=True` để TabPFN thử chạy cả khi số dòng vượt khuyến nghị chính thức 10,000 samples. Model chỉ bị skip khi môi trường thật sự không chạy được, ví dụ thiếu dependency, version không tương thích, thiếu token/license hoặc lỗi phần cứng; lý do skip phải nằm trong `model_cards.md`.

Kết quả sinh ra:

- `reports/model_selection/metrics.csv`: metric đầy đủ.
- `reports/model_selection/leaderboard.md`: leaderboard tổng.
- `reports/model_selection/final_leaderboard.md`: leaderboard operational dùng để chọn champion.
- `reports/model_selection/runtime.csv`: runtime từng model.
- `reports/model_selection/model_cards.md`: môi trường, model chạy/skip, manifest export.
- `models/champions/manifest.json`: registry champion cho dashboard.
- `models/champions/t*_*.joblib`: artifact champion theo horizon.

Nếu mục tiêu là báo cáo hoặc kết luận model nào tốt nhất, dùng kết quả từ lệnh này. Không dùng quick run để kết luận.

Champion hiện tại trong workspace:

| Horizon |     Champion |     MAE |    RMSE | Vai trò           |
| ------- | -----------: | ------: | ------: | ----------------- |
| t+1h    | RandomForest |  3.4821 |  6.2596 | Nowcast ngắn hạn  |
| t+24h   |        Ridge | 11.5001 | 15.7914 | Forecast vận hành |
| t+48h   |        Ridge | 12.6485 | 16.9014 | Forecast vận hành |
| t+72h   |        Ridge | 13.5593 | 18.0155 | Forecast vận hành |

## Bước 5: Chạy Dashboard

```powershell
.\.venv\Scripts\streamlit.exe run src/ui/app_streamlit.py
```

Dashboard sẽ:

- Đọc `data/features/features_targets.parquet`.
- Load `models/champions/manifest.json`.
- Dùng đúng champion model theo từng horizon.
- Hiển thị PM2.5 dự báo, AQI, cấp cảnh báo và tên model đang dùng.

Nếu chưa có champion registry, app có fallback sang artifact legacy, nhưng flow chuẩn là luôn chạy bước 4 trước để dashboard dùng champion mới nhất.

## Lệnh Quick Chỉ Để Debug

Lệnh này chỉ kiểm tra pipeline có chạy không:

```powershell
.\.venv\Scripts\python.exe -m src.pipelines.model_selection_pipeline --mode operational --quick --export-champions
```

Không dùng kết quả quick run để viết báo cáo, chọn model cuối hoặc kết luận khoa học dữ liệu.

## Cấu Trúc Kết Quả

```text
data/features/features_targets.parquet     # Feature table cuối cho model
models/champions/manifest.json             # Registry champion deployment
models/champions/*.joblib                  # Artifact champion theo horizon
reports/model_selection/                   # Metric, leaderboard, runtime, model cards
src/pipelines/ingestion_pipeline.py        # Lấy dữ liệu raw
src/pipelines/feature_pipeline.py          # Tạo feature/target
src/pipelines/model_selection_pipeline.py  # Chọn/export champion model
src/pipelines/prediction_pipeline.py       # Kiểm tra freshness cho dashboard
src/modeling/champion.py                   # Loader champion/legacy model
src/ui/app_streamlit.py                    # Dashboard Streamlit
```

Các báo cáo cũ trong `reports/archive/` và draft trong `docs/archive/` chỉ để truy vết quyết định cũ. Kết quả model hiện hành lấy từ `reports/model_selection/`.

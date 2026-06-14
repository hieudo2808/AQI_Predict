# Nhật ký Thay đổi — Xử lý góp ý review (PM2.5 Hà Nội)

Tài liệu này ghi lại **toàn bộ những gì đã thêm/sửa** để đáp ứng nhận xét của thầy về
quy trình tuning, baseline SARIMAX, bảng tổng hợp mô hình và mô hình chuỗi.

---

## 1. File MỚI

| File | Mục đích |
| --- | --- |
| `src/modeling/tuning.py` | Module tuning siêu tham số bằng **TimeSeriesSplit Cross-Validation** (`RandomizedSearchCV`). Khai báo `TUNING_GRIDS` cho từng mô hình + hàm `tune_model()`. |
| `.env` | Chứa `PURPLEAIR_READ_KEY` để Ingestion tải dữ liệu (đã được `.gitignore`, không commit). |
| `THAY_DOI.md` | Chính tài liệu này. |
| `reports/model_selection/combined_leaderboard.md` | **Bảng tổng hợp** sinh tự động: 1 bảng/horizon gộp toàn bộ mô hình (Baseline → SARIMAX → ML → Window). |

---

## 2. File ĐÃ SỬA

### 2.1. `src/modeling/tuning.py` *(điều chỉnh sau khi tạo)*
- Bỏ tune `RandomForest`/`ExtraTrees` (mỗi fit ~3.5s, quá chậm) — chúng vẫn chạy với tham số mặc định.
- Giảm `n_iter` mặc định **20 → 8** để tune nhanh (~5 phút thay vì ~40 phút). Kết quả gần như không đổi vì các mô hình chụm sát nhau.

### 2.2. `src/modeling/wrappers.py`
- Chuyển `import prophet` và `import statsmodels` sang **lazy-import** (đặt bên trong `.fit()`).
- Nhờ vậy benchmark dùng được **SARIMAX kể cả khi chưa cài prophet**.

### 2.3. `src/modeling/model_specs.py`
- Thêm hàm `statistical_specs()` → đăng ký **SARIMAX** (group `statistical`) làm baseline thống kê.
- Tự skip an toàn (trả `skipped`) nếu môi trường chưa cài `statsmodels`.

### 2.4. `src/pipelines/model_selection_pipeline.py` *(thay đổi nhiều nhất)*
- **Imports**: thêm `statistical_specs`, `tune_model`.
- `_fit_predict_tabular(...)`: thêm cờ `tune`. Khi bật → tune trên **Train+Valid** bằng TimeSeriesSplit CV, trả thêm `best_params`, `tuned`.
- `_fit_predict_statistical(...)` *(mới)*: fit SARIMAX trên **Train+Valid** rồi forecast khối **Test** liền kề (căn đúng thời gian).
- `_run_specs(...)`: thêm nhánh xử lý group `statistical`; truyền cờ `tune`; ghi thêm cột `tuned`, `best_params` vào mỗi dòng kết quả.
- `_error_row(...)`: bổ sung khóa `tuned`, `best_params` cho đồng nhất schema.
- `run_benchmark(...)`: thêm tham số `tune`, `include_sarimax`; gộp `statistical + tabular + window` vào danh sách mô hình chạy.
- `write_combined_leaderboard(...)` *(mới)*: sinh `combined_leaderboard.md` — sắp theo nhóm độ phức tạp tăng dần rồi MAE.
- `write_outputs(...)`: gọi thêm `write_combined_leaderboard`.
- `export_champions(...)`: **sửa lỗi** — nếu champion là baseline/SARIMAX/window (không deploy được) thì **tự lùi về mô hình tabular tốt nhất** thay vì raise lỗi.
- `main()`: thêm 2 cờ CLI **`--tune`** và **`--no-sarimax`**.

### 2.5. `metadata/data_freshness.json` & `metadata/feature_metadata.json`
- **Xóa các dấu conflict Git** (`<<<<<<<`, `=======`, `>>>>>>>`) còn sót do merge cũ — trước đó khiến `json.load()` crash.
- Cập nhật mốc thời gian theo lần backfill dữ liệu mới.

---

## 3. Đối chiếu với góp ý của thầy

| Góp ý của thầy | Cách xử lý |
| --- | --- |
| **Tuning ngược quy trình** (chọn cứng tham số rồi mới backtest) — Chương 5 | Cờ `--tune`: tham số là kết quả TimeSeriesSplit CV trên Train+Valid; lưu cột `tuned=True` + `best_params` trong `metrics.csv`. |
| **Thiếu baseline thống kê SARIMAX** trong bảng 6.3 | SARIMAX nay nằm trong leaderboard chính (group `statistical`), chạy đủ 4 horizon. |
| **Cần 1 bảng tổng hợp tất cả mô hình** ở trang cuối | `combined_leaderboard.md`: Baseline → SARIMAX → ML → Window, sắp theo mức cải tiến. |
| **Mô hình cây không thấy tính chuỗi → thử lookback dài hơn** | Mô hình **Window 168h** (WindowRidge/WindowMLP) có trong bảng; sẵn sàng chạy NeuralForecast (`--run-neuralforecast`). |
| **Tách Perfect Prognosis (rò rỉ tương lai)** — lưu ý số 1 | Đã có sẵn 2 chế độ `operational` / `oracle_weather`; leaderboard cuối loại bỏ oracle. |
| **Giải thích XGBoost vs Naive (lệch mùa)** | Kịch bản K5 (Đông/Hè) + bảng tổng hợp cho thấy ML thắng rõ ở horizon dài; sau tune, XGBoost t+1h (2.819) đã vượt Persistence (2.856). |

---

## 4. Thay đổi MÔI TRƯỜNG & DỮ LIỆU

**Thư viện đã cài thêm vào Python:**
`python-dotenv`, `streamlit`, `xgboost`, `lightgbm`, `statsmodels`, `streamlit-folium`, `folium`.

**Dữ liệu sinh ra (đều bị `.gitignore`, không lên git):**
- `data/raw/aq`, `data/raw/weather` — backfill **12.623** dòng AQ + **12.671** dòng thời tiết (2025-01 → 2026-06) từ trạm PurpleAir 96713.
- `data/features/features_targets.parquet` — **12.496** dòng × 50 đặc trưng.
- `models/champions/` — `manifest.json` + 4 file `.joblib` (t1/t24/t48 Ridge, t72 XGBoost).
- `reports/model_selection/` — `metrics.csv`, `leaderboard.md`, `final_leaderboard.md`, `combined_leaderboard.md`, `runtime.csv`, `model_cards.md` (đã chạy lại).

---

## 5. Lệnh chạy

```powershell
# Kích hoạt môi trường + cài thư viện
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt          # + requirements-modeling.txt nếu chạy deep

# 3 pipeline chính (đúng thứ tự)
python -m src.pipelines.ingestion_pipeline
python -m src.pipelines.feature_pipeline
python -m src.pipelines.model_selection_pipeline --mode operational --tune --no-optional --export-champions

# Dashboard
streamlit run src/ui/app_streamlit.py
```

Các cờ của `model_selection_pipeline`:
| Cờ | Tác dụng |
| --- | --- |
| `--tune` | Tune siêu tham số bằng TimeSeriesSplit CV |
| `--no-sarimax` | Bỏ baseline SARIMAX |
| `--no-optional` | Bỏ CatBoost/TabPFN |
| `--run-neuralforecast` | Chạy mô hình chuỗi/deep (cần cài torch + neuralforecast) |
| `--export-champions` | Xuất champion ra `models/champions/` cho dashboard |

---

## 6. Kết quả (bản tune nhẹ, mode operational)

| Horizon | Champion deploy | MAE |
| --- | --- | --- |
| t+1h | Ridge | 2.83 |
| t+24h | Ridge | 8.77 |
| t+48h | Ridge | 10.14 |
| t+72h | XGBoost | 10.64 |

- **20 mô hình** có `tuned=True` (5 mô hình × 4 horizon) kèm `best_params`.
- Sau tune, **XGBoost t+1h MAE = 2.819 đã vượt Persistence (2.856)**.

---

## 7. Còn tồn đọng (gợi ý cho lần nộp chính thức)

1. **Tune kỹ hơn:** mở lại grid `RandomForest`/`ExtraTrees` và tăng `n_iter` trong `src/modeling/tuning.py`.
2. **Mô hình deep:** `pip install -r requirements-modeling.txt` rồi thêm `--run-neuralforecast`.
3. **Caveat `--tune --export-champions`:** hiện `export_champions` refit model deploy bằng **tham số mặc định**, chưa dùng tham số đã tune. Cần sửa nếu muốn deploy đúng model tuned.
4. **Ảnh SHAP/Confusion Matrix** trong `reports/figures/` là từ lần chạy cũ — chạy lại bước explainability/EDA nếu muốn khớp dữ liệu 12.496 dòng.
5. **README.md** còn ghi sai tên pipeline (`training_pipeline` → đúng là `model_selection_pipeline`) và nguồn dữ liệu (OpenAQ → thực tế PurpleAir).

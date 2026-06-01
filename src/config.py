"""
Cấu hình trung tâm cho project PM2.5 Hà Nội.
Tất cả hằng số, danh sách features, hyperparameters được định nghĩa ở đây.
"""
import numpy as np

# ─── Đường dẫn dữ liệu ───
DATA_URL = 'data/hanoi-1581130/air_quality_historical.csv'
FIGURES_DIR = 'reports/figures'

# ─── Thông số API & Tọa độ (Hà Nội) ───
LATITUDE = 21.0285
LONGITUDE = 105.8542
TIMEZONE = "Asia/Ho_Chi_Minh"
OPENAQ_LOCATION_ID = 2161292  # Trạm Lưu Quang Vũ (do trạm US Embassy/AirNow cũ bị gián đoạn dữ liệu)

# ─── Seed & plotting ───
RANDOM_SEED = 42
FIGURE_SIZE = (13, 5)
FONT_SIZE = 11
PLOT_STYLE = 'whitegrid'

# ─── Biến mục tiêu ───
TARGET = 'pm2_5'

# ─── Multi-horizon forecasting (hourly) ───
HORIZONS = [1, 24, 72]  # t+1h, t+24h, t+72h
DEFAULT_HORIZON = 24    # Horizon mặc định khi chạy single-horizon
FORECAST_STRATEGY = 'direct'  # Mỗi horizon 1 model riêng

# ─── US AQI Breakpoints (EPA/AirNow) ───
AQI_BREAKPOINTS = [0, 50, 100, 150, 200, 300, 500]

AQI_LEVELS_US = [
    'Good',
    'Moderate',
    'Unhealthy for Sensitive Groups',
    'Unhealthy',
    'Very Unhealthy',
    'Hazardous',
]

AQI_LEVELS_VN = [
    'Tốt',
    'Trung bình',
    'Kém',
    'Xấu',
    'Rất xấu',
    'Nguy hại',
]

AQI_COLORS = [
    '#00e400',  # Green  - Good
    '#ffff00',  # Yellow - Moderate
    '#ff7e00',  # Orange - USG
    '#ff0000',  # Red    - Unhealthy
    '#8f3f97',  # Purple - Very Unhealthy
    '#7e0023',  # Maroon - Hazardous
]

# ─── Danh sách features (35 đặc trưng) ───
FEATURES = [
    # Chất ô nhiễm (dùng làm input, KHÔNG phải target)
    'pm10',
    'carbon_monoxide',
    'nitrogen_dioxide',
    'sulphur_dioxide',
    'ozone',

    # Lag features pm2_5 (lịch sử)
    'pm2_5_lag_1',
    'pm2_5_lag_2',
    'pm2_5_lag_3',
    'pm2_5_lag_6',
    'pm2_5_lag_12',
    'pm2_5_lag_24',
    'pm2_5_lag_48',
    'pm2_5_lag_168',

    # Rolling statistics pm2_5
    'pm2_5_roll6_mean',
    'pm2_5_roll24_mean',
    'pm2_5_roll72_mean',
    'pm2_5_roll24_std',
    'pm2_5_roll24_max',

    # Features cyclical (chu kỳ)
    'hour_sin',
    'hour_cos',
    'dow_sin',
    'dow_cos',
    'month_sin',
    'month_cos',
    'is_weekend',

    # Trend features
    'pm2_5_diff_1h',
    'pm2_5_diff_24h',

    # Weather features (từ Open-Meteo Weather API)
    'temperature_2m',
    'relative_humidity_2m',
    'wind_speed_10m',
    'pressure_msl',
    'precipitation',
]

# ─── Cột cần loại bỏ khỏi features (tránh data leakage) ───
LEAKAGE_COLUMNS = ['european_aqi']

# ─── Tỷ lệ train/validation/test (theo thời gian) ───
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
TEST_RATIO = 0.2

# ─── Cross-validation ───
CV_N_SPLITS = 5

# ─── Outlier detection ───
OUTLIER_N_SIGMA = 3

# ─── Cấu hình model ───
MODEL_CONFIGS = {
    'Random Forest': {
        'n_estimators': 300,
        'max_depth': 12,
        'min_samples_leaf': 2,
        'n_jobs': -1,
        'random_state': RANDOM_SEED,
    },
    'XGBoost': {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'verbosity': 0,
    },
    'LightGBM': {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'verbose': -1,
    },
    'Prophet': {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
    },
    'SARIMAX': {
        'order': (1, 0, 1),
        'seasonal_order': (0, 0, 0, 0),
        'use_exog': True,
    },
}

# ─── Màu sắc biểu đồ ───
MODEL_COLORS = ['#95a5a6', '#7f8c8d', '#2980b9', '#c0392b', '#27ae60', '#8e44ad', '#f39c12', '#d35400']
MODEL_SHORT_NAMES = ['Naive', 'SNaive', 'LR', 'RF', 'XGB', 'LGBM', 'Prophet', 'SARIMAX']

# ─── Seasonal definition (cho báo cáo metric theo mùa) ───
SEASONS = {
    'DJF': [12, 1, 2],   # Đông
    'MAM': [3, 4, 5],    # Xuân
    'JJA': [6, 7, 8],    # Hè
    'SON': [9, 10, 11],   # Thu
}

# ─── SHAP Analysis Constants ───
SHAP_MAX_SAMPLES = 1000   # Giới hạn samples cho SHAP để tăng tốc tính toán
SHAP_TOP_N = 15           # Top N features hiển thị trong SHAP summary plot

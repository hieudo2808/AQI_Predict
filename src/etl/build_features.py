"""
Module tạo đặc trưng (feature engineering) cho dữ liệu chuỗi thời gian PM2.5 Hà Nội.
"""
import logging
import numpy as np
import pandas as pd
from src.config import TARGET, HORIZONS

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng thời gian và chu kỳ từ index datetime.

    Bao gồm: month, day_of_week, day_of_year, year, hour, is_weekend,
    hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos.

    Parameters:
        df (pd.DataFrame): DataFrame sạch có index là DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame đã thêm features thời gian.
    """
    df = df.copy()

    # Đảm bảo index được sắp xếp
    df = df.sort_index()

    # Lấy các biến thời gian cơ bản từ index
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['year'] = df.index.year

    # Biến nhị phân cuối tuần và các mùa ở Hà Nội
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)  # Mùa đông HN
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)  # Mùa hè HN
    df['season'] = pd.cut(
        df['month'], bins=[0, 3, 6, 9, 12],
        labels=[1, 2, 3, 4], include_lowest=True
    ).astype(int)

    # Cyclical Encoding (Sine/Cosine) để giữ tính tuần hoàn thời gian
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    logger.info("Đã tạo đặc trưng thời gian và chu kỳ (cyclical features)")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các lag features và rolling statistics cho TARGET (pm2_5).

    - Lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h.
    - Rolling Mean: 6h, 24h, 72h.
    - Rolling Std: 24h.
    - Rolling Max: 24h.

    QUY TẮC AN TOÀN (Causal Rule): Rolling features phải được tính toán trên
    df[TARGET].shift(1) để tránh rò rỉ dữ liệu của thời điểm t.

    Parameters:
        df (pd.DataFrame): DataFrame sạch có index là DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame đã thêm lag và rolling features.
    """
    df = df.copy()

    # 1. Tạo Lag features
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        df[f'pm2_5_lag_{lag}'] = df[TARGET].shift(lag)

    # 2. Tạo Rolling features (strictly causal sử dụng shift(1))
    target_shifted = df[TARGET].shift(1)
    df['pm2_5_roll6_mean'] = target_shifted.rolling(window=6).mean()
    df['pm2_5_roll24_mean'] = target_shifted.rolling(window=24).mean()
    df['pm2_5_roll72_mean'] = target_shifted.rolling(window=72).mean()
    df['pm2_5_roll24_std'] = target_shifted.rolling(window=24).std()
    df['pm2_5_roll24_max'] = target_shifted.rolling(window=24).max()

    logger.info("Đã tạo lag và rolling features (không leak)")
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng xu hướng / chênh lệch nồng độ:
    - pm2_5_diff_1h: chênh lệch nồng độ so với giờ trước (t - t-1)
    - pm2_5_diff_24h: chênh lệch nồng độ so với cùng giờ ngày trước (t - t-24)

    Parameters:
        df (pd.DataFrame): DataFrame.

    Returns:
        pd.DataFrame: DataFrame có thêm trend features.
    """
    df = df.copy()

    df['pm2_5_diff_1h'] = df[TARGET].diff(1)
    df['pm2_5_diff_24h'] = df[TARGET].diff(24)

    logger.info("Đã tạo đặc trưng trend/difference")
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng tỷ lệ (ví dụ: PM2.5 / PM10).

    Parameters:
        df (pd.DataFrame): DataFrame.

    Returns:
        pd.DataFrame: DataFrame có thêm ratio features.
    """
    df = df.copy()

    if 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df[TARGET] / (df['pm10'] + 1e-6)
        logger.info("Đã tạo đặc trưng tỷ lệ PM2.5/PM10")
    else:
        logger.warning("Cột pm10 không tồn tại để tính tỷ lệ pm2.5/pm10.")

    return df


def add_multi_horizon_targets(df: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """
    Tạo các cột target cho chiến lược Direct Multi-horizon forecasting.
    Target = pm2_5.shift(-h) (nồng độ tại thời điểm t+h).

    Parameters:
        df (pd.DataFrame): DataFrame.
        horizons (list, optional): Các horizon (giờ). Mặc định HORIZONS [1, 24, 72].

    Returns:
        pd.DataFrame: DataFrame đã thêm các cột target.
    """
    df = df.copy()
    h_list = horizons or HORIZONS

    for h in h_list:
        col_name = f'target_{TARGET}_t{h}'
        df[col_name] = df[TARGET].shift(-h)
        n_valid = df[col_name].notna().sum()
        logger.info(f"   → {col_name}: {n_valid} mẫu có target (mất {h} dòng cuối)")

    logger.info(f"Đã tạo các cột target multi-horizon: {[f't+{h}' for h in h_list]}")
    return df


def build_all_features(df: pd.DataFrame, horizons: list = None, is_inference: bool = False) -> pd.DataFrame:
    """
    Orchestrate toàn bộ quy trình tạo đặc trưng và target cho chuỗi thời gian PM2.5.

    Gọi lần lượt: time features -> lag/rolling features -> trend features -> ratio features -> targets.

    Parameters:
        df (pd.DataFrame): DataFrame sạch sau tiền xử lý (đã qua timezone localization).
        horizons (list, optional): Horizons dự báo. Mặc định HORIZONS [1, 24, 72].
        is_inference (bool): Bật True khi Dự báo thực tế (Inference) để không drop dòng cuối do khuyết target.

    Returns:
        pd.DataFrame: DataFrame hoàn chỉnh với đầy đủ đặc trưng và target, cột 'date' được reset lại.
    """
    # Đảm bảo index là DatetimeIndex trước khi tính toán
    time_col = None
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ['datetime', 'date', 'time']:
            if col in df.columns:
                time_col = col
                break
        if time_col is not None:
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df = df.copy()
            df.index = pd.to_datetime(df.index)

    df.index.name = 'datetime'

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_trend_features(df)
    df = add_ratio_features(df)
    df = add_multi_horizon_targets(df, horizons)

    h_list = horizons or HORIZONS
    target_cols = [f'target_{TARGET}_t{h}' for h in h_list]

    if is_inference:
        # Trong Inference, giữ lại các dòng cuối có target NaN nhưng features đầy đủ
        # Lags tối đa là 168, nên loại bỏ 168 dòng đầu bị khuyết đặc trưng lag
        feature_cols = [c for c in df.columns if c not in target_cols]
        df = df.dropna(subset=[c for c in feature_cols if df[c].dtype != object])
    else:
        # Trong Training, loại bỏ các dòng bị NaN ở features và các dòng NaN ở target nhỏ nhất
        min_h = min(h_list)
        check_cols = [c for c in df.columns if c not in target_cols or c == f'target_{TARGET}_t{min_h}']
        df = df.dropna(subset=[c for c in check_cols if df[c].dtype != object])

    # Đảm bảo index tăng dần
    df = df.sort_index()
    assert df.index.is_monotonic_increasing, "Lỗi: Index thời gian không tăng dần đơn điệu!"

    # Reset index đưa cột thời gian trở lại tên 'date' để tương thích ngược 100% với codebase
    df = df.reset_index()
    df = df.rename(columns={'datetime': 'date'})

    logger.info(f"Hoàn tất tạo đặc trưng. Dataset cuối: {len(df)} dòng.")
    if 'date' in df.columns and len(df) > 0:
        logger.info(f"   Thời gian: {df['date'].min()} → {df['date'].max()}")

    return df

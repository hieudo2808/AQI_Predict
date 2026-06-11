"""
Module tiền xử lý dữ liệu: xử lý missing values và loại bỏ outlier.
"""
import logging
import numpy as np
import pandas as pd
from src.config import TARGET

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý chuỗi thời gian:
    1. Chuẩn hóa Datetime Index với timezone Asia/Ho_Chi_Minh.
    2. Sắp xếp index theo thời gian tăng dần.
    3. Đảm bảo tần suất Hourly (1H) bằng asfreq('h').
    4. Gán outlier nồng độ cực đoan (< 0 hoặc > 1000) về NaN.
    5. Tạo các cột *_missing_flag lưu trữ trạng thái missing (1: missing ban đầu, 0: không missing) với kiểu int8.
    6. Xử lý missing values bằng nội suy tuyến tính (giới hạn limit=3 cho gaps ngắn).
    7. Forward fill cho gaps ở cuối chuỗi.
    8. Fill đầu chuỗi bằng giá trị hợp lệ đầu tiên (tránh look-ahead leakage).

    Parameters:
        df (pd.DataFrame): DataFrame thô cần xử lý.

    Returns:
        pd.DataFrame: DataFrame đã xử lý hoàn chỉnh.
    """
    df = df.copy()

    # 1. Chuẩn hóa Datetime Index
    time_col = None
    if isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        for col in ['datetime', 'date', 'time']:
            if col in df.columns:
                time_col = col
                break
        if time_col is not None:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)

    df.index.name = 'datetime'

    # Timezone localization
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Ho_Chi_Minh')
    else:
        df.index = df.index.tz_convert('Asia/Ho_Chi_Minh')

    # 2. Sắp xếp index
    df = df.sort_index()

    # 3. Đảm bảo tần suất Hourly
    df = df.asfreq('h')

    # Truncate: Cắt bỏ khoảng thời gian đầu chuỗi nếu PM2.5 hoàn toàn bị missing (VD: Từ Jan 2024 đến Oct 2025)
    if TARGET in df.columns:
        first_valid_target = df[TARGET].first_valid_index()
        if first_valid_target is not None:
            df = df.loc[first_valid_target:].copy()
            logger.info(f"Đã cắt bỏ dữ liệu rác trước {first_valid_target} (vì không có dữ liệu đo thực tế)")

    # Báo cáo missingness trước xử lý
    logger.info("--- Báo cáo missingness trước xử lý ---")
    missing_before = df.isnull().sum()
    for col, count in missing_before.items():
        if count > 0:
            logger.info(f"Cột {col}: missing {count} ({count/len(df)*100:.2f}%)")

    # 4. Gán outlier nồng độ cực đoan (< 0 hoặc > 1000) về NaN
    # Áp dụng cho các cột chất ô nhiễm / đo lường chính
    pollution_cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'ozone', 'value', 'us_aqi']
    cols_to_clean = [c for c in pollution_cols if c in df.columns]
    for col in cols_to_clean:
        outliers_mask = (df[col] < 0) | (df[col] > 1000)
        num_outliers = outliers_mask.sum()
        if num_outliers > 0:
            logger.warning(f"Gán {num_outliers} giá trị cực đoan (<0 hoặc >1000) trong cột {col} về NaN")
            df.loc[outliers_mask, col] = np.nan

    # 5. Sinh cột missing flag (int8) trước khi điền khuyết cho tất cả cột numeric ban đầu
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[f"{col}_missing_flag"] = df[col].isnull().astype('int8')

    # 6. Xử lý missing values bằng nội suy tuyến tính (phù hợp chuỗi thời gian)
    for col in numeric_cols:
        # Bước 1: Linear interpolation cho các gaps ở giữa chuỗi, giới hạn limit=3
        df[col] = df[col].interpolate(method='linear', limit=3)
        # Bước 2: Forward fill cho các gaps ở cuối chuỗi
        df[col] = df[col].ffill()
        # Bước 3: Fill đầu chuỗi bằng giá trị hợp lệ đầu tiên (tránh look-ahead leakage)
        first_valid_idx = df[col].first_valid_index()
        if first_valid_idx is not None:
            first_val = df[col].loc[first_valid_idx]
        else:
            first_val = 0.0
        df[col] = df[col].fillna(value=first_val)

    # Báo cáo missingness sau xử lý
    logger.info("--- Báo cáo missingness sau xử lý ---")
    missing_after = df[numeric_cols].isnull().sum()
    for col, count in missing_after.items():
        if count > 0:
            logger.warning(f"Cột {col} vẫn còn {count} missing values")

    logger.info("Hoàn tất tiền xử lý và nội suy chuỗi thời gian")
    return df

"""
Module tải và đọc dữ liệu thô từ các file Parquet.
"""
import os
import pandas as pd
from src.config import TARGET


def load_raw_data(aq_path='data/raw/open_meteo_aq.parquet', weather_path='data/raw/open_meteo_weather.parquet'):
    """
    Tải dữ liệu hourly AQ và Weather từ các Parquet file và ghép chúng lại theo index datetime.
    
    Parameters:
        aq_path (str): Đường dẫn tới file AQ Parquet.
        weather_path (str): Đường dẫn tới file Weather Parquet.

    Returns:
        pd.DataFrame: DataFrame ghép nối, đã sort theo thời gian, cột datetime đổi thành 'date'.
    """
    if not os.path.exists(aq_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu AQ thô tại: {aq_path}")

    df_aq = pd.read_parquet(aq_path)

    if os.path.exists(weather_path):
        df_weather = pd.read_parquet(weather_path)
        # Ghép nối theo index datetime
        df_raw = df_aq.join(df_weather, how='outer')
        print(f"📥 Đã tải và ghép nối dữ liệu AQ ({len(df_aq)} dòng) và Weather ({len(df_weather)} dòng).")
    else:
        print(f"⚠️ Không tìm thấy file dữ liệu Weather tại {weather_path}, chỉ sử dụng AQ.")
        df_raw = df_aq

    # Đưa index datetime thành cột thường và đổi tên thành 'date' để tương thích với các module sau
    df_raw = df_raw.reset_index()
    df_raw = df_raw.rename(columns={'datetime': 'date'})
    df_raw = df_raw.sort_values('date').reset_index(drop=True)

    return df_raw


def describe_data(df_raw):
    """
    In thống kê mô tả cơ bản về dataset hourly.

    Parameters:
        df_raw (pd.DataFrame): DataFrame thô.
    """
    print(f'📊 Dataset: {len(df_raw):,} bản ghi (giờ)')
    if 'date' in df_raw.columns and len(df_raw) > 0:
        print(f'📅 Thời gian: {df_raw["date"].min()} → {df_raw["date"].max()}')
    print(f'📋 Các cột: {df_raw.columns.tolist()}')
    print(f'\n❗ Missing values:')
    missing = df_raw.isnull().sum()[df_raw.isnull().sum() > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print('  Không có missing values.')

    # Thống kê biến target chính
    if TARGET in df_raw.columns:
        print(f'\n📈 Thống kê mô tả {TARGET} (Target):')
        target_stats = df_raw[TARGET].describe()
        print(target_stats.round(2))
        print(f'⚠️  Số giờ {TARGET} > 35 µg/m³: {(df_raw[TARGET] > 35).sum()} / {len(df_raw)} giờ')
        print(f'⚠️  Số giờ {TARGET} > 75 µg/m³: {(df_raw[TARGET] > 75).sum()} / {len(df_raw)} giờ')

"""
Module crawl dữ liệu từ Open-Meteo Air Quality & Weather APIs.
Hỗ trợ:
  - Tải dữ liệu hourly của AQ và Weather
  - Kiểm tra chống rò rỉ dữ liệu tương lai (Freshness validation)
  - Hợp nhất và ghi đè bổ sung (incremental append) vào file Parquet
  - Ghi crawl_log.json
"""
import os
import json
import argparse
import pandas as pd
from datetime import datetime, date, timedelta
import requests

from src.utils.logger import get_logger

logger = get_logger("FetchOpenMeteo")

HANOI_LAT, HANOI_LON = 21.0245, 105.8412


def validate_dates(start_date: str, end_date: str):
    """Đảm bảo khoảng thời gian crawl nằm trong quá khứ."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    if end_date > yesterday:
        raise ValueError(
            f"End date {end_date} không được vượt quá ngày hôm qua {yesterday} để tránh Data Leakage!"
        )
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} phải nhỏ hơn hoặc bằng end date {end_date}!")


def validate_freshness(df: pd.DataFrame):
    """Xác thực không chứa dữ liệu từ tương lai."""
    if df.index.max() >= pd.Timestamp.now(tz='Asia/Ho_Chi_Minh').tz_localize(None):
        raise ValueError("Dữ liệu chứa thời gian tương lai - Leakage nguy hiểm!")


def fetch_hourly_data(url: str, params: dict) -> pd.DataFrame:
    """Fetch JSON từ API và chuyển thành DataFrame có index là datetime."""
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if 'hourly' not in data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data['hourly'])
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.drop(columns=['time']).set_index('datetime')
    return df


def append_and_save_parquet(df_new: pd.DataFrame, filepath: str):
    """Đọc file parquet cũ (nếu có), ghép nối dữ liệu mới, loại bỏ trùng và lưu lại."""
    if os.path.exists(filepath):
        try:
            df_old = pd.read_parquet(filepath)
            df_combined = pd.concat([df_old, df_new])
            # Giữ lại bản ghi cuối nếu trùng index (dữ liệu re-analysis chuẩn xác nhất)
            df_combined = df_combined.loc[~df_combined.index.duplicated(keep='last')]
            df_combined = df_combined.sort_index()
        except Exception as e:
            logger.warning(f"Lỗi khi đọc file cũ {filepath}: {e}. Ghi đè file mới.")
            df_combined = df_new.sort_index()
    else:
        df_combined = df_new.sort_index()
        
    df_combined.to_parquet(filepath, engine='pyarrow')
    return df_combined


def main(start_date=None, end_date=None):
    # Thiết lập thời gian mặc định: 30 ngày qua
    if not start_date or not end_date:
        today = date.today()
        end_date = (today - timedelta(days=1)).isoformat()
        start_date = (today - timedelta(days=30)).isoformat()
        
    validate_dates(start_date, end_date)
    logger.info(f"Bắt đầu crawl dữ liệu từ {start_date} đến {end_date}...")

    # 1. Air Quality Ingestion
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": HANOI_LAT,
        "longitude": HANOI_LON,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Ho_Chi_Minh"
    }
    
    df_aq_new = fetch_hourly_data(aq_url, aq_params)
    if df_aq_new.empty:
        logger.warning("Ingestion AQ thất bại: Không có dữ liệu hourly.")
        return
        
    # Standardize names
    df_aq_new = df_aq_new.rename(columns={
        'carbon_monoxide': 'co',
        'nitrogen_dioxide': 'no2',
        'sulphur_dioxide': 'so2',
        'aerosol_optical_depth': 'aod',
        'uv_index': 'uv'
    })
    
    validate_freshness(df_aq_new)
    aq_filepath = 'data/raw/open_meteo_aq.parquet'
    df_aq = append_and_save_parquet(df_aq_new, aq_filepath)
    logger.info(f"Đã lưu/cập nhật dữ liệu AQ thô tại {aq_filepath} (Tổng cộng: {len(df_aq)} dòng)")

    # 2. Weather Ingestion
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": HANOI_LAT,
        "longitude": HANOI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Ho_Chi_Minh"
    }
    
    df_weather_new = fetch_hourly_data(weather_url, weather_params)
    if df_weather_new.empty:
        logger.warning("Ingestion Weather thất bại: Không có dữ liệu hourly.")
        return
        
    validate_freshness(df_weather_new)
    weather_filepath = 'data/raw/open_meteo_weather.parquet'
    df_weather = append_and_save_parquet(df_weather_new, weather_filepath)
    logger.info(f"Đã lưu/cập nhật dữ liệu Weather thô tại {weather_filepath} (Tổng cộng: {len(df_weather)} dòng)")

    # 3. Ghi crawl log
    log_path = 'data/raw/crawl_log.json'
    log_data = {
        "timestamp_crawled": datetime.now().isoformat(),
        "source": "Open-Meteo Historical Air Quality & Weather API",
        "aq_endpoint": aq_url,
        "weather_endpoint": weather_url,
        "latitude": HANOI_LAT,
        "longitude": HANOI_LON,
        "date_range": f"{df_aq.index.min().date()} to {df_aq.index.max().date()}",
        "aq_records": len(df_aq),
        "weather_records": len(df_weather)
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã cập nhật crawl log tại: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical data from Open-Meteo")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()
    main(args.start, args.end)

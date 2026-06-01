import os
import json
import requests
import pandas as pd
from datetime import datetime, date, timedelta

def setup_directories():
    """Tạo cấu trúc thư mục dự án."""
    dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'models',
        'reports/figures'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"📁 Created directory: {d}")

def fetch_and_save_parquet():
    """Tải dữ liệu hourly từ Open-Meteo (2023-01-01 đến 2024-12-31) và lưu dạng Parquet."""
    lat, lon = 21.0245, 105.8412
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    timezone = "Asia/Ho_Chi_Minh"

    # 1. Fetch Air Quality (CAMS Reanalysis)
    print("🔄 Downloading historical hourly Air Quality data...")
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone
    }
    
    aq_resp = requests.get(aq_url, params=aq_params)
    aq_resp.raise_for_status()
    aq_json = aq_resp.json()
    
    # Parse to DataFrame
    aq_data = aq_json['hourly']
    df_aq = pd.DataFrame(aq_data)
    df_aq['datetime'] = pd.to_datetime(df_aq['time'])
    df_aq = df_aq.drop(columns=['time']).set_index('datetime')
    
    # Rename columns to standardized names
    df_aq = df_aq.rename(columns={
        'carbon_monoxide': 'co',
        'nitrogen_dioxide': 'no2',
        'sulphur_dioxide': 'so2',
        'aerosol_optical_depth': 'aod',
        'uv_index': 'uv'
    })
    
    # Save to raw
    aq_path = 'data/raw/open_meteo_aq.parquet'
    df_aq.to_parquet(aq_path, engine='pyarrow')
    print(f"✅ Saved Air Quality data ({len(df_aq)} records) to {aq_path}")

    # 2. Fetch Weather (Historical Archive)
    print("🔄 Downloading historical hourly Weather data...")
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone
    }
    
    weather_resp = requests.get(weather_url, params=weather_params)
    weather_resp.raise_for_status()
    weather_json = weather_resp.json()
    
    # Parse to DataFrame
    weather_data = weather_json['hourly']
    df_weather = pd.DataFrame(weather_data)
    df_weather['datetime'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.drop(columns=['time']).set_index('datetime')
    
    # Save to raw
    weather_path = 'data/raw/open_meteo_weather.parquet'
    df_weather.to_parquet(weather_path, engine='pyarrow')
    print(f"✅ Saved Weather data ({len(df_weather)} records) to {weather_path}")

    # 3. Create crawl_log.json
    log_path = 'data/raw/crawl_log.json'
    log_data = {
        "timestamp_crawled": datetime.now().isoformat(),
        "source": "Open-Meteo Historical Air Quality & Weather API",
        "aq_endpoint": aq_url,
        "weather_endpoint": weather_url,
        "latitude": lat,
        "longitude": lon,
        "date_range": f"{start_date} to {end_date}",
        "aq_records": len(df_aq),
        "weather_records": len(df_weather)
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Created crawl log: {log_path}")

if __name__ == '__main__':
    setup_directories()
    fetch_and_save_parquet()

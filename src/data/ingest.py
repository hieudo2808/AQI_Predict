import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from src.config import LATITUDE, LONGITUDE, TIMEZONE

# Load environment variables
load_dotenv()

def get_last_data_time(metadata_path='metadata/data_freshness.json'):
    if not os.path.exists(metadata_path):
        return pd.Timestamp("2024-01-01T00:00:00")
    with open(metadata_path, 'r') as f:
        data = json.load(f)
        return pd.Timestamp(data.get("last_data_time", "2024-01-01T00:00:00"))

def fetch_openmeteo_data(start_time, end_time):
    """Fetch weather data from Open-Meteo."""
    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    
    # Forecast API chỉ hỗ trợ quá khứ 92 ngày. Nếu lớn hơn phải dùng Archive API
    if (end_time - start_time).days > 85 or (pd.Timestamp.now(tz=TIMEZONE).tz_localize(None) - start_time).days > 85:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        
    hourly = "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation"
        
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": hourly,
        "timezone": TIMEZONE,
        "start_date": start_date,
        "end_date": end_date
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    hourly_data = data.get("hourly", {})
    df = pd.DataFrame(hourly_data)
    if "time" in df.columns:
        df = df.rename(columns={"time": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        
    # Lọc lại chính xác theo khoảng thời gian yêu cầu và tránh lấy data tương lai
    now_local = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    actual_end_time = min(end_time, now_local)
    
    df = df[(df["datetime"] > start_time) & (df["datetime"] <= actual_end_time)].reset_index(drop=True)
    return df

def fetch_purpleair_data(start_time, end_time):
    """Fetch actual sensor data from PurpleAir API (History)."""
    from src.config import PURPLEAIR_SENSOR_INDEX, TIMEZONE
    api_key = os.environ.get("PURPLEAIR_READ_KEY")
    if not api_key:
        raise ValueError("Thiếu PURPLEAIR_READ_KEY trong file .env")
        
    url = f"https://api.purpleair.com/v1/sensors/{PURPLEAIR_SENSOR_INDEX}/history/csv"
    headers = {"X-API-Key": api_key}
    
    chunk_size = pd.Timedelta(days=30)
    current_start = start_time
    
    all_data = []
    import time
    from io import StringIO
    import logging
    
    while current_start < end_time:
        current_end = min(current_start + chunk_size, end_time)
        
        chunk_start_ts = int(current_start.timestamp())
        chunk_end_ts = int(current_end.timestamp())
        
        params = {
            "start_timestamp": chunk_start_ts,
            "end_timestamp": chunk_end_ts,
            "average": 60, # 60 phút = Hourly
            "fields": "pm2.5_atm,pm10.0_atm"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            if response.status_code == 200:
                csv_data = StringIO(response.text)
                df_chunk = pd.read_csv(csv_data)
                if not df_chunk.empty:
                    all_data.append(df_chunk)
            elif response.status_code == 404:
                logging.warning(f"PurpleAir trả về 404 cho trạm {PURPLEAIR_SENSOR_INDEX}")
                break
            else:
                logging.warning(f"Lỗi tải PurpleAir (Code {response.status_code}): {response.text}")
                
            time.sleep(1.5) # Tránh hit rate limit (PurpleAir is strict)
        except Exception as e:
            logging.error(f"Lỗi kết nối khi tải PurpleAir: {e}")
            
        current_start = current_end
        
    if not all_data:
        return pd.DataFrame()
        
    df_final = pd.concat(all_data, ignore_index=True)
    
    if 'time_stamp' not in df_final.columns:
        return pd.DataFrame()
        
    # Xử lý lại cột thời gian (từ timestamp -> UTC -> local -> tz-naive để join)
    df_final['datetime'] = pd.to_datetime(df_final['time_stamp'], unit='s', utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    df_final = df_final.rename(columns={'pm2.5_atm': 'pm2_5', 'pm10.0_atm': 'pm10'})
    
    # Chỉ giữ các cột quan trọng
    keep_cols = ['datetime', 'pm2_5', 'pm10']
    available_cols = [c for c in keep_cols if c in df_final.columns]
    df_final = df_final[available_cols]
    
    # Sắp xếp và xóa dòng trùng
    df_final = df_final.sort_values(by='datetime').drop_duplicates(subset=['datetime']).reset_index(drop=True)
    
    return df_final

def append_to_datalake(df, category, base_dir='data/raw'):
    if df.empty:
        return
        
    df = df.copy()
    dt_col = 'datetime'
    if dt_col not in df.columns:
        dt_col = df.index.name
        if dt_col:
            df = df.reset_index()
    
    df['year'] = pd.to_datetime(df[dt_col]).dt.year
    df['month'] = pd.to_datetime(df[dt_col]).dt.month
    
    out_dir = os.path.join(base_dir, category)
    os.makedirs(out_dir, exist_ok=True)
    
    df.to_parquet(out_dir, partition_cols=['year', 'month'], engine='pyarrow', index=False)

def update_freshness_metadata(new_time, metadata_path='metadata/data_freshness.json'):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    freshness = {"last_data_time": new_time.isoformat()}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            data["last_data_time"] = new_time.isoformat()
            freshness = data
            
    with open(metadata_path, 'w') as f:
        json.dump(freshness, f, indent=4)

@st.cache_data(ttl=3600)
def fetch_weather_forecast():
    """Fetch weather forecast for the next 72 hours from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    hourly = "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation"
    
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": hourly,
        "timezone": TIMEZONE,
        "forecast_days": 4
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    hourly_data = data.get("hourly", {})
    df = pd.DataFrame(hourly_data)
    if "time" in df.columns:
        df = df.rename(columns={"time": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        
    return df

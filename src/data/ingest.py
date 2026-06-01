import os
import json
import requests
import pandas as pd
from src.config import LATITUDE, LONGITUDE, TIMEZONE

def get_last_data_time(metadata_path='metadata/data_freshness.json'):
    if not os.path.exists(metadata_path):
        return pd.Timestamp("2024-01-01T00:00:00")
    with open(metadata_path, 'r') as f:
        data = json.load(f)
        return pd.Timestamp(data.get("last_data_time", "2024-01-01T00:00:00"))

def fetch_openmeteo_data(start_time, end_time, is_aq=True):
    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    
    if is_aq:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        hourly = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index,us_aqi"
    else:
        # Forecast API chỉ hỗ trợ quá khứ 92 ngày. Nếu lớn hơn phải dùng Archive API
        if (end_time - start_time).days > 85 or (pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None) - start_time).days > 85:
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
        
    # Lọc chỉ lấy những giờ lớn hơn start_time (tránh trùng lặp do API trả về cả ngày)
    df = df[df["datetime"] > start_time].reset_index(drop=True)
    return df

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
    
    # Đối với pyarrow partition, cách đơn giản là đọc partition hiện tại, nối thêm và ghi đè,
    # hoặc dùng tính năng append_to_dataset (nhưng phức tạp).
    # Ghi thẳng bằng partition_cols sẽ tạo file nhỏ thêm vào thư mục.
    # Trong dataset thật, ghi thêm file parquet là chuẩn data lake.
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

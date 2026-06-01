import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from src.config import LATITUDE, LONGITUDE, TIMEZONE, OPENAQ_LOCATION_ID

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

def fetch_openaq_data(start_time, end_time):
    """Fetch actual sensor data from OpenAQ API v3."""
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        raise ValueError("Thiếu OPENAQ_API_KEY trong file .env")
        
    # Lấy thông tin location để biết danh sách sensor ID
    loc_url = f"https://api.openaq.org/v3/locations/{OPENAQ_LOCATION_ID}"
    headers = {"X-API-Key": api_key}
    
    resp = requests.get(loc_url, headers=headers)
    if resp.status_code == 404:
        raise ValueError(f"Không tìm thấy trạm đo (Location ID {OPENAQ_LOCATION_ID}) trên OpenAQ.")
    resp.raise_for_status()
    
    loc_results = resp.json().get("results", [])
    if not loc_results:
        return pd.DataFrame()
    loc_data = loc_results[0]
    
    sensors = loc_data.get("sensors", [])
    if not sensors:
        return pd.DataFrame()
        
    # Chuyển đổi thời gian sang chuỗi ISO-8601 chuẩn UTC
    start_utc = start_time.tz_localize(TIMEZONE).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Lọc không lấy thời gian tương lai
    now_local = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    actual_end_time = min(end_time, now_local)
    end_utc = actual_end_time.tz_localize(TIMEZONE).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    
    all_records = []
    import time
    
    # Duyệt qua từng sensor (pm25, pm10, co, no2, o3, so2) của trạm
    for sensor in sensors:
        sensor_id = sensor["id"]
        param_name = sensor.get("parameter", {}).get("name")
        if not param_name:
            continue
            
        # API endpoint cho hourly data của 1 sensor
        sensor_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
        params = {
            "datetime_from": start_utc,
            "datetime_to": end_utc,
            "limit": 1000,
            "page": 1
        }
        
        retry_count = 0
        while True:
            try:
                s_resp = requests.get(sensor_url, headers=headers, params=params, timeout=15)
                if s_resp.status_code == 404:
                    break # Bỏ qua nếu endpoint này lỗi với sensor đó
                s_resp.raise_for_status()
                s_data = s_resp.json()
                
                results = s_data.get("results", [])
                for row in results:
                    # OpenAQ v3 /hours endpoint trả về datetime local ở trường period
                    dt_local = row.get("period", {}).get("datetimeTo", {}).get("local")
                    val = row.get("value")
                    if dt_local and val is not None:
                        all_records.append({
                            "datetime": dt_local,
                            "parameter": param_name,
                            "value": val
                        })
                        
                time.sleep(0.2)  # Tránh hit rate limit
                retry_count = 0  # Reset retry count sau khi thành công
                
                if len(results) < params["limit"]:
                    break
                params["page"] += 1
                
            except requests.exceptions.RequestException as e:
                import logging
                retry_count += 1
                if retry_count > 3:
                    logging.warning(f"Đã thử lại 3 lần không thành công ở trang {params['page']}. Bỏ qua các dữ liệu cũ hơn của sensor này.")
                    break
                else:
                    logging.warning(f"Lỗi khi gọi API OpenAQ: {e}. Đang thử lại ({retry_count}/3)...")
                    time.sleep(3)
                    continue
            
    df_raw = pd.DataFrame(all_records)
    if df_raw.empty:
        return pd.DataFrame()
        
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"]).dt.tz_localize(None)
    
    # Pivot table để chuyển parameter thành các cột
    df = df_raw.pivot_table(index="datetime", columns="parameter", values="value", aggfunc="mean").reset_index()
    
    # Chuẩn hóa tên cột
    rename_map = {
        "pm25": "pm2_5",
        "pm10": "pm10",
        "co": "carbon_monoxide",
        "no2": "nitrogen_dioxide",
        "so2": "sulphur_dioxide",
        "o3": "ozone"
    }
    df = df.rename(columns=rename_map)
    
    df = df.sort_values("datetime").reset_index(drop=True)
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

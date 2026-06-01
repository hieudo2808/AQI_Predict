"""
Module chuẩn hóa và làm sạch dữ liệu (ETL Week 3).

Thực hiện:
1. Đọc dữ liệu lịch sử (CSV).
2. Đọc dữ liệu thô mới nhất (JSON từ crawler).
3. Chuẩn hóa timezone, resample từ Hourly sang Daily (cho dữ liệu mới).
4. Ghép nối (merge) dữ liệu, loại bỏ trùng lặp.
5. Xử lý missing values & outliers.
6. Lưu tập dữ liệu đã xử lý vào data/processed.
"""
import json
import glob
import pandas as pd
from pathlib import Path
import pytz

from src.config import DATA_URL
from src.etl.preprocess import handle_missing_values, remove_outliers
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_open_meteo_json(json_path):
    """
    Parse file JSON thô từ Open-Meteo thành DataFrame.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'hourly' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['hourly'])

    # Chuẩn hóa thời gian và timezone
    # Open-Meteo trả về local time (nhưng không có múi giờ trong chuỗi)
    df['time'] = pd.to_datetime(df['time'])

    # Gắn múi giờ từ metadata
    # Fallback: 'Asia/Ho_Chi_Minh' là timezone chuẩn cho Hà Nội
    tz = data.get('timezone', 'Asia/Ho_Chi_Minh')

    # Validate timezone string trước khi sử dụng
    # Nếu timezone không hợp lệ, dùng fallback
    import pytz
    try:
        pytz.timezone(tz)  # Validate
        df['time'] = df['time'].dt.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"⚠️ Timezone '{tz}' không hợp lệ, dùng fallback 'Asia/Ho_Chi_Minh'")
        df['time'] = df['time'].dt.tz_localize('Asia/Ho_Chi_Minh', ambiguous='NaT', nonexistent='shift_forward')

    return df

def resample_to_daily(df_hourly):
    """
    Resample dữ liệu từ Hourly sang Daily bằng cách lấy trung bình.
    US AQI thường có thể lấy Max, nhưng đây là CAMS model nên dùng Mean
    để đồng nhất với data lịch sử.
    """
    if df_hourly.empty:
        return df_hourly
        
    df_hourly = df_hourly.set_index('time')
    
    # Lấy giá trị trung bình theo ngày
    df_daily = df_hourly.resample('D').mean().round(2)
    df_daily = df_daily.reset_index()
    
    # Đổi tên cột 'time' thành 'date' cho khớp với lịch sử
    df_daily = df_daily.rename(columns={'time': 'date'})
    
    # Loại bỏ timezone (đưa về naive datetime) vì data lịch sử là naive Date
    # Hoặc giữ timezone và chuẩn hóa data lịch sử. 
    # Do dữ liệu là theo ngày theo múi giờ VN, chuyển sang date format:
    df_daily['date'] = df_daily['date'].dt.tz_localize(None)
    
    return df_daily

def load_historical_data():
    """
    Cố gắng load dữ liệu CSV lịch sử.
    """
    try:
        df_hist = pd.read_csv(DATA_URL, parse_dates=['date'])
        # Đảm bảo datetime
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        return df_hist
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy dữ liệu lịch sử tại {DATA_URL}")
        return pd.DataFrame()

def run_clean_standardize(config_path="configs/config.yaml"):
    """
    Thực thi pipeline chuẩn hóa:
    Raw JSONs + Historical CSV -> Cleaned DataFrame -> Processed CSV
    """
    config = load_config(config_path)
    raw_dir = Path("data/raw/open_meteo")
    processed_dir = Path(config['paths']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔄 BẮT ĐẦU: CHUẨN HÓA VÀ LÀM SẠCH DỮ LIỆU (ETL - Week 3)")
    print("="*60)
    
    # 1. Khai thác dữ liệu thô mới (JSON)
    json_files = glob.glob(str(raw_dir / "*.json"))
    new_dfs = []
    
    for jf in json_files:
        df_h = parse_open_meteo_json(jf)
        df_d = resample_to_daily(df_h)
        new_dfs.append(df_d)
        
    if new_dfs:
        df_new = pd.concat(new_dfs, ignore_index=True)
        print(f"📥 Nạp dữ liệu JSON mới: {len(df_new)} ngày từ {len(json_files)} file")
    else:
        df_new = pd.DataFrame()
        print("⚠️ Không có file JSON mới nào.")

    # 2. Đọc dữ liệu lịch sử
    df_hist = load_historical_data()
    print(f"📥 Nạp dữ liệu lịch sử: {len(df_hist)} ngày")
    
    # 3. Kết hợp dữ liệu (Merge)
    if not df_new.empty and not df_hist.empty:
        df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    elif not df_new.empty:
        df_combined = df_new
    else:
        df_combined = df_hist
        
    if df_combined.empty:
        print("❌ Không có bất kỳ dữ liệu nào để xử lý.")
        return None
        
    # Sắp xếp thời gian và loại bỏ trùng lặp
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    # Nếu trùng ngày, lấy dữ liệu mới nhất (dữ liệu CAMS có thể được re-analysis)
    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
    
    print(f"🔗 Dữ liệu sau khi merge & deduplicate: {len(df_combined)} ngày (từ {df_combined['date'].min().date()} đến {df_combined['date'].max().date()})")
    
    # 4. Tiền xử lý: Missing + Outliers
    print("\n🛠️ Xử lý Missing values...")
    df_combined = handle_missing_values(df_combined)
    
    print("\n🛠️ Xử lý Outliers (trên US AQI)...")
    df_combined = remove_outliers(df_combined)
    
    # 5. Lưu kết quả processed
    out_file = processed_dir / "dataset.csv"
    df_combined.to_csv(out_file, index=False)
    print("\n" + "="*60)
    print(f"✅ HOÀN_TẤT: Đã ghi dữ liệu chuẩn hóa ra {out_file}")
    print("="*60)
    
    return df_combined

if __name__ == "__main__":
    run_clean_standardize()

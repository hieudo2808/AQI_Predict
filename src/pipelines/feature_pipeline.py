import os
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def should_rebuild_features(feature_metadata_path='metadata/feature_metadata.json', raw_max_time=None):
    if not os.path.exists(feature_metadata_path):
        return True
    
    with open(feature_metadata_path, 'r') as f:
        data = json.load(f)
        
    feature_max_time = pd.Timestamp(data.get("feature_data_max_time", "2000-01-01")).tz_localize(None)
    if raw_max_time is None:
        raw_max_time = feature_max_time # fallback, but shouldn't happen usually
    
    raw_ts = pd.Timestamp(raw_max_time).tz_localize(None)
    
    if raw_ts > feature_max_time:
        return True
    return False

def update_feature_metadata(feature_metadata_path='metadata/feature_metadata.json', raw_max_time=None):
    os.makedirs(os.path.dirname(feature_metadata_path), exist_ok=True)
    
    data = {
        "generated_at": datetime.now().isoformat(),
        "raw_data_max_time": raw_max_time,
        "feature_data_max_time": raw_max_time,
        "feature_version": "v2" # Bumping version just in case
    }
    
    with open(feature_metadata_path, 'w') as f:
        json.dump(data, f, indent=4)

def run_feature_pipeline():
    logger.info("Khởi chạy Feature Pipeline...")
    
    # 1. Đọc data_freshness.json để biết raw_data_max_time
    freshness_path = 'metadata/data_freshness.json'
    if not os.path.exists(freshness_path):
        logger.warning("Không tìm thấy metadata/data_freshness.json. Hủy chạy feature_pipeline.")
        return
        
    with open(freshness_path, 'r') as f:
        freshness_data = json.load(f)
        raw_max_time = freshness_data.get("last_data_time")
        
    if not raw_max_time:
        logger.warning("Không tìm thấy last_data_time. Hủy chạy.")
        return
        
    # 2. Kiểm tra xem có cần rebuild không
    if not should_rebuild_features('metadata/feature_metadata.json', raw_max_time):
        logger.info(f"Dữ liệu Feature đã được đồng bộ với Raw data (max_time={raw_max_time}). Bỏ qua rebuild.")
        return
        
    logger.info(f"Phát hiện dữ liệu Raw mới ({raw_max_time}). Đang tiến hành rebuild features...")
    
    # 3. Nạp dữ liệu từ Data Lake (PyArrow hỗ trợ đọc thẳng thư mục có chứa file/thư mục partition)
    # Lưu ý: Trong kiến trúc thật, nên đọc bằng pd.read_parquet("data/raw/aq")
    try:
        df_aq = pd.read_parquet('data/raw/aq')
        df_weather = pd.read_parquet('data/raw/weather')
    except Exception as e:
        logger.error(f"Lỗi khi đọc Raw Data Lake: {e}")
        return
        
    # Chuẩn hóa cột datetime nếu cần và merge
    if "datetime" not in df_aq.columns and df_aq.index.name != "datetime":
        logger.error("df_aq không có cột datetime.")
        return
        
    df_aq = df_aq.reset_index(drop=False) if 'datetime' not in df_aq.columns else df_aq
    df_weather = df_weather.reset_index(drop=False) if 'datetime' not in df_weather.columns else df_weather
    
    # Đảm bảo datetime cùng type
    df_aq['datetime'] = pd.to_datetime(df_aq['datetime'])
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    
    # Drop duplicates do API có thể trả về các dòng trùng giờ
    df_aq = df_aq.drop_duplicates(subset=['datetime'])
    df_weather = df_weather.drop_duplicates(subset=['datetime'])
    
    # Merge outer để giữ toàn bộ timeline
    df_merged = pd.merge(df_aq, df_weather, on='datetime', how='outer')
    
    # Bỏ các cột partition year/month dư thừa sau khi merge
    cols_to_drop = [c for c in df_merged.columns if 'year_' in c or 'month_' in c or c == 'year' or c == 'month']
    if cols_to_drop:
        df_merged = df_merged.drop(columns=cols_to_drop)
        
    df_merged = df_merged.set_index('datetime').sort_index()
    
    # 4. Preprocess
    from src.etl.preprocess import preprocess
    logger.info("Bắt đầu Preprocessing...")
    df_clean = preprocess(df_merged)
    
    # 5. Build features
    from src.etl.build_features import build_all_features
    logger.info("Bắt đầu Build Features...")
    df_features = build_all_features(df_clean)
    
    # 6. Save to features_targets.parquet
    os.makedirs('data/features', exist_ok=True)
    out_path = 'data/features/features_targets.parquet'
    
    # Do file features_targets cũ không được partition, ta lưu nguyên file đơn
    # vì các bước train model phụ thuộc vào file này.
    df_features.to_parquet(out_path)
    logger.info(f"Đã lưu file Feature ({len(df_features)} dòng) tại {out_path}.")
    
    # 7. Update metadata
    update_feature_metadata('metadata/feature_metadata.json', raw_max_time)
    logger.info("Đã cập nhật feature_metadata.json.")
    logger.info("Feature Pipeline hoàn tất.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_feature_pipeline()

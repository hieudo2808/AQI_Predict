import os
import json
import logging
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

class StaleDataError(Exception):
    pass

def check_data_freshness(metadata_path='metadata/feature_metadata.json', max_age_hours=6):
    """
    Kiểm tra xem dữ liệu features có quá cũ không.
    Ném lỗi StaleDataError nếu cũ hơn max_age_hours.
    """
    if not os.path.exists(metadata_path):
        raise StaleDataError("Không tìm thấy metadata. Vui lòng chạy Feature Pipeline trước.")
        
    with open(metadata_path, 'r') as f:
        data = json.load(f)
        
    feature_max_time_str = data.get("feature_data_max_time")
    if not feature_max_time_str:
        raise StaleDataError("Không có mốc thời gian trong metadata.")
        
    feature_max_time = pd.Timestamp(feature_max_time_str).tz_localize(None)
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    
    age = now - feature_max_time
    if age > timedelta(hours=max_age_hours):
        raise StaleDataError(f"Dữ liệu đã cũ hơn {max_age_hours} tiếng (Mới nhất: {feature_max_time}, Hiện tại: {now}). Vui lòng cập nhật.")
        
    return True

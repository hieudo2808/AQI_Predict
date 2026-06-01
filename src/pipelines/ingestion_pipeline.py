import logging
import pandas as pd
from src.data.ingest import get_last_data_time, fetch_openmeteo_data, append_to_datalake, update_freshness_metadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ingestion():
    logger.info("Bắt đầu Ingestion Pipeline...")
    
    last_time = get_last_data_time()
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    
    logger.info(f"Thời điểm dữ liệu gần nhất (last_data_time): {last_time}")
    logger.info(f"Thời điểm hiện tại (now): {now}")
    
    if last_time >= now:
        logger.info("Dữ liệu đã được cập nhật mới nhất. Bỏ qua Ingestion.")
        return

    logger.info(f"Đang tải Air Quality data từ {last_time} đến {now}...")
    df_aq = fetch_openmeteo_data(last_time, now, is_aq=True)
    
    logger.info(f"Đang tải Weather data từ {last_time} đến {now}...")
    df_weather = fetch_openmeteo_data(last_time, now, is_aq=False)
    
    if not df_aq.empty:
        append_to_datalake(df_aq, 'aq')
        logger.info(f"Đã lưu {len(df_aq)} dòng vào Data Lake (AQ).")
        
    if not df_weather.empty:
        append_to_datalake(df_weather, 'weather')
        logger.info(f"Đã lưu {len(df_weather)} dòng vào Data Lake (Weather).")

    if not df_aq.empty or not df_weather.empty:
        # Lấy mốc thời gian lớn nhất
        max_t = None
        if not df_aq.empty:
            max_t = df_aq['datetime'].max()
        if not df_weather.empty:
            w_max = df_weather['datetime'].max()
            if max_t is None or w_max > max_t:
                max_t = w_max
        
        if max_t:
            update_freshness_metadata(pd.Timestamp(max_t))
            logger.info(f"Đã cập nhật data_freshness.json với mốc thời gian: {max_t}")
    else:
        logger.warning("Không có dữ liệu mới nào được tải về.")

    logger.info("Ingestion Pipeline hoàn tất.")

if __name__ == '__main__':
    run_ingestion()

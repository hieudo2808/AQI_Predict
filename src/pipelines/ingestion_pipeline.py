import pandas as pd
from src.data.ingest import (
    get_last_data_time, 
    fetch_openmeteo_data, 
    fetch_openaq_data,
    append_to_datalake, 
    update_freshness_metadata
)
from src.utils.logger import get_logger

logger = get_logger("IngestionPipeline")

def run_ingestion():
    logger.info("Bắt đầu Ingestion Pipeline...")
    
    last_time = get_last_data_time()
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    
    logger.info(f"Thời điểm dữ liệu gần nhất (last_data_time): {last_time}")
    logger.info(f"Thời điểm hiện tại (now): {now}")
    
    if last_time >= now:
        logger.info("Dữ liệu đã được cập nhật mới nhất. Bỏ qua Ingestion.")
        return

    aq_success = True
    # Tải dữ liệu AQ từ OpenAQ
    logger.info(f"Đang tải Air Quality data từ OpenAQ ({last_time} đến {now})...")
    try:
        df_aq = fetch_openaq_data(last_time, now)
        if not df_aq.empty:
            append_to_datalake(df_aq, 'aq')
            logger.info(f"Đã lưu {len(df_aq)} dòng vào Data Lake (AQ).")
        else:
            logger.warning("API OpenAQ không trả về dòng dữ liệu nào mới.")
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu OpenAQ: {e}")
        df_aq = pd.DataFrame()
        aq_success = False
        
    weather_success = True
    # Tải dữ liệu Weather từ Open-Meteo
    logger.info(f"Đang tải Weather data từ Open-Meteo ({last_time} đến {now})...")
    try:
        df_weather = fetch_openmeteo_data(last_time, now)
        if not df_weather.empty:
            append_to_datalake(df_weather, 'weather')
            logger.info(f"Đã lưu {len(df_weather)} dòng vào Data Lake (Weather).")
        else:
            logger.warning("API Open-Meteo không trả về dòng dữ liệu nào mới.")
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu Weather: {e}")
        df_weather = pd.DataFrame()
        weather_success = False

    # Chỉ cập nhật time khi cả 2 nguồn không bị lỗi (để tránh skip dữ liệu nguồn bị lỗi)
    if aq_success and weather_success:
        if not df_aq.empty or not df_weather.empty:
            max_t = None
            if not df_aq.empty and not df_weather.empty:
                max_t = min(df_aq['datetime'].max(), df_weather['datetime'].max())
            elif not df_aq.empty:
                max_t = df_aq['datetime'].max()
            elif not df_weather.empty:
                max_t = df_weather['datetime'].max()
            
            if max_t:
                update_freshness_metadata(pd.Timestamp(max_t))
                logger.info(f"Đã cập nhật data_freshness.json với mốc thời gian: {max_t}")
        else:
            logger.warning("Không có dữ liệu mới nào được tải về.")
    else:
        logger.warning("Một trong các API bị lỗi, data_freshness.json sẽ không được cập nhật để tránh mất dữ liệu ở lần chạy sau.")

    logger.info("Ingestion Pipeline hoàn tất.")

if __name__ == '__main__':
    run_ingestion()

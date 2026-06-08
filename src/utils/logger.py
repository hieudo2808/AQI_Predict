import logging
import os
from datetime import datetime

# Define standard log directory
LOG_DIR = "logs"

def get_logger(name: str, log_file: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Tạo và cấu hình một Centralized Logger chuẩn cho dự án.
    Không lạm dụng màu mè, tập trung vào cấu trúc rõ ràng: [LEVEL] - Module - Message
    """
    logger = logging.getLogger(name)
    
    # Tránh tạo handler trùng lặp nếu logger đã được gọi nhiều lần
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Định dạng Log rõ ràng, tối giản
    formatter = logging.Formatter('[%(levelname)s] - %(name)s - %(message)s')
    
    # 1. Console Handler (In ra màn hình)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 2. File Handler (Lưu ra file nếu cần)
    if log_file is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        # Tự động sinh tên file theo ngày nếu không được cung cấp
        today_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(LOG_DIR, f"pipeline_run_{today_str}.log")
        
    # Tạo thư mục chứa file log nếu chưa có
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

"""
Module đánh giá hiệu năng mô hình dự báo PM2.5 theo 5 kịch bản backtesting K1-K5
và trực quan hóa confusion matrix 6x6.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix
from src.config import TARGET, FIGURES_DIR
from src.aqi_formula import pm25_to_aqi, aqi_to_level
from src.utils.logger import get_logger

logger = get_logger("Evaluate")

# Danh sách 6 mức AQI chuẩn của EPA
AQI_CLASSES = [
    'Good',
    'Moderate',
    'Unhealthy for Sensitive',
    'Unhealthy',
    'Very Unhealthy',
    'Hazardous'
]


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Tính MAE và RMSE cho nồng độ PM2.5."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def evaluate_classification_metrics(y_true_pm25: np.ndarray, y_pred_pm25: np.ndarray) -> tuple[float, np.ndarray]:
    """Quy đổi PM2.5 sang AQI level và tính Macro F1 cùng Confusion Matrix 6x6."""
    # Chuyển đổi nồng độ sang nhãn AQI level
    y_true_levels = [aqi_to_level(pm25_to_aqi(c))[0] for c in y_true_pm25]
    y_pred_levels = [aqi_to_level(pm25_to_aqi(c))[0] for c in y_pred_pm25]

    # Tính Macro F1 score trên 6 nhãn tiêu chuẩn
    f1 = f1_score(y_true_levels, y_pred_levels, labels=AQI_CLASSES, average='macro', zero_division=0)
    
    # Tính Confusion Matrix
    cm = confusion_matrix(y_true_levels, y_pred_levels, labels=AQI_CLASSES)
    
    return f1, cm


def plot_confusion_matrix(cm: np.ndarray, horizon_label: str, model_name: str, save_dir: str = FIGURES_DIR) -> None:
    """Vẽ và lưu Confusion Matrix 6x6 dạng heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=AQI_CLASSES, yticklabels=AQI_CLASSES,
                cmap='Blues', ax=ax, cbar=True)
    
    ax.set_title(f'Confusion Matrix AQI Level ({model_name}) — {horizon_label}', fontsize=12, pad=15)
    ax.set_xlabel('Predicted Level')
    ax.set_ylabel('Actual Level')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cm_{model_name.lower().replace(" ", "_")}_{horizon_label.lower()}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Đã lưu confusion matrix: {save_path}")


def run_5_backtesting_scenarios(test_df: pd.DataFrame,
                               y_pred_dict: dict[int, np.ndarray],
                               model_name: str,
                               save_dir: str = FIGURES_DIR) -> dict:
    """
    Thực thi 5 kịch bản backtesting K1-K5.
    
    Parameters:
        test_df (pd.DataFrame): DataFrame kiểm thử chứa cột 'date' và các target_pm2_5_t{h}.
        y_pred_dict (dict): Dự đoán dạng {1: pred_t1, 24: pred_t24, 72: pred_t72}.
        model_name (str): Tên mô hình (ví dụ: 'XGBoost', 'Baseline').
        save_dir (str): Thư mục lưu hình ảnh.
        
    Returns:
        dict: Dict chứa kết quả chi tiết của 5 kịch bản.
    """
    logger.info(f"Bắt đầu đánh giá 5 kịch bản backtesting cho {model_name}...")
    results = {}
    
    # Đảm bảo cột date ở định dạng datetime
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])

    # --- K1, K2, K3: Horizons t+1h, t+24h, t+72h ---
    for h in [1, 24, 72]:
        target_col = f'target_pm2_5_t{h}'
        if target_col not in test_df.columns or h not in y_pred_dict:
            logger.warning(f"Thiếu target hoặc dự đoán cho horizon t+{h}h. Bỏ qua kịch bản.")
            continue
            
        # Lọc các dòng không bị khuyết target
        valid_mask = test_df[target_col].notna()
        y_true_h = test_df.loc[valid_mask, target_col].values
        y_pred_h = y_pred_dict[h][valid_mask]
        
        # Xóa các dòng NaN từ dự đoán (do quá trình shift tạo ra ở cuối chuỗi)
        mask = ~np.isnan(y_true_h) & ~np.isnan(y_pred_h)
        y_true_h = y_true_h[mask]
        y_pred_h = y_pred_h[mask]
        
        mae, rmse = evaluate_regression_metrics(y_true_h, y_pred_h)
        f1, cm = evaluate_classification_metrics(y_true_h, y_pred_h)
        
        # Vẽ confusion matrix
        plot_confusion_matrix(cm, f't{h}h', model_name, save_dir)
        
        results[f'K{1 if h==1 else (2 if h==24 else 3)}'] = {
            'horizon': h,
            'mae': mae,
            'rmse': rmse,
            'f1_macro': f1,
            'cm': cm
        }
        logger.info(f"Kịch bản K{1 if h==1 else (2 if h==24 else 3)} (t+{h}h) | MAE={mae:.2f} | RMSE={rmse:.2f} | F1={f1:.3f}")

    # --- K4: Episode ô nhiễm cao (Top 5% thực tế tại t+24h) ---
    target_k4 = f'target_pm2_5_t24'
    if target_k4 in test_df.columns and 24 in y_pred_dict:
        valid_mask = test_df[target_k4].notna()
        test_valid = test_df[valid_mask].copy()
        y_true_all = test_valid[target_k4].values
        y_pred_all = y_pred_dict[24][valid_mask]
        
        # Ngưỡng phân vị 95 (top 5% episode cao)
        threshold_95 = np.percentile(y_true_all, 95)
        # Xóa các dòng NaN ở y_true_h hoặc y_pred_h (chỉ ảnh hưởng phần đuôi chuỗi hoặc dòng thiếu dữ liệu)
        mask = ~np.isnan(y_true_all) & ~np.isnan(y_pred_all)
        y_true_h = y_true_all[mask]
        y_pred_h = y_pred_all[mask]
        
        k4_mask = y_true_h >= threshold_95
        
        y_true_k4 = y_true_h[k4_mask]
        y_pred_k4 = y_pred_h[k4_mask]
        
        mae_k4, rmse_k4 = evaluate_regression_metrics(y_true_k4, y_pred_k4)
        
        # Tính Recall cho các mức ô nhiễm nguy hại (Sensitive trở lên)
        y_true_levels = [aqi_to_level(pm25_to_aqi(c))[0] for c in y_true_k4]
        y_pred_levels = [aqi_to_level(pm25_to_aqi(c))[0] for c in y_pred_k4]
        
        unhealthy_levels = {'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy', 'Hazardous'}
        true_unhealthy_count = sum(1 for lvl in y_true_levels if lvl in unhealthy_levels)
        pred_unhealthy_count = sum(1 for true_lvl, pred_lvl in zip(y_true_levels, y_pred_levels) 
                                   if true_lvl in unhealthy_levels and pred_lvl in unhealthy_levels)
                                   
        recall_k4 = pred_unhealthy_count / true_unhealthy_count if true_unhealthy_count > 0 else 0.0
        
        results['K4'] = {
            'threshold_95': threshold_95,
            'mae': mae_k4,
            'rmse': rmse_k4,
            'recall_unhealthy': recall_k4
        }
        logger.info(f"Kịch bản K4 (Top 5% Episode >= {threshold_95:.2f} µg/m³) | MAE={mae_k4:.2f} | Recall Unhealthy={recall_k4:.3f}")

    # --- K5: Mùa Đông vs Mùa Hè (Horizon t+24h) ---
    target_k5 = f'target_pm2_5_t24'
    if target_k5 in test_df.columns and 24 in y_pred_dict:
        valid_mask = test_df[target_k5].notna()
        test_valid = test_df[valid_mask].copy()
        test_valid['pred_pm2_5'] = y_pred_dict[24][valid_mask]
        
        # Bỏ qua các dòng có dự đoán NaN
        test_valid = test_valid.dropna(subset=['pred_pm2_5'])
        
        # Lọc theo tháng
        test_valid['month'] = test_valid['date'].dt.month
        
        df_winter = test_valid[test_valid['month'].isin([12, 1, 2])] # DJF
        df_summer = test_valid[test_valid['month'].isin([6, 7, 8])]  # JJA
        
        mae_winter = mean_absolute_error(df_winter[target_k5], df_winter['pred_pm2_5']) if len(df_winter) > 0 else np.nan
        mae_summer = mean_absolute_error(df_summer[target_k5], df_summer['pred_pm2_5']) if len(df_summer) > 0 else np.nan
        
        results['K5'] = {
            'mae_winter': mae_winter,
            'mae_summer': mae_summer,
            'count_winter': len(df_winter),
            'count_summer': len(df_summer)
        }
        logger.info(f"Kịch bản K5 | MAE Mùa Đông (DJF, N={len(df_winter)})={mae_winter:.2f} | MAE Mùa Hè (JJA, N={len(df_summer)})={mae_summer:.2f}")

    return results

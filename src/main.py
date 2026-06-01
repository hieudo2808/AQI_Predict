"""
╔══════════════════════════════════════════════════════════════╗
║  US AQI Hà Nội — Pipeline Runner (Hourly, Multi-Horizon)   ║
║  Chạy: python -m src.main                                  ║
╚══════════════════════════════════════════════════════════════╝

Thứ tự pipeline:
  Bước 1+2: Ingest + Preprocess (ETL)
  Bước 3:   Feature Engineering + Multi-Horizon Targets
  Bước 4:   Huấn luyện & Lưu mô hình (XGBoost + SARIMA × 3 horizons)
  Bước 5:   Backtesting 5 kịch bản (K1–K5) + Confusion Matrix
  Bước 6:   EDA Plots (11 biểu đồ bắt buộc)
  Bước 7:   Explainability (SHAP Summary + Waterfall)
"""
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    DEFAULT_HORIZON, FEATURES, FIGURES_DIR,
    HORIZONS, TRAIN_RATIO, VALID_RATIO,
)
from src.etl.build_features import build_all_features
from src.etl.clean_standardize import run_clean_standardize
from src.modeling.train import (
    prepare_data,
    train_and_save_models,
)
from src.modeling.evaluate import run_5_backtesting_scenarios
from src.visualization.eda import generate_all_eda_plots
from src.explainability.explain import run_explainability

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def run_pipeline(
    horizons: list | None = None,
    save_models: bool = True,
    models_dir: str = 'models',
) -> dict | None:
    """
    Chạy toàn bộ pipeline từ Ingest đến Explainability.

    Parameters:
        horizons (list, optional): Danh sách horizons. Mặc định HORIZONS từ config.
        save_models (bool): Có lưu mô hình vào models_dir không.
        models_dir (str): Thư mục lưu mô hình.

    Returns:
        dict: Kết quả pipeline gồm df, backtest_results, và paths.
    """
    start_time = time.time()
    h_list = horizons or HORIZONS
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ════════════════════════════════════════════════════
    # BƯỚC 1+2: INGEST + PREPROCESS (ETL)
    # ════════════════════════════════════════════════════
    logger.info('=' * 60)
    logger.info('BƯỚC 1+2: Nạp dữ liệu thô và Tiền xử lý')
    logger.info('=' * 60)

    features_path = 'data/features/features_targets.parquet'
    if os.path.exists(features_path):
        logger.info('Tim thay features_targets.parquet — bo qua lai ETL.')
        df = pd.read_parquet(features_path)
    else:
        logger.info('Khong tim thay features cache, chay ETL day du...')
        df = run_clean_standardize()
        if df is None or df.empty:
            logger.error('ETL that bai. Dung pipeline.')
            return None

        # ════════════════════════════════════════════════════
        # BƯỚC 3: FEATURE ENGINEERING + MULTI-HORIZON TARGETS
        # ════════════════════════════════════════════════════
        logger.info('=' * 60)
        logger.info('BUOC 3: Feature Engineering + Multi-Horizon Targets')
        logger.info('=' * 60)
        df = build_all_features(df, horizons=h_list)
        df.to_parquet(features_path)
        logger.info('Da luu features_targets.parquet.')

    logger.info('Dataset: %d dòng × %d cột', len(df), len(df.columns))

    # ════════════════════════════════════════════════════
    # BƯỚC 4: HUẤN LUYỆN & LƯU MÔ HÌNH
    # ════════════════════════════════════════════════════
    logger.info('=' * 60)
    logger.info('BƯỚC 4: Huấn luyện & Lưu mô hình (%s)', h_list)
    logger.info('=' * 60)

    if save_models:
        train_and_save_models(df, features=FEATURES, horizons=h_list, save_dir=models_dir)
    else:
        logger.info('save_models=False — bỏ qua huấn luyện lại.')

    # ════════════════════════════════════════════════════
    # BƯỚC 5: BACKTESTING 5 KỊCH BẢN
    # ════════════════════════════════════════════════════
    logger.info('=' * 60)
    logger.info('BƯỚC 5: Backtesting 5 kịch bản (K1–K5)')
    logger.info('=' * 60)

    # Xây dựng test_df và predictions từ tập test period
    n = len(df)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))
    test_df = df.iloc[valid_end:].copy()
    if isinstance(test_df.index, pd.DatetimeIndex):
        idx_name = test_df.index.name or 'datetime'
        test_df = test_df.reset_index().rename(columns={idx_name: 'date'})

    y_pred_np: dict[int, np.ndarray] = {}
    for h in h_list:
        xgb_path = os.path.join(models_dir, f'xgb_t{h}.json')
        if not os.path.exists(xgb_path):
            logger.warning('Khong tim thay %s — bo qua horizon t+%d.', xgb_path, h)
            continue
        data_h = prepare_data(df, horizon=h)
        model_h = xgb.XGBRegressor()
        model_h.load_model(xgb_path)
        preds = model_h.predict(data_h['X_test'])
        # Align predictions sang index cua test_df (co the co do lech NaN cuoi)
        pred_series = pd.Series(preds, index=data_h['X_test'].index)
        y_pred_np[h] = pred_series.reindex(test_df.index).values

    backtest_results: dict = {}
    if y_pred_np:
        backtest_results = run_5_backtesting_scenarios(
            test_df=test_df,
            y_pred_dict=y_pred_np,
            model_name='XGBoost',
            save_dir=FIGURES_DIR,
        )
        _print_backtest_summary(backtest_results)

    # ════════════════════════════════════════════════════
    # BƯỚC 6: EDA PLOTS (11 biểu đồ bắt buộc)
    # ════════════════════════════════════════════════════
    logger.info('=' * 60)
    logger.info('BƯỚC 6: EDA Plots (11 biểu đồ)')
    logger.info('=' * 60)

    # EDA cần DataFrame có cột 'pm2_5'; eda.py dùng dữ liệu giờ
    df_eda = df.copy()
    if isinstance(df_eda.index, pd.DatetimeIndex):
        df_eda = df_eda.reset_index().rename(
            columns={df_eda.index.name or 'datetime': 'date'}
        )
    generate_all_eda_plots(df_eda, save_dir=FIGURES_DIR)

    # ════════════════════════════════════════════════════
    # BƯỚC 7: EXPLAINABILITY (SHAP Summary + Waterfall)
    # ════════════════════════════════════════════════════
    logger.info('=' * 60)
    logger.info('BƯỚC 7: Explainability (SHAP Summary + Waterfall)')
    logger.info('=' * 60)

    default_h = DEFAULT_HORIZON if DEFAULT_HORIZON in h_list else h_list[0]
    default_data = prepare_data(df, horizon=default_h)

    xgb_default_path = os.path.join(models_dir, f'xgb_t{default_h}.json')
    if os.path.exists(xgb_default_path):
        best_model = xgb.XGBRegressor()
        best_model.load_model(xgb_default_path)
        best_model_name = f'XGBoost_t{default_h}'

        # Lưu best_model.joblib để dashboard Streamlit dùng
        joblib.dump(best_model, os.path.join(models_dir, 'best_model.joblib'))
        logger.info('Đã lưu best_model.joblib.')

        run_explainability(
            best_model_name, best_model,
            default_data['X_train'],
            default_data['X_test'],
            default_data['y_test'],
        )
    else:
        logger.warning('Không tìm thấy mô hình XGBoost t+%d — bỏ qua SHAP.', default_h)

    elapsed = time.time() - start_time
    logger.info('=' * 60)
    logger.info('HOÀN TẤT! Tổng thời gian: %.1fs', elapsed)
    logger.info('=' * 60)

    return {
        'df': df,
        'backtest_results': backtest_results,
        'models_dir': models_dir,
    }


def _print_backtest_summary(results: dict) -> None:
    """In bảng kết quả 5 kịch bản backtesting ra màn hình."""
    print('\n📊 BẢNG TỔNG HỢP 5 KỊCH BẢN BACKTESTING (XGBoost):')
    print('=' * 70)
    for scenario in ['K1', 'K2', 'K3']:
        r = results.get(scenario, {})
        h_label = {'K1': 't+1h', 'K2': 't+24h', 'K3': 't+72h'}[scenario]
        print(
            f'{scenario} ({h_label})'
            f'  │ MAE: {r.get("mae", float("nan")):.2f} µg/m³'
            f'  │ RMSE: {r.get("rmse", float("nan")):.2f} µg/m³'
            f'  │ F1-Macro: {r.get("f1_macro", float("nan")):.3f}'
        )
    r4 = results.get('K4', {})
    print(
        f'K4 (Top 5% Episode >= {r4.get("threshold_95", 0):.1f})'
        f'  │ MAE: {r4.get("mae", float("nan")):.2f} µg/m³'
        f'  │ Recall Unhealthy+: {r4.get("recall_unhealthy", float("nan")):.3f}'
    )
    r5 = results.get('K5', {})
    print(
        f'K5 (Mùa)'
        f'  │ Winter MAE: {r5.get("mae_winter", float("nan")):.2f} µg/m³'
        f'  │ Summer MAE: {r5.get("mae_summer", float("nan")):.2f} µg/m³'
    )
    print('=' * 70)


if __name__ == '__main__':
    run_pipeline()

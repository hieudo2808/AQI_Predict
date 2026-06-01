"""
╔══════════════════════════════════════════════════════════════╗
║  US AQI Hà Nội — Training Pipeline                         ║
║  Chạy: python -m src.pipelines.training_pipeline             ║
╚══════════════════════════════════════════════════════════════╝

Pipeline chuyên dùng để retrain mô hình định kỳ (vd: mỗi tuần/tháng).
Yêu cầu Ingestion và Feature Pipeline đã được chạy trước đó.
"""
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
from src.modeling.train import (
    prepare_data,
    train_and_save_models,
)
from src.modeling.evaluate import run_5_backtesting_scenarios
from src.visualization.eda import generate_all_eda_plots
from src.explainability.explain import run_explainability
from src.utils.logger import get_logger

logger = get_logger("TrainingPipeline")


def run_training_pipeline(
    horizons: list | None = None,
    save_models: bool = True,
    models_dir: str = 'models',
) -> dict | None:
    start_time = time.time()
    h_list = horizons or HORIZONS
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    features_path = 'data/features/features_targets.parquet'
    if not os.path.exists(features_path):
        logger.error(f'Không tìm thấy {features_path}. Vui lòng chạy Feature Pipeline trước.')
        return None

    logger.info('=' * 60)
    logger.info('BƯỚC 1: Load dữ liệu Features')
    logger.info('=' * 60)
    df = pd.read_parquet(features_path)
    logger.info('Dataset: %d dòng × %d cột', len(df), len(df.columns))

    logger.info('=' * 60)
    logger.info('BƯỚC 2: Huấn luyện & Lưu mô hình (%s)', h_list)
    logger.info('=' * 60)

    if save_models:
        train_and_save_models(df, features=FEATURES, horizons=h_list, save_dir=models_dir)
    else:
        logger.info('save_models=False — bỏ qua huấn luyện lại.')

    logger.info('=' * 60)
    logger.info('BƯỚC 3: Backtesting 5 kịch bản (K1–K5)')
    logger.info('=' * 60)

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
            continue
        data_h = prepare_data(df, horizon=h)
        model_h = xgb.XGBRegressor()
        model_h.load_model(xgb_path)
        preds = model_h.predict(data_h['X_test'])
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

    logger.info('=' * 60)
    logger.info('BƯỚC 4: EDA Plots (11 biểu đồ)')
    logger.info('=' * 60)
    df_eda = df.copy()
    if isinstance(df_eda.index, pd.DatetimeIndex):
        df_eda = df_eda.reset_index().rename(
            columns={df_eda.index.name or 'datetime': 'date'}
        )
    generate_all_eda_plots(df_eda, save_dir=FIGURES_DIR)

    logger.info('=' * 60)
    logger.info('BƯỚC 5: Explainability (SHAP Summary + Waterfall)')
    logger.info('=' * 60)

    default_h = DEFAULT_HORIZON if DEFAULT_HORIZON in h_list else h_list[0]
    default_data = prepare_data(df, horizon=default_h)

    xgb_default_path = os.path.join(models_dir, f'xgb_t{default_h}.json')
    if os.path.exists(xgb_default_path):
        best_model = xgb.XGBRegressor()
        best_model.load_model(xgb_default_path)
        best_model_name = f'XGBoost_t{default_h}'

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
    logger.info('HOÀN TẤT TRAINING PIPELINE! Tổng thời gian: %.1fs', elapsed)
    logger.info('=' * 60)

    return {
        'df': df,
        'backtest_results': backtest_results,
        'models_dir': models_dir,
    }


def _print_backtest_summary(results: dict) -> None:
    logger.info('BẢNG TỔNG HỢP 5 KỊCH BẢN BACKTESTING (XGBoost):')
    for scenario in ['K1', 'K2', 'K3']:
        r = results.get(scenario, {})
        h_label = {'K1': 't+1h', 'K2': 't+24h', 'K3': 't+72h'}[scenario]
        logger.info(
            f'[{scenario}] ({h_label})'
            f' | MAE: {r.get("mae", float("nan")):.2f} µg/m³'
            f' | RMSE: {r.get("rmse", float("nan")):.2f} µg/m³'
            f' | F1-Macro: {r.get("f1_macro", float("nan")):.3f}'
        )
    r4 = results.get('K4', {})
    logger.info(
        f'[K4] (Top 5% Episode >= {r4.get("threshold_95", 0):.1f})'
        f' | MAE: {r4.get("mae", float("nan")):.2f} µg/m³'
        f' | Recall Unhealthy+: {r4.get("recall_unhealthy", float("nan")):.3f}'
    )
    r5 = results.get('K5', {})
    logger.info(
        f'[K5] (Mùa)'
        f' | Winter MAE: {r5.get("mae_winter", float("nan")):.2f} µg/m³'
        f' | Summer MAE: {r5.get("mae_summer", float("nan")):.2f} µg/m³'
    )


if __name__ == '__main__':
    run_training_pipeline()

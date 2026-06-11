"""
Module huấn luyện và đánh giá mô hình.

Hỗ trợ:
  - Baseline models: Naive Persistence, Seasonal Naive
  - ML models: Linear Regression, Random Forest, XGBoost, LightGBM
  - Đánh giá: TimeSeriesSplit walk-forward CV + hold-out test
  - Multi-horizon: train riêng cho từng horizon (direct strategy)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    FEATURES, TARGET, TRAIN_RATIO, VALID_RATIO,
    MODEL_CONFIGS, HORIZONS, CV_N_SPLITS,
)
from src.modeling.wrappers import ProphetWrapper, SARIMAXWrapper
from src.utils.logger import get_logger

logger = get_logger("Train")

FORECAST_MODES = {'operational', 'oracle_weather'}


# ═════════════════════════════════════════════════════════════
# CHIA DỮ LIỆU
# ═════════════════════════════════════════════════════════════

def prepare_data(df, features=None, weather_cols=None, horizon=1, forecast_mode='operational'):
    """
    Chuẩn bị dữ liệu train/validation/test cho 1 horizon cụ thể.

    Chia theo thời gian (KHÔNG shuffle): Train 70% / Valid 10% / Test 20%.
    Áp dụng Perfect Prognosis nếu weather_cols được cung cấp.

    Parameters:
        df (pd.DataFrame): DataFrame đã có features + target columns.
        features (list, optional): Danh sách tên cột features.
        weather_cols (list, optional): Danh sách các cột thời tiết có thể dịch chuyển
            trong chế độ oracle_weather.
        horizon (int): Horizon cần dự báo theo giờ (1, 24, 48, hoặc 72).
        forecast_mode (str): 'operational' không dùng actual future weather.
            'oracle_weather' shift weather_cols để đo upper-bound Perfect Prognosis.

    Returns:
        dict: {
            'X_train', 'X_valid', 'X_test',
            'y_train', 'y_valid', 'y_test',
            'dates_test', 'horizon',
        }
    """
    if forecast_mode not in FORECAST_MODES:
        raise ValueError(f"forecast_mode phải thuộc {sorted(FORECAST_MODES)}, nhận: {forecast_mode}")

    feat = features or FEATURES
    target_col = f'target_{TARGET}_t{horizon}'

    df_h = df.copy()

    # Chỉ oracle_weather mới được dùng actual future weather.
    if weather_cols and forecast_mode == 'oracle_weather':
        logger.info(f"Đã KÍCH HOẠT Perfect Prognosis: Shift các biến thời tiết đi {-horizon} bước.")
        for w_col in weather_cols:
            if w_col in df_h.columns:
                df_h[w_col] = df_h[w_col].shift(-horizon)
    elif weather_cols:
        logger.info("forecast_mode=operational — không shift actual future weather.")

    # Chỉ lấy các dòng có target hợp lệ cho horizon này và loại bỏ NaN do shift
    valid_mask = df_h[target_col].notna() & df_h[feat].notna().all(axis=1)
    df_h = df_h[valid_mask].copy()

    X = df_h[feat]
    y = df_h[target_col]

    n = len(df_h)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    X_train = X.iloc[:train_end]
    X_valid = X.iloc[train_end:valid_end]
    X_test = X.iloc[valid_end:]

    y_train = y.iloc[:train_end]
    y_valid = y.iloc[train_end:valid_end]
    y_test = y.iloc[valid_end:]

    dates_test = df_h['date'].iloc[valid_end:] if 'date' in df_h.columns else None

    logger.info(f"Horizon t+{horizon}: Features: {len(feat)} | Train: {len(X_train)} | Valid: {len(X_valid)} | Test: {len(X_test)} | Target: {target_col}")

    return {
        'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
        'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
        'dates_test': dates_test,
        'horizon': horizon,
        'forecast_mode': forecast_mode,
        'target_col': target_col,
        'df_h': df_h,
    }


# ═════════════════════════════════════════════════════════════
# BASELINE MODELS
# ═════════════════════════════════════════════════════════════

def naive_persistence(y_train, y_test, horizon=1):
    """
    Baseline Naive Persistence: ŷ(t+h) = y(t).

    Dự báo giá trị tương lai bằng giá trị hiện tại (tức là lùi lại h bước).
    Sử dụng vectorized operations thay vì for-loop (coding-rules.md requirement).

    Parameters:
        y_train (pd.Series): Giá trị training (dùng để lấy giá trị quá khứ).
        y_test (pd.Series): Giá trị thực tế test.
        horizon (int): Cần lùi h bước để lấy giá trị t từ chuỗi y(t+h).

    Returns:
        dict: Kết quả {'name', 'pred', 'MAE', 'RMSE', 'R2'}
    """
    # Vectorized: ghép chuỗi và dùng shift()
    full = pd.concat([y_train, y_test])
    shifted = full.shift(horizon)
    # Lấy predictions cho test period (từ index = len(y_train))
    pred = shifted.iloc[len(y_train):].values
    # Fallback: fill NaN bằng mean của train
    train_mean = y_train.mean()
    pred = np.where(np.isnan(pred), train_mean, pred)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        'name': 'Naive Persistence',
        'model': None,
        'pred': pred,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
    }


def seasonal_naive(y_train, y_test, season_period=168, horizon=None):
    """
    Baseline Seasonal Naive: ŷ(t+h) = y(t+h−s).

    Dự báo = giá trị cùng ngày trong tuần trước (s=7 cho daily).
    Sử dụng vectorized operations thay vì for-loop (coding-rules.md requirement).

    Parameters:
        y_train (pd.Series): Giá trị training.
        y_test (pd.Series): Giá trị thực tế test.
        season_period (int): Chu kỳ mùa vụ theo số bước thời gian.
            Với dữ liệu hourly: 24 = ngày, 168 = tuần.
        horizon (int, optional): Horizon dự báo. Nếu season_period < horizon thì
            baseline không hợp lệ vì cần quan sát sau forecast origin.

    Returns:
        dict: Kết quả {'name', 'pred', 'MAE', 'RMSE', 'R2'}
    """
    if horizon is not None and season_period < horizon:
        raise ValueError(
            f"Seasonal naive không hợp lệ: season_period={season_period} < horizon={horizon}. "
            "Điều này sẽ dùng thông tin sau forecast origin."
        )

    # Vectorized: ghép chuỗi và dùng shift()
    full = pd.concat([y_train, y_test])
    shifted = full.shift(season_period)
    # Lấy predictions cho test period
    pred = shifted.iloc[len(y_train):].values
    # Fallback: fill NaN bằng mean của train
    train_mean = y_train.mean()
    pred = np.where(np.isnan(pred), train_mean, pred)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        'name': f'Seasonal Naive ({season_period}h)',
        'model': None,
        'pred': pred,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
    }


def shifted_baseline(y_train, y_valid, y_test, shift_period, name, horizon=None):
    """
    Baseline theo shift trên chuỗi target đầy đủ train+valid+test.

    y_valid được giữ trong chuỗi để không tạo khoảng hở giả giữa train và test.
    """
    if horizon is not None and shift_period < horizon:
        raise ValueError(
            f"{name} không hợp lệ: shift_period={shift_period} < horizon={horizon}."
        )

    full = pd.concat([y_train, y_valid, y_test])
    shifted = full.shift(shift_period)
    test_start = len(y_train) + len(y_valid)
    pred = shifted.iloc[test_start:].values
    train_mean = y_train.mean()
    pred = np.where(np.isnan(pred), train_mean, pred)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        'name': name,
        'model': None,
        'pred': pred,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
    }


def rolling_mean_baseline(y_train, y_valid, y_test, window, horizon, name=None):
    """Rolling-mean baseline chỉ dùng giá trị quan sát được tại forecast origin."""
    full = pd.concat([y_train, y_valid, y_test])
    observed_at_origin = full.shift(horizon)
    rolled = observed_at_origin.rolling(window=window, min_periods=1).mean()
    test_start = len(y_train) + len(y_valid)
    pred = rolled.iloc[test_start:].values
    train_mean = y_train.mean()
    pred = np.where(np.isnan(pred), train_mean, pred)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        'name': name or f'Rolling Mean ({window}h)',
        'model': None,
        'pred': pred,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
    }


# ═════════════════════════════════════════════════════════════
# TRAIN & EVALUATE
# ═════════════════════════════════════════════════════════════

def train_evaluate(name, model, X_train, y_train, X_test, y_test):
    """
    Huấn luyện và đánh giá một mô hình trên tập test.

    Parameters:
        name (str): Tên mô hình.
        model: Instance của model (sklearn-compatible).
        X_train, y_train: Dữ liệu training.
        X_test, y_test: Dữ liệu testing.

    Returns:
        dict: Kết quả {'name', 'model', 'pred', 'MAE', 'RMSE', 'R2'}
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    logger.info(f'[{name:<20}] MAE={mae:6.2f} | RMSE={rmse:6.2f} | R2={r2:.4f}')

    return {
        'name': name,
        'model': model,
        'pred': pred,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
    }


def run_cv(name, model_class, model_params, X, y, n_splits=None):
    """
    Cross-validation theo TimeSeriesSplit.

    Parameters:
        name (str): Tên mô hình.
        model_class: Class của model.
        model_params (dict): Tham số khởi tạo.
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        n_splits (int, optional): Số folds. Mặc định CV_N_SPLITS.

    Returns:
        dict: {'name', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std'}
    """
    splits = n_splits or CV_N_SPLITS
    tscv = TimeSeriesSplit(n_splits=splits)
    mae_list, rmse_list, r2_list = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        model = model_class(**model_params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[te_idx])

        mae_list.append(mean_absolute_error(y.iloc[te_idx], pred))
        rmse_list.append(np.sqrt(mean_squared_error(y.iloc[te_idx], pred)))
        r2_list.append(r2_score(y.iloc[te_idx], pred))

    result = {
        'name': name,
        'mae_mean': round(float(np.mean(mae_list)), 2),
        'mae_std': round(float(np.std(mae_list)), 2),
        'rmse_mean': round(float(np.mean(rmse_list)), 2),
        'rmse_std': round(float(np.std(rmse_list)), 2),
        'r2_mean': round(float(np.mean(r2_list)), 4),
        'r2_std': round(float(np.std(r2_list)), 4),
    }

    logger.info(f'[{name:<20}] CV MAE={result["mae_mean"]:5.2f}±{result["mae_std"]:.2f} | RMSE={result["rmse_mean"]:5.2f}±{result["rmse_std"]:.2f} | R2={result["r2_mean"]:.4f}')

    return result


# ═════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ═════════════════════════════════════════════════════════════

def run_all_models(data_dict):
    """
    Chạy tất cả mô hình (Baseline + ML) trên 1 horizon.

    Parameters:
        data_dict (dict): Output từ prepare_data().

    Returns:
        list[dict]: Danh sách kết quả của từng mô hình.
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    y_valid = data_dict['y_valid']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    horizon = data_dict['horizon']

    logger.info(f'Bắt đầu huấn luyện cho horizon t+{horizon}...')
    
    # ─── Baseline ───
    results = [
        shifted_baseline(y_train, y_valid, y_test, horizon, 'Persistence', horizon=horizon),
        rolling_mean_baseline(y_train, y_valid, y_test, 24, horizon, 'Rolling Mean (24h)'),
        rolling_mean_baseline(y_train, y_valid, y_test, 168, horizon, 'Rolling Mean (168h)'),
        shifted_baseline(
            y_train, y_valid, y_test, 168, 'Seasonal Naive (168h)', horizon=horizon
        ),
    ]
    if horizon <= 24:
        results.append(
            shifted_baseline(
                y_train, y_valid, y_test, 24, 'Seasonal Naive (24h)', horizon=horizon
            )
        )
    # In kết quả baseline
    for r in results:
        logger.info(f'[{r["name"]:<20}] MAE={r["MAE"]:6.2f} | RMSE={r["RMSE"]:6.2f} | R2={r["R2"]:.4f}')

    # ─── ML Models ───
    results.extend([
        train_evaluate(
            'Linear Regression',
            LinearRegression(),
            X_train, y_train, X_test, y_test
        ),
        train_evaluate(
            'Random Forest',
            RandomForestRegressor(**MODEL_CONFIGS['Random Forest']),
            X_train, y_train, X_test, y_test
        ),
        train_evaluate(
            'XGBoost',
            xgb.XGBRegressor(**MODEL_CONFIGS['XGBoost']),
            X_train, y_train, X_test, y_test
        ),
        train_evaluate(
            'LightGBM',
            lgb.LGBMRegressor(**MODEL_CONFIGS['LightGBM']),
            X_train, y_train, X_test, y_test
        ),
        train_evaluate(
            'Prophet',
            ProphetWrapper(**MODEL_CONFIGS['Prophet']),
            X_train, y_train, X_test, y_test
        ),
        train_evaluate(
            'SARIMAX',
            SARIMAXWrapper(**MODEL_CONFIGS['SARIMAX']),
            X_train, y_train, X_test, y_test
        ),
    ])

    logger.info('Hoàn thành huấn luyện tất cả mô hình!')
    return results


def run_cv_all_models(data_dict):
    """
    Chạy TimeSeriesSplit cross-validation cho tất cả ML models.

    Parameters:
        data_dict (dict): Output từ prepare_data().

    Returns:
        list[dict]: Danh sách kết quả CV.
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    horizon = data_dict['horizon']

    logger.info(f'TimeSeriesSplit CV (k={CV_N_SPLITS}) cho horizon t+{horizon}...')

    cv_results = [
        run_cv('Linear Regression', LinearRegression, {}, X_train, y_train),
        run_cv('Random Forest', RandomForestRegressor, MODEL_CONFIGS['Random Forest'], X_train, y_train),
        run_cv('XGBoost', xgb.XGBRegressor, MODEL_CONFIGS['XGBoost'], X_train, y_train),
        run_cv('LightGBM', lgb.LGBMRegressor, MODEL_CONFIGS['LightGBM'], X_train, y_train),
        run_cv('Prophet', ProphetWrapper, MODEL_CONFIGS['Prophet'], X_train, y_train),
        run_cv('SARIMAX', SARIMAXWrapper, MODEL_CONFIGS['SARIMAX'], X_train, y_train),
    ]

    return cv_results


# ═════════════════════════════════════════════════════════════
# SO SÁNH KẾT QUẢ
# ═════════════════════════════════════════════════════════════

def compare_models(results):
    """
    Tạo bảng so sánh kết quả các mô hình.

    Parameters:
        results (list[dict]): Danh sách kết quả từ run_all_models.

    Returns:
        pd.DataFrame: Bảng so sánh với cột Xếp hạng.
    """
    df_results = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ['model', 'pred']} for r in results]
    ).set_index('name')

    df_results['Xếp hạng'] = df_results['MAE'].rank(ascending=True).astype(int)
    df_results = df_results.sort_values('MAE', ascending=True)

    logger.info('Bảng so sánh kết quả:\n' + df_results.to_string())

    best = df_results.index[0]
    logger.info(f'Mô hình tốt nhất: {best}')

    return df_results


def compare_cv_results(cv_results):
    """
    Tạo bảng so sánh kết quả CV.

    Parameters:
        cv_results (list[dict]): Danh sách kết quả từ run_cv_all_models.

    Returns:
        pd.DataFrame: Bảng so sánh CV.
    """
    df = pd.DataFrame(cv_results).set_index('name')
    df['Xếp hạng'] = df['mae_mean'].rank(ascending=True).astype(int)
    df = df.sort_values('mae_mean', ascending=True)

    logger.info('Bảng so sánh CV:\n' + df.to_string())

    return df


def train_and_save_models(df, features=None, horizons=None, save_dir='models'):
    """
    Huấn luyện mô hình XGBoost và SARIMA cho các horizon và lưu chúng vào thư mục save_dir.
    
    Parameters:
        df (pd.DataFrame): DataFrame có features và target.
        features (list, optional): Danh sách features.
        horizons (list, optional): Danh sách horizons. Mặc định [1, 24, 72].
        save_dir (str): Thư mục lưu mô hình.
    """
    import os
    import pickle
    from src.config import WEATHER_FEATURES
    
    os.makedirs(save_dir, exist_ok=True)
    feat = features or FEATURES
    h_list = horizons or HORIZONS
    
    for h in h_list:
        data_dict = prepare_data(df, features=feat, weather_cols=WEATHER_FEATURES, horizon=h)
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        # 1. Huấn luyện XGBoost
        xgb_params = MODEL_CONFIGS['XGBoost']
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        xgb_path = os.path.join(save_dir, f'xgb_t{h}.json')
        xgb_model.save_model(xgb_path)
        logger.info(f"Đã lưu mô hình XGBoost horizon t+{h}h vào: {xgb_path}")
        
        # 2. Huấn luyện SARIMA
        sarima_params = MODEL_CONFIGS['SARIMAX']
        sarima_model = SARIMAXWrapper(**sarima_params)
        sarima_model.fit(X_train, y_train)
        
        sarima_path = os.path.join(save_dir, f'sarima_t{h}.pkl')
        with open(sarima_path, 'wb') as f:
            pickle.dump(sarima_model, f)
        logger.info(f"Đã lưu mô hình SARIMA horizon t+{h}h vào: {sarima_path}")

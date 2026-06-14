"""
Hyperparameter tuning bằng Time-Series Cross-Validation.

Mục tiêu: bộ siêu tham số của mỗi mô hình phải là KẾT QUẢ của quá trình
TimeSeriesSplit CV trên tập Train+Validation, KHÔNG phải chọn cứng từ đầu
(theo góp ý quy trình Chương 5).

Quy tắc:
  - Chỉ tune các mô hình có grid khai báo trong TUNING_GRIDS.
  - Dùng TimeSeriesSplit (không shuffle) để tôn trọng trật tự thời gian.
  - Scoring = neg_mean_absolute_error (đồng nhất với tiêu chí chọn champion theo MAE).
  - best_estimator_ được refit lại trên toàn bộ (X, y) đưa vào, sẵn sàng predict test.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import CV_N_SPLITS, RANDOM_SEED


# Grid cho từng mô hình. Tên tham số khớp với estimator/pipeline do model_specs tạo ra.
# Pipeline make_pipeline(StandardScaler(), Ridge()) đặt tên bước là 'ridge', 'elasticnet'.
TUNING_GRIDS: dict[str, dict[str, list]] = {
    "Ridge": {
        "ridge__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0],
    },
    "ElasticNet": {
        "elasticnet__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 30.0],
        "elasticnet__l1_ratio": [0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95],
    },
    # RandomForest / ExtraTrees cố tình KHÔNG tune (mỗi fit ~3.5s trên 8700 dòng,
    # rất chậm). Chúng vẫn chạy với tham số mặc định để có mặt trong leaderboard.
    "XGBoost": {
        "n_estimators": [100, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 7, 8, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    # Window models: tune độ co giãn của Ridge ở đầu pipeline.
    "WindowRidge_168h": {
        "ridge__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0],
    },
}


def _grid_size(grid: dict[str, list]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def tune_model(
    spec,
    X,
    y,
    n_splits: int | None = None,
    n_iter: int = 16,
    seed: int = RANDOM_SEED,
) -> tuple[Any, dict, bool]:
    """
    Tìm siêu tham số tốt nhất cho `spec` bằng TimeSeriesSplit CV.

    Parameters:
        spec (ModelSpec): Đặc tả mô hình (dùng spec.name để tra grid, spec.factory() để tạo base).
        X, y: Dữ liệu tune (nên là Train+Validation).
        n_splits (int): Số fold TimeSeriesSplit. Mặc định CV_N_SPLITS.
        n_iter (int): Số tổ hợp thử tối đa của RandomizedSearchCV.
        seed (int): Random seed.

    Returns:
        tuple (model, best_params, tuned):
            - model: estimator đã refit trên toàn bộ (X, y).
            - best_params: dict tham số tốt nhất ('' nếu không tune).
            - tuned: True nếu có chạy CV search, False nếu mô hình không có grid.
    """
    grid = TUNING_GRIDS.get(spec.name)
    base = spec.factory()

    # Mô hình không khai báo grid (CatBoost, TabPFN, WindowMLP...) -> giữ tham số mặc định.
    if not grid:
        base.fit(X, y)
        return base, {}, False

    splits = n_splits or CV_N_SPLITS
    tscv = TimeSeriesSplit(n_splits=splits)
    max_combos = _grid_size(grid)
    n_iter_eff = int(min(n_iter, max_combos))

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=grid,
        n_iter=n_iter_eff,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
    )
    search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_), True


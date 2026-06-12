"""
Model selection pipeline for PM2.5 forecasting.

This pipeline compares candidate forecasting models, writes selection artifacts
under reports/model_selection, and exports deployment champions for the
dashboard.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.aqi_formula import aqi_to_level, pm25_to_aqi
from src.config import (
    FEATURES,
    HORIZONS,
    TARGET,
    WEATHER_FEATURES,
)
from src.modeling.champion import DEFAULT_MANIFEST_PATH
from src.modeling.model_specs import (
    ModelSpec,
    build_neuralforecast_models as _build_neuralforecast_models,
    module_available as _module_available,
    neuralforecast_model_names as _neuralforecast_model_names,
    preflight_environment,
    safe_model_name as _safe_name,
    statistical_specs as _statistical_specs,
    tabular_specs as _tabular_specs,
    window_sequence_specs as _window_sequence_specs,
)
from src.modeling.train import prepare_data, rolling_mean_baseline, shifted_baseline
from src.modeling.tuning import tune_model as _tune_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features_targets.parquet"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_selection"
MODEL_DIR = PROJECT_ROOT / "models" / "model_selection"
CHAMPION_DIR = PROJECT_ROOT / "models" / "champions"
UNHEALTHY_LEVELS = {
    "Unhealthy for Sensitive",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
}


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    true_levels = [aqi_to_level(pm25_to_aqi(v))[0] for v in y_true]
    pred_levels = [aqi_to_level(pm25_to_aqi(v))[0] for v in y_pred]
    true_unhealthy = [lvl in UNHEALTHY_LEVELS for lvl in true_levels]
    recall_den = sum(true_unhealthy)
    recall_num = sum(
        1 for is_true, pred_lvl in zip(true_unhealthy, pred_levels)
        if is_true and pred_lvl in UNHEALTHY_LEVELS
    )
    recall_unhealthy = recall_num / recall_den if recall_den else 0.0

    threshold_95 = np.percentile(y_true, 95)
    high_mask = y_true >= threshold_95
    top5_mae = mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.any() else np.nan

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 6),
        "recall_unhealthy_plus": round(float(recall_unhealthy), 6),
        "mae_top5": round(float(top5_mae), 4),
        "threshold_95": round(float(threshold_95), 4),
        "n_test": int(len(y_true)),
    }


def _run_neuralforecast_models(
    df: pd.DataFrame,
    mode: str,
    horizons: list[int],
    quick: bool,
) -> tuple[list[dict], list[dict]]:
    if not _module_available("neuralforecast"):
        return [], [
            {
                "model": name,
                "status": "skipped",
                "reason": "neuralforecast>=3.1.0 is not installed; current Python/Torch/Ray constraints may block modern sequential models",
            }
            for name in _neuralforecast_model_names()
        ]

    try:
        from neuralforecast import NeuralForecast
    except Exception as exc:  # noqa: BLE001
        return [], [
            {
                "model": name,
                "status": "skipped",
                "reason": f"neuralforecast import failed: {exc}",
            }
            for name in _neuralforecast_model_names()
        ]

    rows = []
    skipped = []
    max_horizon = max(horizons)
    train_cut = int(len(df) * 0.8)
    holdout_size = len(df) - train_cut
    if holdout_size < max_horizon:
        return [], [
            {
                "model": name,
                "status": "skipped",
                "reason": "not enough holdout rows for max horizon",
            }
            for name in _neuralforecast_model_names()
        ]

    nf_train = df[["date", TARGET]].rename(columns={"date": "ds", TARGET: "y"})
    nf_train["unique_id"] = "hanoi_pm25"
    nf_train = nf_train[["unique_id", "ds", "y"]]

    try:
        models = _build_neuralforecast_models(max_horizon=max_horizon, quick=quick)
    except Exception as exc:  # noqa: BLE001
        return [], [
            {
                "model": name,
                "status": "skipped",
                "reason": f"neuralforecast model construction failed: {str(exc)[:300]}",
            }
            for name in _neuralforecast_model_names()
        ]

    start = time.perf_counter()
    try:
        nf = NeuralForecast(models=models, freq="h")
        n_windows = 2 if quick else min(6, max(1, holdout_size // max_horizon))
        test_size = max_horizon * n_windows
        if len(nf_train) <= 168 + max_horizon + test_size:
            raise ValueError("not enough rows for NeuralForecast rolling cross-validation")
        forecast = nf.cross_validation(
            df=nf_train,
            val_size=max_horizon,
            test_size=test_size,
            n_windows=n_windows,
            step_size=max_horizon,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        return [], [
            {
                "model": name,
                "status": "skipped",
                "reason": f"neuralforecast run failed: {str(exc)[:300]}",
            }
            for name in _neuralforecast_model_names()
        ]
    runtime = time.perf_counter() - start

    if "cutoff" not in forecast.columns or "y" not in forecast.columns:
        return [], [
            {
                "model": name,
                "status": "skipped",
                "reason": "NeuralForecast cross_validation output lacks cutoff or y columns",
            }
            for name in _neuralforecast_model_names()
        ]

    forecast = forecast.copy()
    forecast["forecast_horizon"] = (
        (pd.to_datetime(forecast["ds"]) - pd.to_datetime(forecast["cutoff"])) / pd.Timedelta(hours=1)
    ).round().astype(int)

    for model_name in _neuralforecast_model_names():
        if model_name not in forecast.columns:
            skipped.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "reason": "model column missing from NeuralForecast output",
                }
            )
            continue
        for horizon in horizons:
            eval_rows = forecast[forecast["forecast_horizon"].eq(horizon)]
            if eval_rows.empty:
                skipped.append(
                    {
                        "model": model_name,
                        "status": "skipped",
                        "reason": f"no cross-validation rows for horizon {horizon}",
                    }
                )
                continue
            y_true = eval_rows["y"].to_numpy()
            y_pred = eval_rows[model_name].to_numpy()
            metrics = _regression_metrics(y_true, y_pred)
            rows.append(
                {
                    "mode": mode,
                    "horizon": horizon,
                    "model": model_name,
                    "group": "neuralforecast",
                    "complexity_rank": 4,
                    "runtime_seconds": round(float(runtime), 4),
                    "status": "ok",
                    "error": "",
                    **metrics,
                }
            )

    if mode == "oracle_weather":
        for row in rows:
            row["error"] = "NeuralForecast runner currently uses y history only; oracle weather is not injected."

    return rows, skipped


def _baseline_results(data: dict) -> list[dict]:
    y_train = data["y_train"]
    y_valid = data["y_valid"]
    y_test = data["y_test"]
    horizon = data["horizon"]
    rows = []

    specs = [
        lambda: shifted_baseline(y_train, y_valid, y_test, horizon, "Persistence", horizon=horizon),
        lambda: rolling_mean_baseline(y_train, y_valid, y_test, 24, horizon, "RollingMean_24h"),
        lambda: rolling_mean_baseline(y_train, y_valid, y_test, 168, horizon, "RollingMean_168h"),
        lambda: shifted_baseline(y_train, y_valid, y_test, 168, "SeasonalNaive_168h", horizon=horizon),
    ]
    if horizon <= 24:
        specs.append(
            lambda: shifted_baseline(y_train, y_valid, y_test, 24, "SeasonalNaive_24h", horizon=horizon)
        )

    for build_result in specs:
        result = build_result()
        metrics = _regression_metrics(y_test.to_numpy(), np.asarray(result["pred"], dtype=float))
        rows.append(
            {
                "model": result["name"],
                "group": "corrected_baseline",
                "complexity_rank": 0,
                "runtime_seconds": 0.0,
                **metrics,
            }
        )
    return rows


def _fit_predict_tabular(
    data: dict, spec: ModelSpec, tune: bool = False
) -> tuple[np.ndarray, float, object, dict, bool]:
    X_train = data["X_train"]
    y_train = data["y_train"]

    start = time.perf_counter()
    if tune:
        # Tune bằng TimeSeriesSplit CV trên Train(+Valid). best_estimator_ refit luôn
        # trên toàn bộ dữ liệu tune -> tham số là kết quả CV, không phải chọn cứng.
        if data.get("X_valid") is not None and len(data["X_valid"]) > 0:
            X_tune = pd.concat([X_train, data["X_valid"]])
            y_tune = pd.concat([y_train, data["y_valid"]])
        else:
            X_tune, y_tune = X_train, y_train
        model, best_params, tuned = _tune_model(spec, X_tune, y_tune)
    else:
        model = spec.factory()
        model.fit(X_train, y_train)
        best_params, tuned = {}, False

    pred = model.predict(data["X_test"])
    runtime = time.perf_counter() - start
    return np.asarray(pred, dtype=float), runtime, model, best_params, tuned


def _fit_full_tabular_model(df: pd.DataFrame, horizon: int, mode: str, spec: ModelSpec):
    data = prepare_data(
        df,
        features=FEATURES,
        weather_cols=WEATHER_FEATURES,
        horizon=horizon,
        forecast_mode=mode,
    )
    X = pd.concat([data["X_train"], data["X_valid"], data["X_test"]])
    y = pd.concat([data["y_train"], data["y_valid"], data["y_test"]])
    model = spec.factory()
    model.fit(X, y)
    return model, list(X.columns), len(X)


def _window_data(data: dict, lookback: int = 168) -> dict | None:
    df_h = data["df_h"].copy()
    target_col = data["target_col"]

    X = pd.DataFrame(
        {f"pm2_5_seq_lag_{lag}": df_h[TARGET].shift(lag) for lag in range(1, lookback + 1)},
        index=df_h.index,
    )
    y = df_h[target_col]
    valid_mask = y.notna() & X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    train_idx = data["y_train"].index.intersection(X.index)
    test_idx = data["y_test"].index.intersection(X.index)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None

    return {
        "X_train": X.loc[train_idx],
        "y_train": y.loc[train_idx],
        "X_test": X.loc[test_idx],
        "y_test": y.loc[test_idx],
    }


def _fit_predict_statistical(data: dict, spec: ModelSpec) -> tuple[np.ndarray, float, object]:
    """Fit SARIMAX trên Train+Valid rồi forecast khối Test liền kề.

    SARIMAX.forecast() nối tiếp từ điểm cuối dữ liệu huấn luyện, nên phải fit trên
    Train+Valid để dự báo căn đúng vào tập Test (Test là phần đuôi liền sau Valid).
    """
    X_fit = pd.concat([data["X_train"], data["X_valid"]])
    y_fit = pd.concat([data["y_train"], data["y_valid"]])

    model = spec.factory()
    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    pred = np.asarray(model.predict(data["X_test"]), dtype=float)
    runtime = time.perf_counter() - start
    return pred, runtime, model


def _run_specs(
    data: dict, specs: list[ModelSpec], save_models: bool, mode: str, horizon: int, tune: bool = False
) -> list[dict]:
    rows = []
    for spec in specs:
        try:
            best_params: dict = {}
            tuned = False
            if spec.group == "statistical":
                pred, runtime, model = _fit_predict_statistical(data, spec)
                y_true = data["y_test"].to_numpy()
            elif spec.group == "sequential_window":
                seq = _window_data(data)
                if seq is None:
                    rows.append(_error_row(spec, "not enough sequence rows"))
                    continue
                pred, runtime, model, best_params, tuned = _fit_predict_tabular(seq, spec, tune=tune)
                y_true = seq["y_test"].to_numpy()
            else:
                pred, runtime, model, best_params, tuned = _fit_predict_tabular(data, spec, tune=tune)
                y_true = data["y_test"].to_numpy()

            metrics = _regression_metrics(y_true, pred)
            rows.append(
                {
                    "model": spec.name,
                    "group": spec.group,
                    "complexity_rank": spec.complexity_rank,
                    "runtime_seconds": round(float(runtime), 4),
                    "status": "ok",
                    "error": "",
                    "tuned": tuned,
                    "best_params": json.dumps(best_params, ensure_ascii=False) if best_params else "",
                    **metrics,
                }
            )
            if save_models:
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, MODEL_DIR / f"{mode}_t{horizon}_{_safe_name(spec.name)}.joblib")
        except Exception as exc:  # noqa: BLE001 - benchmark must continue after model failures.
            rows.append(_error_row(spec, str(exc)))
    return rows


def _error_row(spec: ModelSpec, error: str) -> dict:
    return {
        "model": spec.name,
        "group": spec.group,
        "complexity_rank": spec.complexity_rank,
        "runtime_seconds": 0.0,
        "status": "error",
        "error": error[:500],
        "tuned": False,
        "best_params": "",
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "recall_unhealthy_plus": np.nan,
        "mae_top5": np.nan,
        "threshold_95": np.nan,
        "n_test": 0,
    }


def run_benchmark(
    mode: str = "operational",
    horizons: list[int] | None = None,
    quick: bool = False,
    include_optional: bool = True,
    include_tabpfn: bool = True,
    save_models: bool = False,
    run_neuralforecast: bool = False,
    tune: bool = False,
    include_sarimax: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES_PATH)
    h_list = horizons or HORIZONS
    tabular_specs, skipped = _tabular_specs(
        include_optional=include_optional and not quick,
        include_tabpfn=include_tabpfn and not quick,
    )
    sequence_specs = _window_sequence_specs()
    if include_sarimax:
        stat_specs, stat_skipped = _statistical_specs(include_optional=include_optional)
        skipped.extend(stat_skipped)
    else:
        stat_specs = []
    if quick:
        keep = {"Ridge", "RandomForest", "WindowRidge_168h"}
        tabular_specs = [spec for spec in tabular_specs if spec.name in keep]
        sequence_specs = [spec for spec in sequence_specs if spec.name in keep]
        stat_specs = []

    rows = []
    for horizon in h_list:
        data = prepare_data(
            df,
            features=FEATURES,
            weather_cols=WEATHER_FEATURES,
            horizon=horizon,
            forecast_mode=mode,
        )

        for result in _baseline_results(data):
            rows.append({"mode": mode, "horizon": horizon, "status": "ok", "error": "", **result})

        all_specs = stat_specs + tabular_specs + sequence_specs
        for result in _run_specs(data, all_specs, save_models, mode, horizon, tune=tune):
            rows.append({"mode": mode, "horizon": horizon, **result})

    if run_neuralforecast and not quick:
        nf_rows, nf_skipped = _run_neuralforecast_models(df, mode, h_list, quick=quick)
        rows.extend(nf_rows)
        skipped.extend(nf_skipped)

    metrics = pd.DataFrame(rows)
    skip_rows = [
        {"model": item["model"], "status": "skipped", "reason": item["reason"]}
        for item in skipped
    ]
    if (not run_neuralforecast) or (not _module_available("neuralforecast")) or quick:
        skip_rows.extend(
            {
                "model": name,
                "status": "skipped",
                "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
            }
            for name in _neuralforecast_model_names()
        )

    return _select_champions(metrics), pd.DataFrame(skip_rows)


def _select_champions(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["rank_mae"] = np.nan
    metrics["selected_champion"] = False
    ok_mask = metrics["status"].eq("ok") & metrics["MAE"].notna()
    metrics.loc[ok_mask, "rank_mae"] = (
        metrics[ok_mask].groupby(["mode", "horizon"])["MAE"].rank(method="min", ascending=True)
    )

    for _, group in metrics[ok_mask].groupby(["mode", "horizon"]):
        min_mae = group["MAE"].min()
        candidates = group[group["MAE"] <= min_mae * 1.01].sort_values(
            ["complexity_rank", "runtime_seconds", "MAE"]
        )
        metrics.loc[candidates.index[0], "selected_champion"] = True

    return metrics.sort_values(["mode", "horizon", "rank_mae", "complexity_rank", "MAE"], na_position="last")


def _md_table(df: pd.DataFrame, columns: list[str]) -> str:
    data = df[columns].fillna("")
    widths = {
        col: max(len(str(col)), *(len(str(v)) for v in data[col].tolist()))
        for col in columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    lines = [header, sep]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in columns) + " |")
    return "\n".join(lines)


def _spec_lookup(include_optional: bool, include_tabpfn: bool = True) -> dict[str, ModelSpec]:
    specs, _ = _tabular_specs(include_optional=include_optional, include_tabpfn=include_tabpfn)
    specs.extend(_window_sequence_specs())
    return {spec.name: spec for spec in specs}


def export_champions(metrics: pd.DataFrame, include_optional: bool, include_tabpfn: bool = True) -> dict:
    df = pd.read_parquet(FEATURES_PATH)
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    specs = _spec_lookup(include_optional=include_optional, include_tabpfn=include_tabpfn)
    operational = metrics[
        metrics["mode"].eq("operational")
        & metrics["status"].eq("ok")
        & metrics["selected_champion"].eq(True)
    ].sort_values("horizon")

    manifest = {
        "version": 1,
        "created_at": pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").isoformat(),
        "target": TARGET,
        "forecast_mode": "operational",
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins",
        "champions": {},
    }

    for _, row in operational.iterrows():
        horizon = int(row["horizon"])
        model_name = row["model"]
        spec = specs.get(model_name)
        if spec is None or spec.group == "corrected_baseline":
            candidates = metrics[
                metrics["mode"].eq("operational")
                & metrics["horizon"].eq(horizon)
                & metrics["status"].eq("ok")
                & metrics["group"].isin(["tabular_linear", "tabular_tree", "tabular_boosting", "tabular_foundation"])
            ].sort_values(["rank_mae", "complexity_rank", "runtime_seconds"])
            if candidates.empty:
                raise RuntimeError(f"Không có deployment-capable champion cho horizon {horizon}")
            row = candidates.iloc[0]
            model_name = row["model"]
            spec = specs[model_name]

        if spec.group == "sequential_window":
            raise RuntimeError(
                f"Champion {model_name} cho horizon {horizon} dùng sequential_window chưa hỗ trợ deployment export."
            )

        model, feature_columns, train_rows = _fit_full_tabular_model(df, horizon, "operational", spec)
        artifact_rel = Path("models") / "champions" / f"t{horizon}_{_safe_name(model_name)}.joblib"
        joblib.dump(model, PROJECT_ROOT / artifact_rel)
        manifest["champions"][str(horizon)] = {
            "horizon": horizon,
            "model_name": model_name,
            "artifact_path": str(artifact_rel).replace("\\", "/"),
            "forecast_mode": "operational",
            "uses_future_weather_forecast": False,
            "feature_columns": feature_columns,
            "target": TARGET,
            "trained_at": pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").isoformat(),
            "train_rows": int(train_rows),
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "R2": float(row["R2"]),
            "recall_unhealthy_plus": float(row["recall_unhealthy_plus"]),
            "mae_top5": float(row["mae_top5"]),
            "selection_rule": manifest["selection_rule"],
        }

    DEFAULT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def write_combined_leaderboard(metrics: pd.DataFrame) -> None:
    """Một bảng tổng hợp DUY NHẤT cho mỗi horizon (mode operational).

    Gộp toàn bộ: Baseline (Naive/Rolling/SeasonalNaive) → SARIMAX (thống kê) →
    ML tuyến tính/cây/boosting → Window/Deep. Sắp theo complexity_rank rồi MAE để
    người đọc thấy mô hình càng cải tiến thì hiệu năng càng tăng.
    """
    ok = metrics[metrics["status"].eq("ok") & metrics["mode"].eq("operational")].copy()
    if ok.empty:
        return
    if "tuned" not in ok.columns:
        ok["tuned"] = False
    ok["tuned"] = ok["tuned"].fillna(False)

    group_order = {
        "corrected_baseline": 0,
        "statistical": 1,
        "tabular_linear": 2,
        "tabular_tree": 3,
        "tabular_boosting": 4,
        "tabular_foundation": 5,
        "sequential_window": 6,
        "neuralforecast": 7,
    }
    ok["_g"] = ok["group"].map(group_order).fillna(99)
    ok = ok.sort_values(["horizon", "_g", "MAE"])

    sections = [
        "# Bảng Tổng Hợp So Sánh Toàn Bộ Mô Hình (Operational)\n",
        "Sắp xếp theo nhóm độ phức tạp tăng dần (Baseline → Thống kê → ML → Chuỗi/Deep), "
        "trong mỗi nhóm xếp theo MAE. Cột `tuned` cho biết tham số có qua TimeSeriesSplit CV hay không.\n",
    ]
    cols = ["group", "model", "tuned", "MAE", "RMSE", "R2", "recall_unhealthy_plus", "mae_top5", "runtime_seconds"]
    for horizon in sorted(ok["horizon"].unique()):
        block = ok[ok["horizon"].eq(horizon)]
        sections.append(f"## Horizon t+{int(horizon)}h\n")
        sections.append(_md_table(block, cols) + "\n")

    (REPORT_DIR / "combined_leaderboard.md").write_text("\n".join(sections), encoding="utf-8")


def write_outputs(metrics: pd.DataFrame, skipped: pd.DataFrame, export_manifest: dict | None = None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False, encoding="utf-8")
    write_combined_leaderboard(metrics)
    metrics[["mode", "horizon", "model", "group", "runtime_seconds", "status", "error"]].to_csv(
        REPORT_DIR / "runtime.csv", index=False, encoding="utf-8"
    )

    ok = metrics[metrics["status"].eq("ok")].copy()
    leaderboard = ok[ok["selected_champion"]].sort_values(["mode", "horizon"])
    final_leaderboard = leaderboard[leaderboard["mode"].eq("operational")]
    (REPORT_DIR / "leaderboard.md").write_text(
        "# Model Selection Leaderboard\n\n"
        "Selection rule: lowest MAE wins; within 1%, choose simpler and faster model.\n\n"
        + _md_table(
            leaderboard,
            ["mode", "horizon", "model", "group", "MAE", "RMSE", "R2", "recall_unhealthy_plus", "mae_top5"],
        )
        + "\n\n## Full Metrics\n\n"
        + _md_table(
            ok,
            [
                "mode",
                "horizon",
                "rank_mae",
                "model",
                "group",
                "MAE",
                "RMSE",
                "R2",
                "runtime_seconds",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    (REPORT_DIR / "final_leaderboard.md").write_text(
        "# Final Operational Leaderboard\n\n"
        "This is the deployment-selection leaderboard. Oracle-weather rows are excluded.\n\n"
        + _md_table(
            final_leaderboard,
            ["mode", "horizon", "model", "group", "MAE", "RMSE", "R2", "recall_unhealthy_plus", "mae_top5"],
        )
        + "\n",
        encoding="utf-8",
    )

    model_cards = {
        "scope": "single-station Hanoi PM2.5 model selection",
        "primary_mode": "operational",
        "oracle_weather_note": "oracle_weather uses actual future weather as Perfect Prognosis upper-bound, not primary leaderboard evidence.",
        "preflight": preflight_environment(),
        "tabpfn_training": "uses the full eligible training window; no benchmark row cap or estimator cap is applied by this project; ignore_pretraining_limits=True is set so TabPFN can attempt datasets above its official 10,000-sample guidance",
        "champion_manifest": export_manifest,
        "sources": {
            "TabPFN": "https://github.com/PriorLabs/TabPFN",
            "NeuralForecast exogenous variables": "https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html",
            "NeuralForecast PatchTST": "https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html",
            "NeuralForecast TFT": "https://nixtlaverse.nixtla.io/neuralforecast/models.tft.html",
        },
        "skipped_optional_models": skipped.to_dict(orient="records") if not skipped.empty else [],
    }
    (REPORT_DIR / "model_cards.md").write_text(
        "# Model Selection Model Cards\n\n"
        "```json\n"
        + json.dumps(model_cards, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )


def _parse_horizons(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PM2.5 model selection pipeline.")
    parser.add_argument("--mode", choices=["operational", "oracle_weather", "both"], default="both")
    parser.add_argument("--horizons", default=None, help="Comma-separated horizons, e.g. 1,24,48,72")
    parser.add_argument("--quick", action="store_true", help="Run cheap smoke benchmark only.")
    parser.add_argument("--no-optional", action="store_true", help="Skip optional CatBoost/TabPFN models.")
    parser.add_argument("--skip-tabpfn", action="store_true", help="Skip TabPFN while still allowing CatBoost.")
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist fitted non-baseline models under models/model_selection.",
    )
    parser.add_argument("--run-neuralforecast", action="store_true", help="Run NeuralForecast models if installed.")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters via TimeSeriesSplit CV on Train+Valid instead of fixed configs.")
    parser.add_argument("--no-sarimax", action="store_true", help="Skip the SARIMAX statistical baseline.")
    parser.add_argument("--export-champions", action="store_true", help="Export operational champion artifacts and manifest.")
    parser.add_argument("--preflight", action="store_true", help="Print optional dependency and hardware preflight JSON, then exit.")
    args = parser.parse_args()

    if args.preflight:
        print(json.dumps(preflight_environment(), indent=2, ensure_ascii=False))
        return

    modes = ["operational", "oracle_weather"] if args.mode == "both" else [args.mode]
    all_metrics = []
    all_skipped = []
    for mode in modes:
        metrics, skipped = run_benchmark(
            mode=mode,
            horizons=_parse_horizons(args.horizons),
            quick=args.quick,
            include_optional=not args.no_optional,
            include_tabpfn=not args.skip_tabpfn,
            save_models=args.save_models,
            run_neuralforecast=args.run_neuralforecast,
            tune=args.tune,
            include_sarimax=not args.no_sarimax,
        )
        all_metrics.append(metrics)
        all_skipped.append(skipped.assign(mode=mode) if not skipped.empty else skipped)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    skipped_df = pd.concat(all_skipped, ignore_index=True)
    manifest = (
        export_champions(
            metrics_df,
            include_optional=not args.no_optional,
            include_tabpfn=not args.skip_tabpfn,
        )
        if args.export_champions
        else None
    )
    write_outputs(metrics_df, skipped_df, export_manifest=manifest)
    print(f"Wrote benchmark artifacts to {REPORT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

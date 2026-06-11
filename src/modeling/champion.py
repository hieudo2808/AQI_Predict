"""Champion model registry and inference helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from src.config import FEATURES, WEATHER_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "champions" / "manifest.json"


@dataclass
class ModelBundle:
    model: Any
    label: str
    source: str
    horizon: int
    feature_columns: list[str]
    metadata: dict
    uses_future_weather_forecast: bool = False


def _bundle(
    *,
    model: Any,
    label: str,
    source: str,
    horizon: int,
    feature_columns: list[str] | None = None,
    metadata: dict | None = None,
    uses_future_weather_forecast: bool = False,
) -> ModelBundle:
    return ModelBundle(
        model=model,
        label=label,
        source=source,
        horizon=horizon,
        feature_columns=list(feature_columns or FEATURES),
        metadata=metadata or {},
        uses_future_weather_forecast=uses_future_weather_forecast,
    )


def load_champion_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict | None:
    path = Path(manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_champion_entry(horizon: int, manifest: dict | None) -> dict | None:
    if not manifest:
        return None
    champions = manifest.get("champions", {})
    return champions.get(str(horizon)) or champions.get(horizon)


def load_model_for_horizon(
    horizon: int,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    root: str | Path = PROJECT_ROOT,
) -> ModelBundle:
    root_path = Path(root)
    manifest = load_champion_manifest(manifest_path)
    entry = get_champion_entry(horizon, manifest)
    if entry:
        artifact_path = root_path / entry["artifact_path"]
        model = joblib.load(artifact_path)
        model_name = entry.get("model_name", "Champion")
        return _bundle(
            model=model,
            label=f"Champion: {model_name}",
            source="champion",
            horizon=horizon,
            feature_columns=entry.get("feature_columns"),
            metadata=entry,
            uses_future_weather_forecast=bool(entry.get("uses_future_weather_forecast", False)),
        )

    xgb_path = root_path / "models" / f"xgb_t{horizon}.json"
    if xgb_path.exists():
        model = xgb.XGBRegressor()
        model.load_model(xgb_path)
        return _bundle(
            model=model,
            label="Legacy XGBoost",
            source="legacy_xgboost",
            horizon=horizon,
            metadata={"artifact_path": str(xgb_path.relative_to(root_path))},
        )

    sarima_path = root_path / "models" / f"sarima_t{horizon}.pkl"
    if sarima_path.exists():
        return _bundle(
            model=joblib.load(sarima_path),
            label="Legacy SARIMA",
            source="legacy_sarima",
            horizon=horizon,
            metadata={"artifact_path": str(sarima_path.relative_to(root_path))},
        )

    return _bundle(
        model=None,
        label="Missing model",
        source="missing",
        horizon=horizon,
    )


def build_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    base_row: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one-row feature input for tabular or sequence-window champions."""
    source_row = base_row.copy() if base_row is not None else df.iloc[[-1]].copy()
    values: dict[str, float] = {}

    for col in feature_columns:
        if col in source_row.columns:
            values[col] = source_row.iloc[0][col]
            continue

        if col.startswith("pm2_5_seq_lag_"):
            lag = int(col.rsplit("_", 1)[-1])
            if len(df) < lag:
                raise ValueError(f"Not enough history to build {col}")
            values[col] = df["pm2_5"].iloc[-lag]
            continue

        raise KeyError(f"Missing feature '{col}' for champion model")

    return pd.DataFrame([values], columns=feature_columns)


def apply_future_weather_if_allowed(
    features: pd.DataFrame,
    weather_row: pd.Series | None,
    uses_future_weather_forecast: bool,
) -> pd.DataFrame:
    if not uses_future_weather_forecast or weather_row is None:
        return features

    updated = features.copy()
    for col in WEATHER_FEATURES:
        if col in updated.columns and col in weather_row.index:
            updated[col] = weather_row[col]
    return updated

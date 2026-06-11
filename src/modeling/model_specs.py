"""Model specifications for champion selection and optional dependency preflight."""
from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_CONFIGS, RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    name: str
    group: str
    factory: Callable[[], object]
    complexity_rank: int
    requires: str | None = None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def preflight_environment() -> dict:
    """Capture optional modeling dependency and hardware availability."""
    modules = {}
    for name in ["catboost", "tabpfn", "torch", "neuralforecast"]:
        available = module_available(name)
        modules[name] = {"available": available, "version": None}
        if available:
            try:
                module = __import__(name)
                modules[name]["version"] = getattr(module, "__version__", "unknown")
            except Exception as exc:  # noqa: BLE001 - preflight should not fail benchmark.
                modules[name]["error"] = str(exc)

    torch_info = {"cuda_available": False, "device_count": 0}
    if modules["torch"]["available"]:
        try:
            import torch

            torch_info = {
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        except Exception as exc:  # noqa: BLE001
            torch_info["error"] = str(exc)

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": modules,
        "torch": torch_info,
    }


def safe_model_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def tabular_specs(include_optional: bool, include_tabpfn: bool = True) -> tuple[list[ModelSpec], list[dict]]:
    specs = [
        ModelSpec(
            "Ridge",
            "tabular_linear",
            lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            complexity_rank=1,
        ),
        ModelSpec(
            "ElasticNet",
            "tabular_linear",
            lambda: make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=RANDOM_SEED),
            ),
            complexity_rank=1,
        ),
        ModelSpec(
            "RandomForest",
            "tabular_tree",
            lambda: RandomForestRegressor(**MODEL_CONFIGS["Random Forest"]),
            complexity_rank=2,
        ),
        ModelSpec(
            "ExtraTrees",
            "tabular_tree",
            lambda: ExtraTreesRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            complexity_rank=2,
        ),
    ]

    import lightgbm as lgb
    import xgboost as xgb

    specs.extend(
        [
            ModelSpec(
                "XGBoost",
                "tabular_boosting",
                lambda: xgb.XGBRegressor(**MODEL_CONFIGS["XGBoost"]),
                complexity_rank=2,
            ),
            ModelSpec(
                "LightGBM",
                "tabular_boosting",
                lambda: lgb.LGBMRegressor(**MODEL_CONFIGS["LightGBM"]),
                complexity_rank=2,
            ),
        ]
    )

    skipped = []
    if include_optional and module_available("catboost"):
        from catboost import CatBoostRegressor

        specs.append(
            ModelSpec(
                "CatBoost",
                "tabular_boosting",
                lambda: CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.03,
                    loss_function="RMSE",
                    random_seed=RANDOM_SEED,
                    verbose=False,
                ),
                complexity_rank=2,
            )
        )
    else:
        skipped.append({"model": "CatBoost", "reason": "catboost is not installed or optional models disabled"})

    if include_optional and include_tabpfn and module_available("tabpfn"):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        specs.append(
            ModelSpec(
                "TabPFNRegressor_v2",
                "tabular_foundation",
                lambda: TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2,
                    ignore_pretraining_limits=True,
                    random_state=RANDOM_SEED,
                    show_progress_bar=False,
                ),
                complexity_rank=3,
                requires="tabpfn",
            )
        )
    else:
        if not include_tabpfn:
            reason = "TabPFN manual run skipped for this benchmark; rerun without --skip-tabpfn when time/GPU is available"
        else:
            reason = "tabpfn is not installed or optional models disabled"
        skipped.append({"model": "TabPFNRegressor_v2", "reason": reason})

    return specs, skipped


def window_sequence_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "WindowRidge_168h",
            "sequential_window",
            lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            complexity_rank=2,
        ),
        ModelSpec(
            "WindowMLP_168h",
            "sequential_window",
            lambda: make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=300,
                    random_state=RANDOM_SEED,
                    early_stopping=True,
                ),
            ),
            complexity_rank=3,
        ),
    ]


def neuralforecast_model_names() -> list[str]:
    return ["DLinear", "NLinear", "LSTM", "GRU", "TCN", "NHITS", "NBEATSx", "TFT", "PatchTST"]


def build_neuralforecast_models(max_horizon: int, quick: bool):
    from neuralforecast.models import DLinear, GRU, LSTM, NBEATSx, NHITS, NLinear, PatchTST, TCN, TFT

    max_steps = 20 if quick else 100
    input_size = 168
    common = {
        "h": max_horizon,
        "input_size": input_size,
        "max_steps": max_steps,
        "random_seed": RANDOM_SEED,
        "enable_progress_bar": False,
        "logger": False,
        "enable_checkpointing": False,
    }
    compact = {
        "batch_size": 32,
        "windows_batch_size": 256,
        "valid_batch_size": 32,
    }

    return [
        DLinear(**common, alias="DLinear"),
        NLinear(**common, alias="NLinear"),
        LSTM(**common, encoder_hidden_size=64, decoder_hidden_size=64, **compact, alias="LSTM"),
        GRU(**common, encoder_hidden_size=64, decoder_hidden_size=64, **compact, alias="GRU"),
        TCN(**common, **compact, alias="TCN"),
        NHITS(**common, **compact, alias="NHITS"),
        NBEATSx(**common, **compact, alias="NBEATSx"),
        TFT(**common, hidden_size=32, **compact, alias="TFT"),
        PatchTST(**common, **compact, alias="PatchTST"),
    ]

# Model Selection Model Cards

```json
{
  "scope": "single-station Hanoi PM2.5 model selection",
  "primary_mode": "operational",
  "oracle_weather_note": "oracle_weather uses actual future weather as Perfect Prognosis upper-bound, not primary leaderboard evidence.",
  "preflight": {
    "python": "3.13.13 (tags/v3.13.13:01104ce, Apr  7 2026, 19:25:48) [MSC v.1944 64 bit (AMD64)]",
    "platform": "Windows-11-10.0.26200-SP0",
    "modules": {
      "catboost": {
        "available": true,
        "version": "1.2.10"
      },
      "tabpfn": {
        "available": true,
        "version": "8.0.6"
      },
      "torch": {
        "available": true,
        "version": "2.12.0+cpu"
      },
      "neuralforecast": {
        "available": true,
        "version": null,
        "error": "module 'pytorch_lightning.utilities' has no attribute 'distributed'"
      }
    },
    "torch": {
      "cuda_available": false,
      "device_count": 0
    }
  },
  "tabpfn_training": "uses the full eligible training window; no benchmark row cap or estimator cap is applied by this project",
  "champion_manifest": {
    "version": 1,
    "created_at": "2026-06-09T10:17:58.409205+07:00",
    "target": "pm2_5",
    "forecast_mode": "operational",
    "selection_rule": "lowest MAE; within 1%, simpler/faster wins",
    "champions": {
      "1": {
        "horizon": 1,
        "model_name": "XGBoost",
        "artifact_path": "models/champions/t1_xgboost.joblib",
        "forecast_mode": "operational",
        "uses_future_weather_forecast": false,
        "feature_columns": [
          "pm2_5_missing_flag",
          "pm2_5",
          "pm10",
          "pm2_5_lag_1",
          "pm2_5_lag_2",
          "pm2_5_lag_3",
          "pm2_5_lag_6",
          "pm2_5_lag_12",
          "pm2_5_lag_24",
          "pm2_5_lag_48",
          "pm2_5_lag_168",
          "pm2_5_roll6_mean",
          "pm2_5_roll24_mean",
          "pm2_5_roll72_mean",
          "pm2_5_roll24_std",
          "pm2_5_roll24_max",
          "hour_sin",
          "hour_cos",
          "dow_sin",
          "dow_cos",
          "month_sin",
          "month_cos",
          "is_weekend",
          "pm2_5_diff_1h",
          "pm2_5_diff_24h",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-09T10:17:59.236633+07:00",
        "train_rows": 21185,
        "MAE": 3.5169,
        "RMSE": 6.3496,
        "R2": 0.908251,
        "recall_unhealthy_plus": 0.945098,
        "mae_top5": 10.7296,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      },
      "24": {
        "horizon": 24,
        "model_name": "Ridge",
        "artifact_path": "models/champions/t24_ridge.joblib",
        "forecast_mode": "operational",
        "uses_future_weather_forecast": false,
        "feature_columns": [
          "pm2_5_missing_flag",
          "pm2_5",
          "pm10",
          "pm2_5_lag_1",
          "pm2_5_lag_2",
          "pm2_5_lag_3",
          "pm2_5_lag_6",
          "pm2_5_lag_12",
          "pm2_5_lag_24",
          "pm2_5_lag_48",
          "pm2_5_lag_168",
          "pm2_5_roll6_mean",
          "pm2_5_roll24_mean",
          "pm2_5_roll72_mean",
          "pm2_5_roll24_std",
          "pm2_5_roll24_max",
          "hour_sin",
          "hour_cos",
          "dow_sin",
          "dow_cos",
          "month_sin",
          "month_cos",
          "is_weekend",
          "pm2_5_diff_1h",
          "pm2_5_diff_24h",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-09T10:17:59.295072+07:00",
        "train_rows": 21162,
        "MAE": 11.2693,
        "RMSE": 15.2216,
        "R2": 0.47288,
        "recall_unhealthy_plus": 0.803361,
        "mae_top5": 29.8364,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      },
      "48": {
        "horizon": 48,
        "model_name": "Ridge",
        "artifact_path": "models/champions/t48_ridge.joblib",
        "forecast_mode": "operational",
        "uses_future_weather_forecast": false,
        "feature_columns": [
          "pm2_5_missing_flag",
          "pm2_5",
          "pm10",
          "pm2_5_lag_1",
          "pm2_5_lag_2",
          "pm2_5_lag_3",
          "pm2_5_lag_6",
          "pm2_5_lag_12",
          "pm2_5_lag_24",
          "pm2_5_lag_48",
          "pm2_5_lag_168",
          "pm2_5_roll6_mean",
          "pm2_5_roll24_mean",
          "pm2_5_roll72_mean",
          "pm2_5_roll24_std",
          "pm2_5_roll24_max",
          "hour_sin",
          "hour_cos",
          "dow_sin",
          "dow_cos",
          "month_sin",
          "month_cos",
          "is_weekend",
          "pm2_5_diff_1h",
          "pm2_5_diff_24h",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-09T10:17:59.346100+07:00",
        "train_rows": 21138,
        "MAE": 12.4674,
        "RMSE": 16.5421,
        "R2": 0.378075,
        "recall_unhealthy_plus": 0.758833,
        "mae_top5": 32.7248,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      },
      "72": {
        "horizon": 72,
        "model_name": "Ridge",
        "artifact_path": "models/champions/t72_ridge.joblib",
        "forecast_mode": "operational",
        "uses_future_weather_forecast": false,
        "feature_columns": [
          "pm2_5_missing_flag",
          "pm2_5",
          "pm10",
          "pm2_5_lag_1",
          "pm2_5_lag_2",
          "pm2_5_lag_3",
          "pm2_5_lag_6",
          "pm2_5_lag_12",
          "pm2_5_lag_24",
          "pm2_5_lag_48",
          "pm2_5_lag_168",
          "pm2_5_roll6_mean",
          "pm2_5_roll24_mean",
          "pm2_5_roll72_mean",
          "pm2_5_roll24_std",
          "pm2_5_roll24_max",
          "hour_sin",
          "hour_cos",
          "dow_sin",
          "dow_cos",
          "month_sin",
          "month_cos",
          "is_weekend",
          "pm2_5_diff_1h",
          "pm2_5_diff_24h",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-09T10:17:59.382345+07:00",
        "train_rows": 21114,
        "MAE": 13.4648,
        "RMSE": 17.8789,
        "R2": 0.274032,
        "recall_unhealthy_plus": 0.747469,
        "mae_top5": 34.2529,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      }
    }
  },
  "sources": {
    "TabPFN": "https://github.com/PriorLabs/TabPFN",
    "NeuralForecast exogenous variables": "https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html",
    "NeuralForecast PatchTST": "https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html",
    "NeuralForecast TFT": "https://nixtlaverse.nixtla.io/neuralforecast/models.tft.html"
  },
  "skipped_optional_models": [
    {
      "model": "DLinear",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "NLinear",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "LSTM",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "GRU",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "TCN",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "NHITS",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "NBEATSx",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "TFT",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "PatchTST",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "operational"
    },
    {
      "model": "DLinear",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "NLinear",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "LSTM",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "GRU",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "TCN",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "NHITS",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "NBEATSx",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "TFT",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    },
    {
      "model": "PatchTST",
      "status": "skipped",
      "reason": "neuralforecast import failed: module 'pytorch_lightning.utilities' has no attribute 'distributed'",
      "mode": "oracle_weather"
    }
  ]
}
```

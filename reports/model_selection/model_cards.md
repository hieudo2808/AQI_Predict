# Model Selection Model Cards

```json
{
  "scope": "single-station Hanoi PM2.5 model selection",
  "primary_mode": "operational",
  "oracle_weather_note": "oracle_weather uses actual future weather as Perfect Prognosis upper-bound, not primary leaderboard evidence.",
  "preflight": {
    "python": "3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]",
    "platform": "Windows-11-10.0.26200-SP0",
    "modules": {
      "catboost": {
        "available": false,
        "version": null
      },
      "tabpfn": {
        "available": false,
        "version": null
      },
      "torch": {
        "available": true,
        "version": "2.9.1+cpu"
      },
      "neuralforecast": {
        "available": false,
        "version": null
      }
    },
    "torch": {
      "cuda_available": false,
      "device_count": 0
    }
  },
  "tabpfn_training": "uses the full eligible training window; no benchmark row cap or estimator cap is applied by this project; ignore_pretraining_limits=True is set so TabPFN can attempt datasets above its official 10,000-sample guidance",
  "champion_manifest": {
    "version": 1,
    "created_at": "2026-06-13T08:47:34.184733+07:00",
    "target": "pm2_5",
    "forecast_mode": "operational",
    "selection_rule": "lowest MAE; within 1%, simpler/faster wins",
    "champions": {
      "1": {
        "horizon": 1,
        "model_name": "Ridge",
        "artifact_path": "models/champions/t1_ridge.joblib",
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
        "trained_at": "2026-06-13T08:47:34.204744+07:00",
        "train_rows": 12496,
        "MAE": 2.8321,
        "RMSE": 4.2043,
        "R2": 0.89889,
        "recall_unhealthy_plus": 0.834933,
        "mae_top5": 5.5394,
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
        "trained_at": "2026-06-13T08:47:34.228253+07:00",
        "train_rows": 12473,
        "MAE": 8.772,
        "RMSE": 11.4072,
        "R2": 0.257059,
        "recall_unhealthy_plus": 0.368522,
        "mae_top5": 21.6993,
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
        "trained_at": "2026-06-13T08:47:34.251252+07:00",
        "train_rows": 12449,
        "MAE": 10.137,
        "RMSE": 12.9524,
        "R2": 0.044015,
        "recall_unhealthy_plus": 0.228407,
        "mae_top5": 24.8672,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      },
      "72": {
        "horizon": 72,
        "model_name": "XGBoost",
        "artifact_path": "models/champions/t72_xgboost.joblib",
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
        "trained_at": "2026-06-13T08:47:35.227376+07:00",
        "train_rows": 12425,
        "MAE": 10.6423,
        "RMSE": 13.1213,
        "R2": 0.020732,
        "recall_unhealthy_plus": 0.216891,
        "mae_top5": 25.419,
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
      "model": "CatBoost",
      "status": "skipped",
      "reason": "catboost is not installed or optional models disabled",
      "mode": "operational"
    },
    {
      "model": "TabPFNRegressor_v2",
      "status": "skipped",
      "reason": "tabpfn is not installed or optional models disabled",
      "mode": "operational"
    },
    {
      "model": "DLinear",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "NLinear",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "LSTM",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "GRU",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "TCN",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "NHITS",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "NBEATSx",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "TFT",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    },
    {
      "model": "PatchTST",
      "status": "skipped",
      "reason": "neuralforecast>=3.1.0/torch stack is not installed, not compatible with this Python env, or not requested for this run",
      "mode": "operational"
    }
  ]
}
```

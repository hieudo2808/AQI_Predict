# Model Selection Model Cards

```json
{
  "scope": "single-station Hanoi PM2.5 model selection",
  "primary_mode": "operational",
  "oracle_weather_note": "oracle_weather uses actual future weather as Perfect Prognosis upper-bound, not primary leaderboard evidence.",
  "preflight": {
    "python": "3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]",
    "platform": "Windows-10-10.0.22631-SP0",
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
        "version": "2.5.1+cu121"
      },
      "neuralforecast": {
        "available": false,
        "version": null
      }
    },
    "torch": {
      "cuda_available": true,
      "device_count": 1
    }
  },
  "tabpfn_training": "uses the full eligible training window; no benchmark row cap or estimator cap is applied by this project; ignore_pretraining_limits=True is set so TabPFN can attempt datasets above its official 10,000-sample guidance",
  "champion_manifest": {
    "version": 1,
    "created_at": "2026-06-13T21:04:14.910868+07:00",
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
          "is_winter",
          "is_summer",
          "season",
          "pm25_pm10_ratio",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-13T21:04:14.954321+07:00",
        "train_rows": 10859,
        "MAE": 6.0777,
        "RMSE": 10.7241,
        "R2": 0.891434,
        "recall_unhealthy_plus": 0.946397,
        "mae_top5": 13.5159,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      },
      "24": {
        "horizon": 24,
        "model_name": "ElasticNet",
        "artifact_path": "models/champions/t24_elasticnet.joblib",
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
          "is_winter",
          "is_summer",
          "season",
          "pm25_pm10_ratio",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-13T21:04:19.287733+07:00",
        "train_rows": 10836,
        "MAE": 18.9246,
        "RMSE": 28.2343,
        "R2": 0.24783,
        "recall_unhealthy_plus": 0.928358,
        "mae_top5": 75.0473,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      },
      "48": {
        "horizon": 48,
        "model_name": "ElasticNet",
        "artifact_path": "models/champions/t48_elasticnet.joblib",
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
          "is_winter",
          "is_summer",
          "season",
          "pm25_pm10_ratio",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-13T21:04:23.489221+07:00",
        "train_rows": 10812,
        "MAE": 21.4891,
        "RMSE": 31.248,
        "R2": 0.080652,
        "recall_unhealthy_plus": 0.943114,
        "mae_top5": 90.1858,
        "selection_rule": "lowest MAE; within 1%, simpler/faster wins"
      },
      "72": {
        "horizon": 72,
        "model_name": "ElasticNet",
        "artifact_path": "models/champions/t72_elasticnet.joblib",
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
          "is_winter",
          "is_summer",
          "season",
          "pm25_pm10_ratio",
          "temperature_2m",
          "relative_humidity_2m",
          "wind_speed_10m",
          "pressure_msl",
          "precipitation"
        ],
        "target": "pm2_5",
        "trained_at": "2026-06-13T21:04:27.446394+07:00",
        "train_rows": 10788,
        "MAE": 23.222,
        "RMSE": 33.1884,
        "R2": -0.034804,
        "recall_unhealthy_plus": 0.953153,
        "mae_top5": 93.771,
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

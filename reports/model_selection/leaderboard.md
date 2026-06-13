# Model Selection Leaderboard

Selection rule: lowest MAE wins; within 1%, choose simpler and faster model.

| mode        | horizon | model            | group              | MAE    | RMSE    | R2       | recall_unhealthy_plus | mae_top5 |
| ----------- | ------- | ---------------- | ------------------ | ------ | ------- | -------- | --------------------- | -------- |
| operational | 1       | Ridge            | tabular_linear     | 2.8321 | 4.2043  | 0.89889  | 0.834933              | 5.5394   |
| operational | 24      | WindowRidge_168h | sequential_window  | 8.5161 | 10.9141 | 0.319903 | 0.414587              | 20.7607  |
| operational | 48      | RollingMean_168h | corrected_baseline | 9.5537 | 12.1962 | 0.152382 | 0.253359              | 25.5802  |
| operational | 72      | RollingMean_168h | corrected_baseline | 9.787  | 12.4141 | 0.123446 | 0.228407              | 25.5454  |

## Full Metrics

| mode        | horizon | rank_mae | model              | group              | MAE     | RMSE    | R2        | runtime_seconds |
| ----------- | ------- | -------- | ------------------ | ------------------ | ------- | ------- | --------- | --------------- |
| operational | 1       | 1.0      | XGBoost            | tabular_boosting   | 2.8191  | 4.1659  | 0.900724  | 31.6009         |
| operational | 1       | 2.0      | Ridge              | tabular_linear     | 2.8321  | 4.2043  | 0.89889   | 2.7108          |
| operational | 1       | 3.0      | ExtraTrees         | tabular_tree       | 2.8476  | 4.223   | 0.897988  | 0.974           |
| operational | 1       | 4.0      | ElasticNet         | tabular_linear     | 2.8535  | 4.241   | 0.897115  | 1.2298          |
| operational | 1       | 5.0      | Persistence        | corrected_baseline | 2.8562  | 4.3913  | 0.889695  | 0.0             |
| operational | 1       | 6.0      | LightGBM           | tabular_boosting   | 2.8846  | 4.2401  | 0.897158  | 19.1123         |
| operational | 1       | 7.0      | RandomForest       | tabular_tree       | 2.8942  | 4.3404  | 0.892237  | 3.8347          |
| operational | 1       | 8.0      | SARIMAX            | statistical        | 2.9022  | 4.2466  | 0.896845  | 4.4068          |
| operational | 1       | 9.0      | WindowRidge_168h   | sequential_window  | 4.4629  | 6.3322  | 0.770637  | 0.4007          |
| operational | 1       | 10.0     | WindowMLP_168h     | sequential_window  | 5.6988  | 7.3473  | 0.6912    | 2.8689          |
| operational | 1       | 11.0     | RollingMean_24h    | corrected_baseline | 6.8492  | 9.0873  | 0.527629  | 0.0             |
| operational | 1       | 12.0     | RollingMean_168h   | corrected_baseline | 8.5823  | 11.0874 | 0.29681   | 0.0             |
| operational | 1       | 13.0     | SeasonalNaive_24h  | corrected_baseline | 9.3716  | 12.5746 | 0.095513  | 0.0             |
| operational | 1       | 14.0     | SeasonalNaive_168h | corrected_baseline | 11.8505 | 15.1605 | -0.314755 | 0.0             |
| operational | 24      | 1.0      | WindowRidge_168h   | sequential_window  | 8.5161  | 10.9141 | 0.319903  | 0.3906          |
| operational | 24      | 2.0      | Ridge              | tabular_linear     | 8.772   | 11.4072 | 0.257059  | 0.2288          |
| operational | 24      | 3.0      | ElasticNet         | tabular_linear     | 8.7897  | 11.1905 | 0.285013  | 1.2886          |
| operational | 24      | 4.0      | XGBoost            | tabular_boosting   | 8.8097  | 11.0719 | 0.300087  | 30.8424         |
| operational | 24      | 5.0      | LightGBM           | tabular_boosting   | 8.9597  | 11.254  | 0.276875  | 12.4852         |
| operational | 24      | 6.0      | SARIMAX            | statistical        | 9.1396  | 11.7218 | 0.215508  | 18.2711         |
| operational | 24      | 7.0      | RollingMean_168h   | corrected_baseline | 9.1869  | 11.8127 | 0.2033    | 0.0             |
| operational | 24      | 8.0      | Persistence        | corrected_baseline | 9.3873  | 12.5868 | 0.095456  | 0.0             |
| operational | 24      | 8.0      | SeasonalNaive_24h  | corrected_baseline | 9.3873  | 12.5868 | 0.095456  | 0.0             |
| operational | 24      | 10.0     | ExtraTrees         | tabular_tree       | 9.5302  | 11.8672 | 0.195931  | 1.0352          |
| operational | 24      | 11.0     | RollingMean_24h    | corrected_baseline | 9.603   | 12.751  | 0.071703  | 0.0             |
| operational | 24      | 12.0     | RandomForest       | tabular_tree       | 9.6588  | 12.1292 | 0.16004   | 3.8156          |
| operational | 24      | 13.0     | WindowMLP_168h     | sequential_window  | 10.4597 | 13.1758 | 0.008825  | 4.3488          |
| operational | 24      | 14.0     | SeasonalNaive_168h | corrected_baseline | 11.8637 | 15.1732 | -0.314479 | 0.0             |
| operational | 48      | 1.0      | RollingMean_168h   | corrected_baseline | 9.5537  | 12.1962 | 0.152382  | 0.0             |
| operational | 48      | 2.0      | WindowRidge_168h   | sequential_window  | 9.563   | 12.1518 | 0.158539  | 0.3868          |
| operational | 48      | 3.0      | Ridge              | tabular_linear     | 10.137  | 12.9524 | 0.044015  | 0.2289          |
| operational | 48      | 4.0      | SARIMAX            | statistical        | 10.1579 | 12.792  | 0.067545  | 18.5423         |
| operational | 48      | 5.0      | ElasticNet         | tabular_linear     | 10.3268 | 12.8523 | 0.058737  | 1.2185          |
| operational | 48      | 6.0      | LightGBM           | tabular_boosting   | 10.3705 | 12.7383 | 0.075361  | 12.7981         |
| operational | 48      | 7.0      | XGBoost            | tabular_boosting   | 10.4092 | 12.7326 | 0.076182  | 31.8988         |
| operational | 48      | 8.0      | RollingMean_24h    | corrected_baseline | 10.8033 | 14.3471 | -0.172944 | 0.0             |
| operational | 48      | 9.0      | ExtraTrees         | tabular_tree       | 10.9949 | 13.4155 | -0.025561 | 1.0208          |
| operational | 48      | 10.0     | RandomForest       | tabular_tree       | 11.5017 | 14.0424 | -0.123651 | 3.782           |
| operational | 48      | 11.0     | Persistence        | corrected_baseline | 11.6605 | 15.4409 | -0.358608 | 0.0             |
| operational | 48      | 12.0     | SeasonalNaive_168h | corrected_baseline | 11.8821 | 15.1878 | -0.314445 | 0.0             |
| operational | 48      | 13.0     | WindowMLP_168h     | sequential_window  | 12.3227 | 15.5476 | -0.377463 | 4.6868          |
| operational | 72      | 1.0      | RollingMean_168h   | corrected_baseline | 9.787   | 12.4141 | 0.123446  | 0.0             |
| operational | 72      | 2.0      | WindowRidge_168h   | sequential_window  | 9.8961  | 12.4065 | 0.124523  | 0.3789          |
| operational | 72      | 3.0      | XGBoost            | tabular_boosting   | 10.6423 | 13.1213 | 0.020732  | 31.5382         |
| operational | 72      | 4.0      | LightGBM           | tabular_boosting   | 10.7067 | 13.2388 | 0.003127  | 12.545          |
| operational | 72      | 5.0      | RollingMean_24h    | corrected_baseline | 10.8687 | 14.4301 | -0.184366 | 0.0             |
| operational | 72      | 6.0      | SARIMAX            | statistical        | 11.1311 | 14.0293 | -0.119481 | 17.3603         |
| operational | 72      | 7.0      | ElasticNet         | tabular_linear     | 11.1372 | 13.7159 | -0.070029 | 1.2773          |
| operational | 72      | 8.0      | Ridge              | tabular_linear     | 11.178  | 14.1818 | -0.143959 | 0.2237          |
| operational | 72      | 9.0      | ExtraTrees         | tabular_tree       | 11.2483 | 13.9794 | -0.111532 | 0.988           |
| operational | 72      | 10.0     | RandomForest       | tabular_tree       | 11.257  | 13.9971 | -0.114343 | 3.8061          |
| operational | 72      | 11.0     | SeasonalNaive_168h | corrected_baseline | 11.8935 | 15.1992 | -0.313983 | 0.0             |
| operational | 72      | 12.0     | Persistence        | corrected_baseline | 11.9054 | 15.8563 | -0.430049 | 0.0             |
| operational | 72      | 13.0     | WindowMLP_168h     | sequential_window  | 12.9817 | 16.5331 | -0.554734 | 7.9973          |

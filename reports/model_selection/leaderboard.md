# Model Selection Leaderboard

Selection rule: lowest MAE wins; within 1%, choose simpler and faster model.

| mode        | horizon | model            | group             | MAE     | RMSE    | R2       | recall_unhealthy_plus | mae_top5 |
| ----------- | ------- | ---------------- | ----------------- | ------- | ------- | -------- | --------------------- | -------- |
| operational | 1       | Ridge            | tabular_linear    | 6.0777  | 10.7241 | 0.891434 | 0.946397              | 13.5159  |
| operational | 24      | ElasticNet       | tabular_linear    | 18.9246 | 28.2343 | 0.24783  | 0.928358              | 75.0473  |
| operational | 48      | WindowRidge_168h | sequential_window | 21.0982 | 28.9469 | 0.21107  | 0.973054              | 68.4414  |
| operational | 72      | WindowRidge_168h | sequential_window | 22.5569 | 30.6277 | 0.118718 | 0.974174              | 72.4808  |

## Full Metrics

| mode        | horizon | rank_mae | model              | group              | MAE     | RMSE    | R2        | runtime_seconds |
| ----------- | ------- | -------- | ------------------ | ------------------ | ------- | ------- | --------- | --------------- |
| operational | 1       | 1.0      | ElasticNet         | tabular_linear     | 6.0196  | 10.7178 | 0.891562  | 5.0581          |
| operational | 1       | 2.0      | SARIMAX            | statistical        | 6.0606  | 10.7524 | 0.890861  | 27.5435         |
| operational | 1       | 3.0      | Ridge              | tabular_linear     | 6.0777  | 10.7241 | 0.891434  | 3.019           |
| operational | 1       | 4.0      | ExtraTrees         | tabular_tree       | 6.16    | 10.8225 | 0.889434  | 0.9074          |
| operational | 1       | 5.0      | XGBoost            | tabular_boosting   | 6.1779  | 10.7974 | 0.889945  | 33.2396         |
| operational | 1       | 6.0      | RandomForest       | tabular_tree       | 6.1782  | 11.0616 | 0.884494  | 4.0603          |
| operational | 1       | 7.0      | LightGBM           | tabular_boosting   | 6.1807  | 10.8943 | 0.887962  | 43.2722         |
| operational | 1       | 8.0      | Persistence        | corrected_baseline | 6.2568  | 11.3591 | 0.878197  | 0.0             |
| operational | 1       | 9.0      | WindowRidge_168h   | sequential_window  | 9.9509  | 16.4666 | 0.744038  | 0.5983          |
| operational | 1       | 10.0     | WindowMLP_168h     | sequential_window  | 11.9564 | 18.1462 | 0.689159  | 4.1501          |
| operational | 1       | 11.0     | RollingMean_24h    | corrected_baseline | 15.4519 | 22.7921 | 0.509618  | 0.0             |
| operational | 1       | 12.0     | RollingMean_168h   | corrected_baseline | 19.4415 | 26.8816 | 0.317851  | 0.0             |
| operational | 1       | 13.0     | SeasonalNaive_24h  | corrected_baseline | 20.9465 | 31.5701 | 0.059153  | 0.0             |
| operational | 1       | 14.0     | SeasonalNaive_168h | corrected_baseline | 28.8193 | 40.9452 | -0.582611 | 0.0             |
| operational | 24      | 1.0      | ElasticNet         | tabular_linear     | 18.9246 | 28.2343 | 0.24783   | 5.398           |
| operational | 24      | 2.0      | RandomForest       | tabular_tree       | 19.065  | 28.7758 | 0.2187    | 4.0648          |
| operational | 24      | 3.0      | XGBoost            | tabular_boosting   | 19.1403 | 28.9527 | 0.209066  | 30.8022         |
| operational | 24      | 4.0      | LightGBM           | tabular_boosting   | 19.1498 | 28.9073 | 0.211545  | 30.5187         |
| operational | 24      | 5.0      | WindowRidge_168h   | sequential_window  | 19.2749 | 27.1485 | 0.304571  | 0.5971          |
| operational | 24      | 6.0      | SARIMAX            | statistical        | 19.2904 | 28.9105 | 0.211367  | 23.8199         |
| operational | 24      | 7.0      | ExtraTrees         | tabular_tree       | 19.2977 | 28.951  | 0.209156  | 0.9348          |
| operational | 24      | 8.0      | Ridge              | tabular_linear     | 20.1016 | 29.6412 | 0.171002  | 0.3158          |
| operational | 24      | 9.0      | RollingMean_168h   | corrected_baseline | 20.8819 | 28.6294 | 0.226633  | 0.0             |
| operational | 24      | 10.0     | Persistence        | corrected_baseline | 20.9493 | 31.5798 | 0.059016  | 0.0             |
| operational | 24      | 10.0     | SeasonalNaive_24h  | corrected_baseline | 20.9493 | 31.5798 | 0.059016  | 0.0             |
| operational | 24      | 12.0     | RollingMean_24h    | corrected_baseline | 21.0357 | 29.5893 | 0.1739    | 0.0             |
| operational | 24      | 13.0     | WindowMLP_168h     | sequential_window  | 24.6879 | 33.1223 | -0.035149 | 6.1937          |
| operational | 24      | 14.0     | SeasonalNaive_168h | corrected_baseline | 28.8307 | 40.9597 | -0.582981 | 0.0             |
| operational | 48      | 1.0      | WindowRidge_168h   | sequential_window  | 21.0982 | 28.9469 | 0.21107   | 0.5776          |
| operational | 48      | 2.0      | ElasticNet         | tabular_linear     | 21.4891 | 31.248  | 0.080652  | 5.4257          |
| operational | 48      | 3.0      | ExtraTrees         | tabular_tree       | 21.8414 | 30.2352 | 0.139281  | 0.912           |
| operational | 48      | 4.0      | RollingMean_168h   | corrected_baseline | 21.9349 | 29.764  | 0.165901  | 0.0             |
| operational | 48      | 5.0      | XGBoost            | tabular_boosting   | 22.0523 | 31.2914 | 0.078101  | 30.0557         |
| operational | 48      | 6.0      | SARIMAX            | statistical        | 22.3285 | 32.6233 | -0.002054 | 25.5149         |
| operational | 48      | 7.0      | LightGBM           | tabular_boosting   | 22.7252 | 31.27   | 0.07936   | 31.0616         |
| operational | 48      | 8.0      | RollingMean_24h    | corrected_baseline | 22.7361 | 31.3119 | 0.076891  | 0.0             |
| operational | 48      | 9.0      | Ridge              | tabular_linear     | 23.1045 | 33.5513 | -0.05987  | 0.3019          |
| operational | 48      | 10.0     | RandomForest       | tabular_tree       | 23.5893 | 32.4131 | 0.010819  | 3.9715          |
| operational | 48      | 11.0     | Persistence        | corrected_baseline | 24.8073 | 35.0138 | -0.154283 | 0.0             |
| operational | 48      | 12.0     | WindowMLP_168h     | sequential_window  | 28.536  | 37.7102 | -0.338914 | 6.1758          |
| operational | 48      | 13.0     | SeasonalNaive_168h | corrected_baseline | 28.8826 | 41.0057 | -0.583155 | 0.0             |
| operational | 72      | 1.0      | WindowRidge_168h   | sequential_window  | 22.5569 | 30.6277 | 0.118718  | 0.6039          |
| operational | 72      | 2.0      | RollingMean_168h   | corrected_baseline | 22.8051 | 30.967  | 0.099089  | 0.0             |
| operational | 72      | 3.0      | ElasticNet         | tabular_linear     | 23.222  | 33.1884 | -0.034804 | 5.0058          |
| operational | 72      | 4.0      | RollingMean_24h    | corrected_baseline | 23.6098 | 33.0169 | -0.024136 | 0.0             |
| operational | 72      | 5.0      | XGBoost            | tabular_boosting   | 23.7152 | 32.9859 | -0.022213 | 31.942          |
| operational | 72      | 6.0      | SARIMAX            | statistical        | 24.2999 | 35.1423 | -0.16023  | 22.7146         |
| operational | 72      | 7.0      | Ridge              | tabular_linear     | 24.8982 | 35.8153 | -0.205096 | 0.3435          |
| operational | 72      | 8.0      | ExtraTrees         | tabular_tree       | 25.3823 | 35.2035 | -0.164275 | 0.916           |
| operational | 72      | 9.0      | Persistence        | corrected_baseline | 25.4339 | 36.0942 | -0.223942 | 0.0             |
| operational | 72      | 10.0     | RandomForest       | tabular_tree       | 27.0097 | 36.3182 | -0.239175 | 3.9996          |
| operational | 72      | 11.0     | LightGBM           | tabular_boosting   | 27.8579 | 37.2862 | -0.306114 | 25.9309         |
| operational | 72      | 12.0     | SeasonalNaive_168h | corrected_baseline | 28.9272 | 41.0504 | -0.583137 | 0.0             |
| operational | 72      | 13.0     | WindowMLP_168h     | sequential_window  | 29.2187 | 38.2783 | -0.376545 | 6.1732          |

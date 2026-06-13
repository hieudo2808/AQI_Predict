# Model Selection Leaderboard

Selection rule: lowest MAE wins; within 1%, choose simpler and faster model.

| mode        | horizon | model            | group              | MAE     | RMSE    | R2       | recall_unhealthy_plus | mae_top5 |
| ----------- | ------- | ---------------- | ------------------ | ------- | ------- | -------- | --------------------- | -------- |
| operational | 1       | ElasticNet       | tabular_linear     | 6.0997  | 10.7776 | 0.889616 | 0.943654              | 13.7713  |
| operational | 24      | ElasticNet       | tabular_linear     | 18.6681 | 27.768  | 0.268573 | 0.939358              | 73.2287  |
| operational | 48      | WindowRidge_168h | sequential_window  | 21.0293 | 28.857  | 0.21013  | 0.970781              | 68.3923  |
| operational | 72      | RollingMean_168h | corrected_baseline | 22.8188 | 30.9879 | 0.091124 | 0.906699              | 66.1922  |

## Full Metrics

| mode        | horizon | rank_mae | model              | group              | MAE     | RMSE    | R2        | runtime_seconds |
| ----------- | ------- | -------- | ------------------ | ------------------ | ------- | ------- | --------- | --------------- |
| operational | 1       | 1.0      | Ridge              | tabular_linear     | 6.0886  | 10.765  | 0.889873  | 5.7164          |
| operational | 1       | 2.0      | ElasticNet         | tabular_linear     | 6.0997  | 10.7776 | 0.889616  | 2.0779          |
| operational | 1       | 3.0      | SARIMAX            | statistical        | 6.1203  | 10.8316 | 0.888506  | 41.6666         |
| operational | 1       | 4.0      | XGBoost            | tabular_boosting   | 6.127   | 10.7684 | 0.889803  | 60.6377         |
| operational | 1       | 5.0      | LightGBM           | tabular_boosting   | 6.1421  | 10.846  | 0.888209  | 48.4792         |
| operational | 1       | 6.0      | ExtraTrees         | tabular_tree       | 6.1799  | 10.8432 | 0.888268  | 1.4234          |
| operational | 1       | 7.0      | RandomForest       | tabular_tree       | 6.1905  | 11.1136 | 0.882626  | 6.3649          |
| operational | 1       | 8.0      | Persistence        | corrected_baseline | 6.2722  | 11.3774 | 0.876986  | 0.0             |
| operational | 1       | 9.0      | WindowRidge_168h   | sequential_window  | 9.961   | 16.4975 | 0.741356  | 0.5982          |
| operational | 1       | 10.0     | WindowMLP_168h     | sequential_window  | 11.8456 | 18.1693 | 0.686283  | 6.694           |
| operational | 1       | 11.0     | RollingMean_24h    | corrected_baseline | 15.4188 | 22.7814 | 0.506797  | 0.0             |
| operational | 1       | 12.0     | RollingMean_168h   | corrected_baseline | 19.4055 | 26.8592 | 0.314431  | 0.0             |
| operational | 1       | 13.0     | SeasonalNaive_24h  | corrected_baseline | 20.9173 | 31.5637 | 0.053242  | 0.0             |
| operational | 1       | 14.0     | SeasonalNaive_168h | corrected_baseline | 28.8372 | 40.9601 | -0.594361 | 0.0             |
| operational | 24      | 1.0      | ElasticNet         | tabular_linear     | 18.6681 | 27.768  | 0.268573  | 1.9551          |
| operational | 24      | 2.0      | LightGBM           | tabular_boosting   | 19.0634 | 28.7958 | 0.213426  | 44.4685         |
| operational | 24      | 3.0      | RandomForest       | tabular_tree       | 19.1239 | 28.6921 | 0.219082  | 6.5508          |
| operational | 24      | 4.0      | WindowRidge_168h   | sequential_window  | 19.2181 | 27.1357 | 0.301505  | 0.5351          |
| operational | 24      | 5.0      | XGBoost            | tabular_boosting   | 19.2631 | 29.0138 | 0.201468  | 63.6344         |
| operational | 24      | 6.0      | SARIMAX            | statistical        | 19.4028 | 28.6571 | 0.220983  | 27.6137         |
| operational | 24      | 7.0      | ExtraTrees         | tabular_tree       | 19.4451 | 28.8463 | 0.210661  | 1.4902          |
| operational | 24      | 8.0      | Ridge              | tabular_linear     | 19.9764 | 29.4135 | 0.179318  | 0.2792          |
| operational | 24      | 9.0      | RollingMean_168h   | corrected_baseline | 20.8304 | 28.5995 | 0.224115  | 0.0             |
| operational | 24      | 10.0     | RollingMean_24h    | corrected_baseline | 20.8702 | 29.3889 | 0.180689  | 0.0             |
| operational | 24      | 11.0     | Persistence        | corrected_baseline | 20.9453 | 31.5912 | 0.053299  | 0.0             |
| operational | 24      | 11.0     | SeasonalNaive_24h  | corrected_baseline | 20.9453 | 31.5912 | 0.053299  | 0.0             |
| operational | 24      | 13.0     | WindowMLP_168h     | sequential_window  | 24.0062 | 32.3505 | 0.007243  | 6.5962          |
| operational | 24      | 14.0     | SeasonalNaive_168h | corrected_baseline | 28.8737 | 40.9955 | -0.59424  | 0.0             |
| operational | 48      | 1.0      | WindowRidge_168h   | sequential_window  | 21.0293 | 28.857  | 0.21013   | 0.4674          |
| operational | 48      | 2.0      | ElasticNet         | tabular_linear     | 21.3178 | 30.581  | 0.112934  | 2.1458          |
| operational | 48      | 3.0      | RollingMean_168h   | corrected_baseline | 21.9305 | 29.7686 | 0.159437  | 0.0             |
| operational | 48      | 4.0      | XGBoost            | tabular_boosting   | 22.1788 | 31.2986 | 0.070814  | 51.9329         |
| operational | 48      | 5.0      | SARIMAX            | statistical        | 22.2578 | 31.9809 | 0.02986   | 25.2811         |
| operational | 48      | 6.0      | LightGBM           | tabular_boosting   | 22.2955 | 31.4466 | 0.062004  | 14.5069         |
| operational | 48      | 7.0      | ExtraTrees         | tabular_tree       | 22.306  | 30.9483 | 0.0915    | 1.3164          |
| operational | 48      | 8.0      | Ridge              | tabular_linear     | 22.7986 | 32.9192 | -0.027899 | 0.2837          |
| operational | 48      | 9.0      | RollingMean_24h    | corrected_baseline | 22.8105 | 31.4615 | 0.061119  | 0.0             |
| operational | 48      | 10.0     | RandomForest       | tabular_tree       | 23.7455 | 32.6382 | -0.010425 | 5.0498          |
| operational | 48      | 11.0     | Persistence        | corrected_baseline | 24.6494 | 34.8485 | -0.151918 | 0.0             |
| operational | 48      | 12.0     | WindowMLP_168h     | sequential_window  | 28.1502 | 37.3132 | -0.320619 | 6.7362          |
| operational | 48      | 13.0     | SeasonalNaive_168h | corrected_baseline | 28.8763 | 41.0125 | -0.595459 | 0.0             |
| operational | 72      | 1.0      | WindowRidge_168h   | sequential_window  | 22.6173 | 30.7122 | 0.107226  | 0.4284          |
| operational | 72      | 2.0      | RollingMean_168h   | corrected_baseline | 22.8188 | 30.9879 | 0.091124  | 0.0             |
| operational | 72      | 3.0      | ElasticNet         | tabular_linear     | 23.2099 | 32.5713 | -0.004128 | 1.4772          |
| operational | 72      | 4.0      | LightGBM           | tabular_boosting   | 23.3558 | 32.5376 | -0.002052 | 11.1755         |
| operational | 72      | 5.0      | RollingMean_24h    | corrected_baseline | 23.909  | 33.6776 | -0.073497 | 0.0             |
| operational | 72      | 6.0      | SARIMAX            | statistical        | 24.2146 | 34.4013 | -0.120132 | 20.9277         |
| operational | 72      | 7.0      | Ridge              | tabular_linear     | 24.5018 | 34.9841 | -0.158403 | 0.229           |
| operational | 72      | 8.0      | Persistence        | corrected_baseline | 25.7186 | 36.6629 | -0.272251 | 0.0             |
| operational | 72      | 9.0      | ExtraTrees         | tabular_tree       | 25.8116 | 35.2185 | -0.173981 | 0.9621          |
| operational | 72      | 10.0     | XGBoost            | tabular_boosting   | 26.063  | 35.0513 | -0.162858 | 48.3077         |
| operational | 72      | 11.0     | RandomForest       | tabular_tree       | 26.7688 | 35.9139 | -0.220797 | 4.2964          |
| operational | 72      | 12.0     | WindowMLP_168h     | sequential_window  | 27.8231 | 36.5463 | -0.26417  | 3.815           |
| operational | 72      | 13.0     | SeasonalNaive_168h | corrected_baseline | 28.9242 | 41.0563 | -0.595431 | 0.0             |

# Bảng Tổng Hợp So Sánh Toàn Bộ Mô Hình (Operational)

Sắp xếp theo nhóm độ phức tạp tăng dần (Baseline → Thống kê → ML → Chuỗi/Deep), trong mỗi nhóm xếp theo MAE. Cột `tuned` cho biết tham số có qua TimeSeriesSplit CV hay không.

## Horizon t+1h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | Persistence        | False | 6.2568  | 11.3591 | 0.878197  | 0.9595                | 13.3191  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 15.4519 | 22.7921 | 0.509618  | 0.914235              | 39.875   | 0.0             |
| corrected_baseline | RollingMean_168h   | False | 19.4415 | 26.8816 | 0.317851  | 0.929124              | 55.8987  | 0.0             |
| corrected_baseline | SeasonalNaive_24h  | False | 20.9465 | 31.5701 | 0.059153  | 0.8648                | 54.2391  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.8193 | 40.9452 | -0.582611 | 0.837403              | 70.3673  | 0.0             |
| statistical        | SARIMAX            | False | 6.0606  | 10.7524 | 0.890861  | 0.951757              | 13.5417  | 27.5435         |
| tabular_linear     | ElasticNet         | True  | 6.0196  | 10.7178 | 0.891562  | 0.952353              | 13.8333  | 5.0581          |
| tabular_linear     | Ridge              | True  | 6.0777  | 10.7241 | 0.891434  | 0.946397              | 13.5159  | 3.019           |
| tabular_tree       | ExtraTrees         | False | 6.16    | 10.8225 | 0.889434  | 0.9595                | 15.505   | 0.9074          |
| tabular_tree       | RandomForest       | False | 6.1782  | 11.0616 | 0.884494  | 0.953544              | 14.6648  | 4.0603          |
| tabular_boosting   | XGBoost            | True  | 6.1779  | 10.7974 | 0.889945  | 0.963669              | 17.8972  | 33.2396         |
| tabular_boosting   | LightGBM           | True  | 6.1807  | 10.8943 | 0.887962  | 0.966647              | 16.272   | 43.2722         |
| sequential_window  | WindowRidge_168h   | True  | 9.9509  | 16.4666 | 0.744038  | 0.941036              | 22.9169  | 0.5983          |
| sequential_window  | WindowMLP_168h     | False | 11.9564 | 18.1462 | 0.689159  | 0.923764              | 25.8125  | 4.1501          |

## Horizon t+24h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 20.8819 | 28.6294 | 0.226633  | 0.92                  | 61.6892  | 0.0             |
| corrected_baseline | Persistence        | False | 20.9493 | 31.5798 | 0.059016  | 0.864478              | 54.2391  | 0.0             |
| corrected_baseline | SeasonalNaive_24h  | False | 20.9493 | 31.5798 | 0.059016  | 0.864478              | 54.2391  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 21.0357 | 29.5893 | 0.1739    | 0.865672              | 60.005   | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.8307 | 40.9597 | -0.582981 | 0.837015              | 70.3673  | 0.0             |
| statistical        | SARIMAX            | False | 19.2904 | 28.9105 | 0.211367  | 0.952836              | 81.9036  | 23.8199         |
| tabular_linear     | ElasticNet         | True  | 18.9246 | 28.2343 | 0.24783   | 0.928358              | 75.0473  | 5.398           |
| tabular_linear     | Ridge              | True  | 20.1016 | 29.6412 | 0.171002  | 0.842388              | 76.5117  | 0.3158          |
| tabular_tree       | RandomForest       | False | 19.065  | 28.7758 | 0.2187    | 0.92                  | 82.5446  | 4.0648          |
| tabular_tree       | ExtraTrees         | False | 19.2977 | 28.951  | 0.209156  | 0.934328              | 84.3208  | 0.9348          |
| tabular_boosting   | XGBoost            | True  | 19.1403 | 28.9527 | 0.209066  | 0.920597              | 84.8915  | 30.8022         |
| tabular_boosting   | LightGBM           | True  | 19.1498 | 28.9073 | 0.211545  | 0.931343              | 84.0939  | 30.5187         |
| sequential_window  | WindowRidge_168h   | True  | 19.2749 | 27.1485 | 0.304571  | 0.945672              | 56.7505  | 0.5971          |
| sequential_window  | WindowMLP_168h     | False | 24.6879 | 33.1223 | -0.035149 | 0.808358              | 57.2698  | 6.1937          |

## Horizon t+48h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 21.9349 | 29.764  | 0.165901  | 0.920359              | 63.7904  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 22.7361 | 31.3119 | 0.076891  | 0.852695              | 65.9223  | 0.0             |
| corrected_baseline | Persistence        | False | 24.8073 | 35.0138 | -0.154283 | 0.835329              | 65.2545  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.8826 | 41.0057 | -0.583155 | 0.836527              | 70.3673  | 0.0             |
| statistical        | SARIMAX            | False | 22.3285 | 32.6233 | -0.002054 | 0.915569              | 93.6777  | 25.5149         |
| tabular_linear     | ElasticNet         | True  | 21.4891 | 31.248  | 0.080652  | 0.943114              | 90.1858  | 5.4257          |
| tabular_linear     | Ridge              | True  | 23.1045 | 33.5513 | -0.05987  | 0.856287              | 96.5321  | 0.3019          |
| tabular_tree       | ExtraTrees         | False | 21.8414 | 30.2352 | 0.139281  | 0.944311              | 81.1227  | 0.912           |
| tabular_tree       | RandomForest       | False | 23.5893 | 32.4131 | 0.010819  | 0.915569              | 83.6103  | 3.9715          |
| tabular_boosting   | XGBoost            | True  | 22.0523 | 31.2914 | 0.078101  | 0.973054              | 89.4809  | 30.0557         |
| tabular_boosting   | LightGBM           | True  | 22.7252 | 31.27   | 0.07936   | 0.944311              | 85.5576  | 31.0616         |
| sequential_window  | WindowRidge_168h   | True  | 21.0982 | 28.9469 | 0.21107   | 0.973054              | 68.4414  | 0.5776          |
| sequential_window  | WindowMLP_168h     | False | 28.536  | 37.7102 | -0.338914 | 0.762275              | 70.5331  | 6.1758          |

## Horizon t+72h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 22.8051 | 30.967  | 0.099089  | 0.906306              | 66.1922  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 23.6098 | 33.0169 | -0.024136 | 0.854054              | 62.4116  | 0.0             |
| corrected_baseline | Persistence        | False | 25.4339 | 36.0942 | -0.223942 | 0.826426              | 60.988   | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.9272 | 41.0504 | -0.583137 | 0.836036              | 70.6472  | 0.0             |
| statistical        | SARIMAX            | False | 24.2999 | 35.1423 | -0.16023  | 0.859459              | 98.9284  | 22.7146         |
| tabular_linear     | ElasticNet         | True  | 23.222  | 33.1884 | -0.034804 | 0.953153              | 93.771   | 5.0058          |
| tabular_linear     | Ridge              | True  | 24.8982 | 35.8153 | -0.205096 | 0.846246              | 100.8322 | 0.3435          |
| tabular_tree       | ExtraTrees         | False | 25.3823 | 35.2035 | -0.164275 | 0.921922              | 85.8985  | 0.916           |
| tabular_tree       | RandomForest       | False | 27.0097 | 36.3182 | -0.239175 | 0.932132              | 84.7076  | 3.9996          |
| tabular_boosting   | XGBoost            | True  | 23.7152 | 32.9859 | -0.022213 | 0.952553              | 88.9974  | 31.942          |
| tabular_boosting   | LightGBM           | True  | 27.8579 | 37.2862 | -0.306114 | 0.871471              | 84.0216  | 25.9309         |
| sequential_window  | WindowRidge_168h   | True  | 22.5569 | 30.6277 | 0.118718  | 0.974174              | 72.4808  | 0.6039          |
| sequential_window  | WindowMLP_168h     | False | 29.2187 | 38.2783 | -0.376545 | 0.768168              | 60.8586  | 6.1732          |

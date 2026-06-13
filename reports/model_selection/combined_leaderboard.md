# Bảng Tổng Hợp So Sánh Toàn Bộ Mô Hình (Operational)

Sắp xếp theo nhóm độ phức tạp tăng dần (Baseline → Thống kê → ML → Chuỗi/Deep), trong mỗi nhóm xếp theo MAE. Cột `tuned` cho biết tham số có qua TimeSeriesSplit CV hay không.

## Horizon t+1h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | Persistence        | False | 6.2722  | 11.3774 | 0.876986  | 0.959668              | 13.3191  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 15.4188 | 22.7814 | 0.506797  | 0.914591              | 39.875   | 0.0             |
| corrected_baseline | RollingMean_168h   | False | 19.4055 | 26.8592 | 0.314431  | 0.929419              | 55.8987  | 0.0             |
| corrected_baseline | SeasonalNaive_24h  | False | 20.9173 | 31.5637 | 0.053242  | 0.865362              | 54.2391  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.8372 | 40.9601 | -0.594361 | 0.838078              | 70.3673  | 0.0             |
| statistical        | SARIMAX            | False | 6.1203  | 10.8316 | 0.888506  | 0.947805              | 13.9785  | 41.6666         |
| tabular_linear     | Ridge              | True  | 6.0886  | 10.765  | 0.889873  | 0.945433              | 13.5838  | 5.7164          |
| tabular_linear     | ElasticNet         | True  | 6.0997  | 10.7776 | 0.889616  | 0.943654              | 13.7713  | 2.0779          |
| tabular_tree       | ExtraTrees         | False | 6.1799  | 10.8432 | 0.888268  | 0.960854              | 15.7762  | 1.4234          |
| tabular_tree       | RandomForest       | False | 6.1905  | 11.1136 | 0.882626  | 0.955516              | 14.7774  | 6.3649          |
| tabular_boosting   | XGBoost            | True  | 6.127   | 10.7684 | 0.889803  | 0.958482              | 16.9817  | 60.6377         |
| tabular_boosting   | LightGBM           | True  | 6.1421  | 10.846  | 0.888209  | 0.96382               | 15.8995  | 48.4792         |
| sequential_window  | WindowRidge_168h   | True  | 9.961   | 16.4975 | 0.741356  | 0.941281              | 22.8274  | 0.5982          |
| sequential_window  | WindowMLP_168h     | False | 11.8456 | 18.1693 | 0.686283  | 0.927046              | 25.063   | 6.694           |

## Horizon t+24h

| group              | model              | tuned | MAE     | RMSE    | R2       | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | -------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 20.8304 | 28.5995 | 0.224115 | 0.920333              | 61.6892  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 20.8702 | 29.3889 | 0.180689 | 0.866231              | 60.005   | 0.0             |
| corrected_baseline | Persistence        | False | 20.9453 | 31.5912 | 0.053299 | 0.865042              | 54.2391  | 0.0             |
| corrected_baseline | SeasonalNaive_24h  | False | 20.9453 | 31.5912 | 0.053299 | 0.865042              | 54.2391  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.8737 | 40.9955 | -0.59424 | 0.837693              | 70.3673  | 0.0             |
| statistical        | SARIMAX            | False | 19.4028 | 28.6571 | 0.220983 | 0.953627              | 80.08    | 27.6137         |
| tabular_linear     | ElasticNet         | True  | 18.6681 | 27.768  | 0.268573 | 0.939358              | 73.2287  | 1.9551          |
| tabular_linear     | Ridge              | True  | 19.9764 | 29.4135 | 0.179318 | 0.847206              | 75.1203  | 0.2792          |
| tabular_tree       | RandomForest       | False | 19.1239 | 28.6921 | 0.219082 | 0.925684              | 81.3288  | 6.5508          |
| tabular_tree       | ExtraTrees         | False | 19.4451 | 28.8463 | 0.210661 | 0.941141              | 82.5275  | 1.4902          |
| tabular_boosting   | LightGBM           | True  | 19.0634 | 28.7958 | 0.213426 | 0.933413              | 83.0404  | 44.4685         |
| tabular_boosting   | XGBoost            | True  | 19.2631 | 29.0138 | 0.201468 | 0.920333              | 84.5433  | 63.6344         |
| sequential_window  | WindowRidge_168h   | True  | 19.2181 | 27.1357 | 0.301505 | 0.945898              | 56.8028  | 0.5351          |
| sequential_window  | WindowMLP_168h     | False | 24.0062 | 32.3505 | 0.007243 | 0.809156              | 53.0498  | 6.5962          |

## Horizon t+48h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 21.9305 | 29.7686 | 0.159437  | 0.920692              | 63.7904  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 22.8105 | 31.4615 | 0.061119  | 0.853309              | 65.9223  | 0.0             |
| corrected_baseline | Persistence        | False | 24.6494 | 34.8485 | -0.151918 | 0.836017              | 65.2545  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.8763 | 41.0125 | -0.595459 | 0.837209              | 70.3673  | 0.0             |
| statistical        | SARIMAX            | False | 22.2578 | 31.9809 | 0.02986   | 0.933214              | 90.0787  | 25.2811         |
| tabular_linear     | ElasticNet         | True  | 21.3178 | 30.581  | 0.112934  | 0.954085              | 86.7446  | 2.1458          |
| tabular_linear     | Ridge              | True  | 22.7986 | 32.9192 | -0.027899 | 0.884914              | 93.2213  | 0.2837          |
| tabular_tree       | ExtraTrees         | False | 22.306  | 30.9483 | 0.0915    | 0.957662              | 85.1381  | 1.3164          |
| tabular_tree       | RandomForest       | False | 23.7455 | 32.6382 | -0.010425 | 0.914729              | 84.7742  | 5.0498          |
| tabular_boosting   | XGBoost            | True  | 22.1788 | 31.2986 | 0.070814  | 0.97257               | 88.2789  | 51.9329         |
| tabular_boosting   | LightGBM           | True  | 22.2955 | 31.4466 | 0.062004  | 0.955874              | 88.3754  | 14.5069         |
| sequential_window  | WindowRidge_168h   | True  | 21.0293 | 28.857  | 0.21013   | 0.970781              | 68.3923  | 0.4674          |
| sequential_window  | WindowMLP_168h     | False | 28.1502 | 37.3132 | -0.320619 | 0.763268              | 67.7452  | 6.7362          |

## Horizon t+72h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 22.8188 | 30.9879 | 0.091124  | 0.906699              | 66.1922  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 23.909  | 33.6776 | -0.073497 | 0.854665              | 62.4116  | 0.0             |
| corrected_baseline | Persistence        | False | 25.7186 | 36.6629 | -0.272251 | 0.827153              | 60.988   | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 28.9242 | 41.0563 | -0.595431 | 0.836722              | 70.6472  | 0.0             |
| statistical        | SARIMAX            | False | 24.2146 | 34.4013 | -0.120132 | 0.912081              | 93.9326  | 20.9277         |
| tabular_linear     | ElasticNet         | True  | 23.2099 | 32.5713 | -0.004128 | 0.965311              | 89.2998  | 1.4772          |
| tabular_linear     | Ridge              | True  | 24.5018 | 34.9841 | -0.158403 | 0.881579              | 95.796   | 0.229           |
| tabular_tree       | ExtraTrees         | False | 25.8116 | 35.2185 | -0.173981 | 0.940191              | 89.514   | 0.9621          |
| tabular_tree       | RandomForest       | False | 26.7688 | 35.9139 | -0.220797 | 0.932416              | 86.1542  | 4.2964          |
| tabular_boosting   | LightGBM           | True  | 23.3558 | 32.5376 | -0.002052 | 0.97189               | 88.2989  | 11.1755         |
| tabular_boosting   | XGBoost            | True  | 26.063  | 35.0513 | -0.162858 | 0.931818              | 85.4921  | 48.3077         |
| sequential_window  | WindowRidge_168h   | True  | 22.6173 | 30.7122 | 0.107226  | 0.972488              | 72.3466  | 0.4284          |
| sequential_window  | WindowMLP_168h     | False | 27.8231 | 36.5463 | -0.26417  | 0.822967              | 63.8604  | 3.815           |

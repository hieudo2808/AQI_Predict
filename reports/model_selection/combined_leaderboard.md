# Bảng Tổng Hợp So Sánh Toàn Bộ Mô Hình (Operational)

Sắp xếp theo nhóm độ phức tạp tăng dần (Baseline → Thống kê → ML → Chuỗi/Deep), trong mỗi nhóm xếp theo MAE. Cột `tuned` cho biết tham số có qua TimeSeriesSplit CV hay không.

## Horizon t+1h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | Persistence        | False | 2.8562  | 4.3913  | 0.889695  | 0.850288              | 5.5872   | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 6.8492  | 9.0873  | 0.527629  | 0.568138              | 13.635   | 0.0             |
| corrected_baseline | RollingMean_168h   | False | 8.5823  | 11.0874 | 0.29681   | 0.318618              | 23.2899  | 0.0             |
| corrected_baseline | SeasonalNaive_24h  | False | 9.3716  | 12.5746 | 0.095513  | 0.522073              | 19.5528  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 11.8505 | 15.1605 | -0.314755 | 0.424184              | 22.9208  | 0.0             |
| statistical        | SARIMAX            | False | 2.9022  | 4.2466  | 0.896845  | 0.802303              | 5.6125   | 4.4068          |
| tabular_linear     | Ridge              | True  | 2.8321  | 4.2043  | 0.89889   | 0.834933              | 5.5394   | 2.7108          |
| tabular_linear     | ElasticNet         | True  | 2.8535  | 4.241   | 0.897115  | 0.840691              | 5.6236   | 1.2298          |
| tabular_tree       | ExtraTrees         | False | 2.8476  | 4.223   | 0.897988  | 0.873321              | 5.3243   | 0.974           |
| tabular_tree       | RandomForest       | False | 2.8942  | 4.3404  | 0.892237  | 0.871401              | 5.6862   | 3.8347          |
| tabular_boosting   | XGBoost            | True  | 2.8191  | 4.1659  | 0.900724  | 0.869482              | 5.3792   | 31.6009         |
| tabular_boosting   | LightGBM           | True  | 2.8846  | 4.2401  | 0.897158  | 0.873321              | 5.5908   | 19.1123         |
| sequential_window  | WindowRidge_168h   | True  | 4.4629  | 6.3322  | 0.770637  | 0.710173              | 8.7299   | 0.4007          |
| sequential_window  | WindowMLP_168h     | False | 5.6988  | 7.3473  | 0.6912    | 0.589251              | 8.4832   | 2.8689          |

## Horizon t+24h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 9.1869  | 11.8127 | 0.2033    | 0.278311              | 25.0407  | 0.0             |
| corrected_baseline | Persistence        | False | 9.3873  | 12.5868 | 0.095456  | 0.522073              | 19.5528  | 0.0             |
| corrected_baseline | SeasonalNaive_24h  | False | 9.3873  | 12.5868 | 0.095456  | 0.522073              | 19.5528  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 9.603   | 12.751  | 0.071703  | 0.439539              | 25.082   | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 11.8637 | 15.1732 | -0.314479 | 0.424184              | 22.9208  | 0.0             |
| statistical        | SARIMAX            | False | 9.1396  | 11.7218 | 0.215508  | 0.272553              | 24.761   | 18.2711         |
| tabular_linear     | Ridge              | True  | 8.772   | 11.4072 | 0.257059  | 0.368522              | 21.6993  | 0.2288          |
| tabular_linear     | ElasticNet         | True  | 8.7897  | 11.1905 | 0.285013  | 0.355086              | 21.8989  | 1.2886          |
| tabular_tree       | ExtraTrees         | False | 9.5302  | 11.8672 | 0.195931  | 0.403071              | 21.7208  | 1.0352          |
| tabular_tree       | RandomForest       | False | 9.6588  | 12.1292 | 0.16004   | 0.358925              | 21.8759  | 3.8156          |
| tabular_boosting   | XGBoost            | True  | 8.8097  | 11.0719 | 0.300087  | 0.357006              | 20.2664  | 30.8424         |
| tabular_boosting   | LightGBM           | True  | 8.9597  | 11.254  | 0.276875  | 0.380038              | 19.8656  | 12.4852         |
| sequential_window  | WindowRidge_168h   | True  | 8.5161  | 10.9141 | 0.319903  | 0.414587              | 20.7607  | 0.3906          |
| sequential_window  | WindowMLP_168h     | False | 10.4597 | 13.1758 | 0.008825  | 0.366603              | 21.0859  | 4.3488          |

## Horizon t+48h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 9.5537  | 12.1962 | 0.152382  | 0.253359              | 25.5802  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 10.8033 | 14.3471 | -0.172944 | 0.357006              | 30.8947  | 0.0             |
| corrected_baseline | Persistence        | False | 11.6605 | 15.4409 | -0.358608 | 0.399232              | 30.1288  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 11.8821 | 15.1878 | -0.314445 | 0.424184              | 22.9208  | 0.0             |
| statistical        | SARIMAX            | False | 10.1579 | 12.792  | 0.067545  | 0.228407              | 26.2945  | 18.5423         |
| tabular_linear     | Ridge              | True  | 10.137  | 12.9524 | 0.044015  | 0.228407              | 24.8672  | 0.2289          |
| tabular_linear     | ElasticNet         | True  | 10.3268 | 12.8523 | 0.058737  | 0.214971              | 25.3672  | 1.2185          |
| tabular_tree       | ExtraTrees         | False | 10.9949 | 13.4155 | -0.025561 | 0.34357               | 22.4235  | 1.0208          |
| tabular_tree       | RandomForest       | False | 11.5017 | 14.0424 | -0.123651 | 0.454894              | 23.0878  | 3.782           |
| tabular_boosting   | LightGBM           | True  | 10.3705 | 12.7383 | 0.075361  | 0.25144               | 24.8473  | 12.7981         |
| tabular_boosting   | XGBoost            | True  | 10.4092 | 12.7326 | 0.076182  | 0.247601              | 24.3287  | 31.8988         |
| sequential_window  | WindowRidge_168h   | True  | 9.563   | 12.1518 | 0.158539  | 0.291747              | 26.0431  | 0.3868          |
| sequential_window  | WindowMLP_168h     | False | 12.3227 | 15.5476 | -0.377463 | 0.332054              | 28.4463  | 4.6868          |

## Horizon t+72h

| group              | model              | tuned | MAE     | RMSE    | R2        | recall_unhealthy_plus | mae_top5 | runtime_seconds |
| ------------------ | ------------------ | ----- | ------- | ------- | --------- | --------------------- | -------- | --------------- |
| corrected_baseline | RollingMean_168h   | False | 9.787   | 12.4141 | 0.123446  | 0.228407              | 25.5454  | 0.0             |
| corrected_baseline | RollingMean_24h    | False | 10.8687 | 14.4301 | -0.184366 | 0.328215              | 27.1707  | 0.0             |
| corrected_baseline | SeasonalNaive_168h | False | 11.8935 | 15.1992 | -0.313983 | 0.424184              | 22.9208  | 0.0             |
| corrected_baseline | Persistence        | False | 11.9054 | 15.8563 | -0.430049 | 0.347409              | 28.9535  | 0.0             |
| statistical        | SARIMAX            | False | 11.1311 | 14.0293 | -0.119481 | 0.174664              | 28.4169  | 17.3603         |
| tabular_linear     | ElasticNet         | True  | 11.1372 | 13.7159 | -0.070029 | 0.190019              | 26.4995  | 1.2773          |
| tabular_linear     | Ridge              | True  | 11.178  | 14.1818 | -0.143959 | 0.191939              | 28.2969  | 0.2237          |
| tabular_tree       | ExtraTrees         | False | 11.2483 | 13.9794 | -0.111532 | 0.333973              | 24.5294  | 0.988           |
| tabular_tree       | RandomForest       | False | 11.257  | 13.9971 | -0.114343 | 0.318618              | 25.121   | 3.8061          |
| tabular_boosting   | XGBoost            | True  | 10.6423 | 13.1213 | 0.020732  | 0.216891              | 25.419   | 31.5382         |
| tabular_boosting   | LightGBM           | True  | 10.7067 | 13.2388 | 0.003127  | 0.213052              | 25.4863  | 12.545          |
| sequential_window  | WindowRidge_168h   | True  | 9.8961  | 12.4065 | 0.124523  | 0.205374              | 26.0387  | 0.3789          |
| sequential_window  | WindowMLP_168h     | False | 12.9817 | 16.5331 | -0.554734 | 0.241843              | 29.2734  | 7.9973          |

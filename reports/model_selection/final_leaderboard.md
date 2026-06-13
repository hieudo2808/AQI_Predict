# Final Operational Leaderboard

This is the deployment-selection leaderboard. Oracle-weather rows are excluded.

| mode        | horizon | model            | group              | MAE     | RMSE    | R2       | recall_unhealthy_plus | mae_top5 |
| ----------- | ------- | ---------------- | ------------------ | ------- | ------- | -------- | --------------------- | -------- |
| operational | 1       | ElasticNet       | tabular_linear     | 6.0997  | 10.7776 | 0.889616 | 0.943654              | 13.7713  |
| operational | 24      | ElasticNet       | tabular_linear     | 18.6681 | 27.768  | 0.268573 | 0.939358              | 73.2287  |
| operational | 48      | WindowRidge_168h | sequential_window  | 21.0293 | 28.857  | 0.21013  | 0.970781              | 68.3923  |
| operational | 72      | RollingMean_168h | corrected_baseline | 22.8188 | 30.9879 | 0.091124 | 0.906699              | 66.1922  |

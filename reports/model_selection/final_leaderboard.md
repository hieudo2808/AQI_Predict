# Final Operational Leaderboard

This is the deployment-selection leaderboard. Oracle-weather rows are excluded.

| mode        | horizon | model            | group              | MAE    | RMSE    | R2       | recall_unhealthy_plus | mae_top5 |
| ----------- | ------- | ---------------- | ------------------ | ------ | ------- | -------- | --------------------- | -------- |
| operational | 1       | Ridge            | tabular_linear     | 2.8321 | 4.2043  | 0.89889  | 0.834933              | 5.5394   |
| operational | 24      | WindowRidge_168h | sequential_window  | 8.5161 | 10.9141 | 0.319903 | 0.414587              | 20.7607  |
| operational | 48      | RollingMean_168h | corrected_baseline | 9.5537 | 12.1962 | 0.152382 | 0.253359              | 25.5802  |
| operational | 72      | RollingMean_168h | corrected_baseline | 9.787  | 12.4141 | 0.123446 | 0.228407              | 25.5454  |

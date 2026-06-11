# Final Operational Leaderboard

This is the deployment-selection leaderboard. Oracle-weather rows are excluded.

| mode        | horizon | model   | group            | MAE     | RMSE    | R2       | recall_unhealthy_plus | mae_top5 |
| ----------- | ------- | ------- | ---------------- | ------- | ------- | -------- | --------------------- | -------- |
| operational | 1       | XGBoost | tabular_boosting | 3.5169  | 6.3496  | 0.908251 | 0.945098              | 10.7296  |
| operational | 24      | Ridge   | tabular_linear   | 11.2693 | 15.2216 | 0.47288  | 0.803361              | 29.8364  |
| operational | 48      | Ridge   | tabular_linear   | 12.4674 | 16.5421 | 0.378075 | 0.758833              | 32.7248  |
| operational | 72      | Ridge   | tabular_linear   | 13.4648 | 17.8789 | 0.274032 | 0.747469              | 34.2529  |

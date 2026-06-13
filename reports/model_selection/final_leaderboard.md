# Final Operational Leaderboard

This is the deployment-selection leaderboard. Oracle-weather rows are excluded.

| mode        | horizon | model            | group             | MAE     | RMSE    | R2       | recall_unhealthy_plus | mae_top5 |
| ----------- | ------- | ---------------- | ----------------- | ------- | ------- | -------- | --------------------- | -------- |
| operational | 1       | Ridge            | tabular_linear    | 6.0777  | 10.7241 | 0.891434 | 0.946397              | 13.5159  |
| operational | 24      | ElasticNet       | tabular_linear    | 18.9246 | 28.2343 | 0.24783  | 0.928358              | 75.0473  |
| operational | 48      | WindowRidge_168h | sequential_window | 21.0982 | 28.9469 | 0.21107  | 0.973054              | 68.4414  |
| operational | 72      | WindowRidge_168h | sequential_window | 22.5569 | 30.6277 | 0.118718 | 0.974174              | 72.4808  |

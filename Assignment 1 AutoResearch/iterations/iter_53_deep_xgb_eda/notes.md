# Iteration 53: deep_xgb_eda

## Hypothesis
Deeper XGBoost grid: max_depth=7,10 + n_estimators=300,500 may find better splits.

## Change
Extended XGBoost grid. Deeper trees, more estimators.

## Config (non-default parameters)
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- split_method = leave_patients_out

# Iteration 25: extended_xgb

## Hypothesis
Larger XGBoost grid: more estimators (300, 500), deeper trees (7), lower learning rate (0.01). With 1610 training samples, a more complex model may fit better.

## Change
XGBoost with extended hyperparameter grid (deeper, more trees).

## Config (non-default parameters)
- split_method = leave_patients_out

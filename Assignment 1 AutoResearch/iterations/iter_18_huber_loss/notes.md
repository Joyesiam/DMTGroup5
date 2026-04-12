# Iteration 18: XGBoost Regression (Huber-like loss)

## Hypothesis
XGBoost for regression (instead of GB) with default squared error may perform
differently. XGBoost has better regularization (reg_alpha, reg_lambda).

## Change
- tabular_reg="xgboost" (was "gb")
- Same features and cleaning as iter_07

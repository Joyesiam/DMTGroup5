# Iteration 139: Optuna Bayesian Search for XGBoost
**Category: Modeling**

## Source
emmaarussi repo

## Hypothesis
Bayesian hyperparameter optimization will find better XGBoost configurations than manual tuning or grid search, especially given the large hyperparameter space.

## Change
Use Optuna with 50 trials to search over XGBoost hyperparameters (max_depth, learning_rate, n_estimators, subsample, colsample_bytree, etc.).

## Implementation
- `optuna.create_study(direction='minimize')` on validation RMSE
- 50 Bayesian trials with TPE sampler
- Log best params and trial history for reproducibility

Run via: `python scripts/run_v6_iterations.py --only 139`

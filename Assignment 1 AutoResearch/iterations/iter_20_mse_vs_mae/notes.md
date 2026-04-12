# Iteration 20: MSE vs MAE Regression Comparison (Task 5B)

## Purpose
Task 5B (6 points) asks to apply MSE and MAE to the regression task and
describe how the model behaves under different metric characteristics.

## What this iteration does
- Train TWO GB regressors: one with loss='squared_error' (MSE), one with loss='absolute_error' (MAE)
- Compare predictions, residuals, behavior on outliers
- Use the best split (leave-patients-out) for reliable comparison

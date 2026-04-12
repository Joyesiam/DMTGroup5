# Iteration 38: log_5lags_confirm

## Hypothesis
Combine best: log_transform + 5 lags + leave-patients-out + extended XGB grid.

## Change
Best of everything: log + 5 lags + extended XGB + leave-patients-out.

## Config (non-default parameters)
- log_transform_before_agg = True
- n_lags = 5
- split_method = leave_patients_out

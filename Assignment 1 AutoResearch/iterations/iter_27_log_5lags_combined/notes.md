# Iteration 27: log_5lags_combined

## Hypothesis
Combine volatility + interactions + log-transform + 5 lags. All small improvements may compound.

## Change
Combined: log_transform + 5 lags + volatility + interactions. Everything that helped or was neutral.

## Config (non-default parameters)
- log_transform_before_agg = True
- n_lags = 5
- split_method = leave_patients_out

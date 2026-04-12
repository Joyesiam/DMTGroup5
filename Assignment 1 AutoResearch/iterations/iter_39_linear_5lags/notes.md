# Iteration 39: linear_5lags

## Hypothesis
Linear interpolation + leave-patients-out was good for GRU (iter_23). Try linear for everything.

## Change
Linear interp + leave-patients-out + all best features. Optimize for temporal.

## Config (non-default parameters)
- imputation_method = linear
- n_lags = 5
- split_method = leave_patients_out

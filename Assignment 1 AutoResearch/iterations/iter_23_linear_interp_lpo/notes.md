# Iteration 23: linear_interp_lpo

## Hypothesis
Linear interpolation helped GRU (iter_08). Combine with leave-patients-out split (iter_15) to see if both improvements stack.

## Change
linear interp + leave-patients-out. Combining best cleaning for temporal + best split.

## Config (non-default parameters)
- imputation_method = linear
- split_method = leave_patients_out

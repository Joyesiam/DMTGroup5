# Iteration 24: log_transform_lpo

## Hypothesis
Log-transform durations (iter_12) was comparable. Try it with leave-patients-out to see if the larger test set reveals an improvement.

## Change
log_transform_before_agg + leave-patients-out.

## Config (non-default parameters)
- log_transform_before_agg = True
- split_method = leave_patients_out

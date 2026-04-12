# Iteration 30: ten_holdout

## Hypothesis
Use 10 holdout patients instead of 5. More patients in test = more robust estimate, fewer in train may hurt.

## Change
n_holdout_patients=10. Larger test set, smaller train set trade-off.

## Config (non-default parameters)
- split_method = leave_patients_out
- n_holdout_patients = 10

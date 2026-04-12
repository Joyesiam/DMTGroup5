# Iteration 59: seven_holdout

## Hypothesis
Best EDA combo with 7 holdout patients. More test data for robustness.

## Change
Best EDA combo but n_holdout_patients=7. Larger test set.

## Config (non-default parameters)
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- split_method = leave_patients_out
- n_holdout_patients = 7

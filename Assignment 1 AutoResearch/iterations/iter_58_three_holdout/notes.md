# Iteration 58: three_holdout

## Hypothesis
Best EDA combo (iter_52 config) with 3 holdout patients. Smaller holdout = more training data.

## Change
Best EDA combo but n_holdout_patients=3. More training data.

## Config (non-default parameters)
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- split_method = leave_patients_out
- n_holdout_patients = 3

# Iteration 60: seed123

## Hypothesis
Best overall config from 43-59, seed=123 (different holdout patients).

## Change
Best config, seed=123. Different patient holdout for cross-validation.

## Config (non-default parameters)
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- split_method = leave_patients_out
- seed = 123

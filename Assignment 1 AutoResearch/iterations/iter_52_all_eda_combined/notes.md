# Iteration 52: all_eda_combined

## Hypothesis
Combine ALL EDA-driven features: morning/evening + drop sparse + lagged valence + momentum.

## Change
Combined: morning_evening + drop_sparse + lagged_valence + momentum. All EDA features.

## Config (non-default parameters)
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- split_method = leave_patients_out

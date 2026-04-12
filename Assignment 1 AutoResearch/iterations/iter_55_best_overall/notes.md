# Iteration 55: best_overall

## Hypothesis
Same as 54 but with linear interpolation (helps GRU).

## Change
All EDA features + linear interpolation. Optimized for temporal model.

## Config (non-default parameters)
- imputation_method = linear
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- include_mood_cluster = True
- include_study_day = True
- include_weekend_distance = True
- n_lags = 5
- split_method = leave_patients_out

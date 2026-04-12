# Iteration 54: max_features

## Hypothesis
Best EDA features + log transform + 5 lags. Combining everything that helped individually.

## Change
EDA features + log_transform + n_lags=5. Maximum feature combination.

## Config (non-default parameters)
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- include_mood_cluster = True
- include_study_day = True
- include_weekend_distance = True
- log_transform_before_agg = True
- n_lags = 5
- split_method = leave_patients_out

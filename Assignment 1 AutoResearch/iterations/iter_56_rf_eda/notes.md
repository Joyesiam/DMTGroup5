# Iteration 56: rf_eda

## Hypothesis
RF instead of XGBoost with all EDA features. RF may benefit from larger feature space.

## Change
tabular_cls='rf' with all EDA features. Random Forest comparison.

## Config (non-default parameters)
- tabular_cls = rf
- add_morning_evening = True
- drop_sparse = True
- include_lagged_valence = True
- include_momentum = True
- split_method = leave_patients_out

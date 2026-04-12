# Iteration 45: lagged_valence

## Hypothesis
Explicit lagged valence (r=0.284) is the 2nd best predictor after mood itself.

## Change
include_lagged_valence=True. Adds valence_lag1, valence_lag2, activity_lag1, activity_lag2.

## Config (non-default parameters)
- include_lagged_valence = True
- split_method = leave_patients_out

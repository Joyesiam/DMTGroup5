# Iteration 34: weighted_xgb

## Hypothesis
XGBoost with class weights adjusted. The 3 classes may be imbalanced after tercile split due to different holdout patients.

## Change
XGBoost with sample_weight based on class frequency. Balanced classes.

## Config (non-default parameters)
- split_method = leave_patients_out

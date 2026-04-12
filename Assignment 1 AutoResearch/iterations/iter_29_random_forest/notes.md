# Iteration 29: random_forest

## Hypothesis
Random Forest as alternative tabular classifier. RF may capture different patterns than XGB. Compare on leave-patients-out.

## Change
tabular_cls='rf'. Random Forest comparison on leave-patients-out.

## Config (non-default parameters)
- tabular_cls = rf
- split_method = leave_patients_out

# Iteration 36: exp_weighted

## Hypothesis
Exponentially weighted features. Recent days weighted more than older days in the window.

## Change
Exponential weighting in rolling window (decay=0.9). Recent days matter more.

## Config (non-default parameters)
- split_method = leave_patients_out

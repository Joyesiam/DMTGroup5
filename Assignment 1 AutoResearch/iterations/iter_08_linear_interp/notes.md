# Iteration 8: Linear Interpolation (Task 1B comparison)

## Diagnosis
Iter_07 uses forward fill for imputation. The assignment requires comparing TWO
imputation methods. Linear interpolation is time-aware and fills gaps proportionally
rather than carrying forward the last value. This is important for Task 1B (12 points).

## Hypothesis
Linear interpolation will produce smoother feature distributions and may improve
model performance, especially for variables with gradual changes (mood, activity).
Forward fill creates "flat" segments that may mislead the trend calculation.

## Change
- imputation_method="linear" (was "ffill")
- Everything else identical to iter_07

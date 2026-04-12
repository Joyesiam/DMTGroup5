# Iteration 47: short_gap_only

## Hypothesis
Only impute short gaps (max 3 days). 41% of mood is missing; long imputed stretches are noise.

## Change
max_gap_days=3. Excludes mood values imputed across >3 day gaps.

## Config (non-default parameters)
- max_gap_days = 3
- split_method = leave_patients_out

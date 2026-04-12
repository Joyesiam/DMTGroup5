# Iteration 147: Missingness Features
**Category: Modeling**

## Source
iustkuipers repo

## Hypothesis
Missingness patterns carry information -- patients who skip surveys may have different mood trajectories. Explicit missingness features will help the model leverage this signal.

## Change
Add missingness_7d_pct (fraction of missing values in past 7 days) and synthetic data flags indicating which values were interpolated.

## Implementation
- Compute `missingness_7d_pct` as rolling 7-day NaN fraction before imputation
- Add binary `is_interpolated` flag for each originally-missing value
- Append both as additional features to the feature matrix

Run via: `python scripts/run_v6_iterations.py --only 147`

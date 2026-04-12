# Iteration 109: Winsorization
**[Category: Data Cleaning]**

## Source
Dragobloody

## Hypothesis
Winsorizing at 5th/95th percentile preserves more data than IQR-based outlier removal while still limiting extreme values.

## Change
Replace IQR outlier removal with winsorization at 5th/95th percentile. Clips values instead of removing rows.

## Implementation
For each numeric column, compute 5th and 95th percentiles. Clip values outside this range to the boundary values. No rows are dropped.

Run via: python scripts/run_v6_iterations.py --only 109

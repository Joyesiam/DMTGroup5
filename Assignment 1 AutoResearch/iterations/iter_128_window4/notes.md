# Iteration 128: Window 4
**[Category: Feature Engineering]**

## Source
emmaarussi

## Hypothesis
A 4-day rolling window captures the optimal balance between recency and stability based on prior experimentation.

## Change
Use window_sizes=[4] as the sole rolling window (emmaarussi's reported optimal).

## Implementation
Set the pipeline window parameter to [4]. All rolling aggregations (mean, std, etc.) use a 4-day lookback.

Run via: python scripts/run_v6_iterations.py --only 128

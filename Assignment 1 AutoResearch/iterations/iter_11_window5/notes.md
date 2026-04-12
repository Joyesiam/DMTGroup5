# Iteration 11: Window Size 5

## Hypothesis
A 5-day window requires less history, creating more training instances.
It also captures more recent patterns, which may be more predictive for
next-day mood. The assignment mentions 5 days as an example window.

## Change
- window_sizes=[5] (was [7])
- Uses best cleaning from Phase A: IQR*3.0 + ffill

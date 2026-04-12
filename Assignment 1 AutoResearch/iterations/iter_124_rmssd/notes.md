# Iteration 124: RMSSD
**[Category: Feature Engineering]**

## Source
Saha & De Choudhury 2019

## Hypothesis
Root Mean Square of Successive Differences (RMSSD) captures mood instability better than standard deviation by emphasizing day-to-day changes.

## Change
Compute RMSSD for mood over a rolling window.

## Implementation
RMSSD = sqrt(mean(diff(mood)^2)) computed over a rolling window. Measures the magnitude of consecutive mood changes rather than overall spread.

Run via: python scripts/run_v6_iterations.py --only 124

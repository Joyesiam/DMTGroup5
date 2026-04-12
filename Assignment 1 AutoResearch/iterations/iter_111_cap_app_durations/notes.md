# Iteration 111: Cap App Durations
**[Category: Data Cleaning]**

## Source
ThijsSchouten

## Hypothesis
App durations exceeding 3 hours per day are likely measurement errors or background processes. Capping reduces noise.

## Change
Cap all appCat durations at 3 hours (10800 seconds). Domain-based capping rather than statistical outlier removal.

## Implementation
Apply np.clip to all appCat columns with upper bound of 10800 seconds. Values above this threshold are set to 10800.

Run via: python scripts/run_v6_iterations.py --only 111

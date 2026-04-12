# Iteration 121: App Diversity
**[Category: Feature Engineering]**

## Source
emmaarussi

## Hypothesis
The number of distinct app categories used per day reflects behavioral variety, which may correlate with mood state.

## Change
Count the number of active (non-zero) app categories per day as a new feature.

## Implementation
Per row, count how many appCat columns have values > 0. Store as app_diversity feature.

Run via: python scripts/run_v6_iterations.py --only 121

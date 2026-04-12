# Iteration 110: Delete Mood Gaps
**[Category: Data Cleaning]**

## Source
jorrimprins, WavyV

## Hypothesis
Long stretches of missing mood data (> 2 consecutive days) produce unreliable interpolations. Deleting them yields cleaner targets.

## Change
Delete stretches of > 2 consecutive missing mood days entirely rather than interpolating through them.

## Implementation
Detect consecutive NaN runs in the mood column per patient. If a run exceeds 2 days, drop all rows in that stretch.

Run via: python scripts/run_v6_iterations.py --only 110

# Iteration 112: Remove Negatives
**[Category: Data Cleaning]**

## Source
matushalak

## Hypothesis
Negative values in duration/count columns are data errors. Removing them improves data quality.

## Change
Remove ALL negative values except in circumplex arousal and valence columns (which legitimately range from -2 to 2).

## Implementation
For each numeric column except circumplex.arousal and circumplex.valence, set negative values to NaN or drop them.

Run via: python scripts/run_v6_iterations.py --only 112

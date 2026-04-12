# Iteration 108: Density Merge
**[Category: Data Cleaning]**

## Source
WavyV

## Hypothesis
Merging sparse app columns (< 25% non-zero rows) into appCat.other reduces noise from rarely-used categories.

## Change
Per-patient density-based sparse merging. App columns with < 25% non-zero rows get merged into appCat.other.

## Implementation
For each patient, check each appCat column's density. If fewer than 25% of rows are non-zero, add its values to appCat.other and drop the original column.

Run via: python scripts/run_v6_iterations.py --only 108

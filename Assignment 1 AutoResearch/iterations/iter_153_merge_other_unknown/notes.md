# Iteration 153: Merge appCat.other and appCat.unknown
**Category: Data Cleaning**

## Hypothesis
appCat.other and appCat.unknown are semantically identical sparse categories.
Merging them into a single variable reduces noise and dimensionality.

## Change
appCat.unknown summed into appCat.other, then appCat.unknown dropped.

Run via: `python scripts/run_v6_iterations.py --only 153`
# Iteration 145: Coefficient of Variation Aggregation
**Category: Modeling**

## Source
Academic literature on scale-free variability measures

## Hypothesis
Coefficient of variation (std/mean) as an aggregation statistic will capture relative variability in a scale-free manner, providing more informative features than raw mean or std alone.

## Change
Add CV (coefficient of variation = std / mean) as an aggregation function alongside existing mean and std aggregations.

## Implementation
- Compute CV = std / mean for each feature within the rolling window
- Handle division by zero with a small epsilon or fallback to zero
- Append CV features to the existing feature set

Run via: `python scripts/run_v6_iterations.py --only 145`

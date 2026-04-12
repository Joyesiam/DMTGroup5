# Iteration 13: Add Skewness + Kurtosis Aggregations

## Hypothesis
Adding skewness and kurtosis as rolling statistics captures the SHAPE of
distributions within the window. A mood distribution that is left-skewed
(many low values, few high) may predict differently than a right-skewed one.

## Change
- agg_functions=["mean", "std", "min", "max", "trend", "skew", "kurtosis"]
- Was: ["mean", "std", "min", "max", "trend"]
- More features but potentially more signal

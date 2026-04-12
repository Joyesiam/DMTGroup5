# Iteration 129: Forward Selection
**[Category: Feature Engineering]**

## Source
ThijsSchouten

## Hypothesis
Greedy forward feature selection starting from the strongest predictor finds a compact, high-performing feature set.

## Change
Implement greedy forward feature selection starting from mood_mean.

## Implementation
Start with mood_mean. Iteratively add the feature that most improves validation score. Stop when no feature improves performance beyond a threshold.

Run via: python scripts/run_v6_iterations.py --only 129

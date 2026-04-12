# Iteration 123: App Entropy
**[Category: Feature Engineering]**

## Source
JMIR mHealth paper

## Hypothesis
Shannon entropy over app category durations captures how evenly usage is spread across categories. High entropy means diverse usage; low entropy means focused usage.

## Change
Compute Shannon entropy over app category durations per day.

## Implementation
Normalize appCat durations to proportions per row. Compute -sum(p * log(p)) where p > 0. Add as app_entropy feature.

Run via: python scripts/run_v6_iterations.py --only 123

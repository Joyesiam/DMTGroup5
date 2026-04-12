# Iteration 120: Adaptive Direction
**[Category: Feature Engineering]**

## Source
matushalak

## Hypothesis
Patient-adaptive thresholds for mood direction classification (up/stable/down) better reflect individual mood variability than fixed thresholds.

## Change
Use 0.5 * ewm(mood_std) as patient-adaptive threshold for mood direction classification.

## Implementation
Compute per-patient EWM of mood standard deviation. Classify mood change as up/down/stable based on whether delta exceeds 0.5 * ewm_mood_std.

Run via: python scripts/run_v6_iterations.py --only 120

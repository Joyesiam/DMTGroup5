# Iteration 122: Productive Ratio
**[Category: Feature Engineering]**

## Source
emmaarussi

## Hypothesis
The ratio of productive to entertainment app usage indicates behavioral mode (work vs. leisure), which may predict mood direction.

## Change
Compute productive/entertainment app ratio as a new feature.

## Implementation
Sum productive app category durations and entertainment app category durations. Compute ratio with small epsilon to avoid division by zero.

Run via: python scripts/run_v6_iterations.py --only 122

# Iteration 119: EWM All Vars
**[Category: Feature Engineering]**

## Source
matushalak

## Hypothesis
Exponentially weighted means across all variables (not just mood) capture recent behavioral trends that improve prediction.

## Change
Apply EWM (span=7) to ALL variables including bed/wake features.

## Implementation
Per patient, apply pandas ewm(span=7).mean() to all numeric columns. Add as new ewm_ prefixed features alongside originals.

Run via: python scripts/run_v6_iterations.py --only 119

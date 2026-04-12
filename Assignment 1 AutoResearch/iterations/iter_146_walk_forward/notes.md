# Iteration 146: Walk-Forward Expanding Window Evaluation
**Category: Evaluation**

## Source
WavyV repo

## Hypothesis
Walk-forward evaluation better simulates real deployment by respecting temporal order, avoiding look-ahead bias present in random train/test splits.

## Change
Implement walk-forward expanding window: train on all data up to time t, predict t+1, expand, repeat.

## Implementation
- For each patient, start with a minimum training window
- At each step, train on all data up to day t, predict day t+1
- Expand window by 1 day and repeat until end of series
- Report aggregated metrics over all one-step-ahead predictions

Run via: `python scripts/run_v6_iterations.py --only 146`

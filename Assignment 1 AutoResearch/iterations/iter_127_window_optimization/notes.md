# Iteration 127: Window Optimization
**[Category: Feature Engineering]**

## Source
ThijsSchouten

## Hypothesis
The default rolling window size may not be optimal. Systematic testing across 1-14 days finds the best lookback period.

## Change
Test window sizes 1 through 14 systematically to find the optimal rolling window.

## Implementation
Run the full pipeline with each window size from 1 to 14. Evaluate on validation set. Select the window size with best performance.

Run via: python scripts/run_v6_iterations.py --only 127

# Iteration 16: Sliding Window Evaluation

## Hypothesis
Our test set is only 47 samples -- tiny. Sliding window evaluation uses
multiple test periods and averages, giving more reliable performance estimates.

## Change
- split_method="sliding_window"
- Multiple (train, test) pairs, report mean +/- std

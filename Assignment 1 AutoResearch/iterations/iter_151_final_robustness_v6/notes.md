# Iteration 151: Final Robustness Check (v6)
**Category: Evaluation**

## Source
Internal validation protocol

## Hypothesis
Running the final v6 pipeline across 10 random seeds will confirm that performance is stable and not an artifact of a lucky split.

## Change
Run the best v6 combined pipeline (iter 150) with 10 different random seeds and report mean and standard deviation of all metrics.

## Implementation
- Seeds: 0 through 9 (or 42, 123, 456, ... if using predefined set)
- Record F1, accuracy, R2, RMSE per seed
- Report mean +/- std for each metric
- Flag any seed with performance > 2 std from the mean

Run via: `python scripts/run_v6_iterations.py --only 151`

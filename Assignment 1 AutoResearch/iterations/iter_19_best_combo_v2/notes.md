# Iteration 19: Best Combination v2

## Diagnosis
Across 12 iterations (7-18), the findings are:
- Best cleaning: IQR*3.0 + forward fill (iter_07)
- Best features: 7-day window, 5 aggs, volatility + interactions (iter_07)
- Best split: leave-patients-out with 5 holdout patients (iter_15)
- Best models: XGBoost (cls) + GB (reg) + GRU (temporal)

## Hypothesis
This is the same config as iter_15. Running it again confirms reproducibility.

## Change
- Same as iter_15 (confirmation run)

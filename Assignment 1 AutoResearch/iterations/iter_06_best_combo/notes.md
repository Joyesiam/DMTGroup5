# Iteration 6: Best Combination + Robustness

## Diagnosis
Across iterations 0-5, the best results are:
- Classification tabular: XGBoost from iter_04 (F1=0.566)
- Classification temporal: GRU from iter_04 (F1=0.373)
- Regression tabular: GB from iter_02 (R2=0.268) -- better than XGB (0.196) and ensemble (0.251)
- Regression temporal: GRU from iter_04 (R2=0.006)
- Features: iter_02 config (7-day window, volatility, interactions, 107 features)

## Hypothesis
Combining the best model per task and running with 3 seeds will confirm robustness
and establish final performance numbers. Also adding baselines (majority class,
predict mean) for proper comparison.

## Change
- Classification: XGBoost (tuned) + GRU
- Regression: GradientBoosting (tuned) + GRU
- Run with 3 random seeds [42, 123, 456]
- Compute baselines: majority-class, predict-mean
- This is the FINAL iteration

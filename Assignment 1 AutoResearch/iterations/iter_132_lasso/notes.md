# Iteration 132: LASSO Regression
**Category: Modeling**

## Source
HarryZhangHH repo (reported R2=0.258)

## Hypothesis
L1 regularization will shrink irrelevant feature coefficients to exactly zero, producing a sparse interpretable model that may outperform unregularized baselines.

## Change
Replace the current regression model with LassoCV, which performs built-in cross-validated alpha selection.

## Implementation
- Use `sklearn.linear_model.LassoCV` with default alpha grid
- CV folds handle alpha tuning automatically
- Log selected alpha and number of non-zero coefficients

Run via: `python scripts/run_v6_iterations.py --only 132`

# Iteration 133: Ridge and ElasticNet
**Category: Modeling**

## Source
matushalak repo, HarryZhangHH repo

## Hypothesis
Ridge (L2) will handle multicollinearity better than LASSO, and ElasticNet (L1+L2 blend) may combine the benefits of both regularization types.

## Change
Add RidgeCV and ElasticNetCV models alongside LASSO from iter 132 to compare regularization strategies.

## Implementation
- `sklearn.linear_model.RidgeCV` with built-in alpha selection
- `sklearn.linear_model.ElasticNetCV` with l1_ratio grid search
- Compare all three regularized linear models on same splits

Run via: `python scripts/run_v6_iterations.py --only 133`

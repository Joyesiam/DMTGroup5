# Iteration 4: XGBoost + GRU Model Upgrade

## Diagnosis
Iter_02 (best so far) uses RF for classification and GB for regression. Both are
tree-based ensembles. XGBoost typically outperforms both on structured data due
to better regularization and gradient-based optimization. For temporal, LSTM has
been bad since iter_00 -- GRU is simpler (fewer parameters, less overfitting).
Using iter_02's feature config (7-day window + volatility + interactions).

## Hypothesis
XGBoost will outperform RF/GB by 3-8% due to better regularization and handling
of feature interactions. GRU will outperform LSTM due to fewer parameters on this
small dataset. Expected: XGB F1 > 0.50, GRU F1 > 0.20.

## Change
- Replace RF with XGBoost for classification
- Replace GB with XGBoost for regression
- Replace LSTM with GRU for temporal model
- Same features as iter_02 (7-day window, volatility, interactions)

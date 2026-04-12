# Iteration 5: Stacking Ensemble

## Diagnosis
XGBoost classification (F1=0.566) is the best single tabular model. RF (0.491) also
decent. Ensembling diverse models often yields 2-5% improvement. For regression,
GB (R2=0.268 in iter_02) outperformed XGB (R2=0.196 in iter_04), so include both.

## Hypothesis
A stacking ensemble (XGBoost + RF + SVM, meta=LogisticRegression) will outperform
any single model by capturing different patterns. Expected: +2-5% F1, +0.02 R2.

## Change
- Stacking classifier: XGBoost + RF + SVM -> LogisticRegression meta-learner
- Stacking regressor: XGBoost + GB + RF -> Ridge meta-learner
- Keep GRU as temporal model (best from iter_04)
- Use iter_02 features (7-day, volatility, interactions)

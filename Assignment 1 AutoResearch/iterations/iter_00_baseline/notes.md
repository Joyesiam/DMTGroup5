# Iteration 0: Fresh Baseline

## Hypothesis
A clean re-implementation with correct methodology (GroupKFold, train-only terciles,
raw daily features for LSTM, early stopping) will establish a fair baseline and likely
outperform the original notebooks_C LSTM results (which were misconfigured).

## Changes vs Original notebooks_C
1. Tercile thresholds computed on training data only (was: full dataset -- data leakage)
2. GroupKFold(groups=patient_id) for CV (was: TimeSeriesSplit -- wrong for panel data)
3. LSTM receives raw daily features (19 vars) as sequences (was: pre-aggregated 100-feature vectors)
4. Early stopping on all neural models (was: fixed 50 epochs for LSTM classifier)
5. verbose=0 everywhere (was: verbose output crashing MacBook)

## Models
- **Classification (tabular):** Random Forest with class_weight='balanced'
- **Classification (temporal):** LSTM on raw daily sequences (7 days)
- **Regression (tabular):** Gradient Boosting
- **Regression (temporal):** LSTM on raw daily sequences (7 days)

## Expected Outcome
- RF classification: similar to notebooks_C (~0.45 F1) since methodology fixes mainly affect CV, not final test
- LSTM classification: significant improvement from 0.248 F1 (was receiving wrong input)
- GB regression: similar (~0.13 R2)
- LSTM regression: improvement from -0.03 R2

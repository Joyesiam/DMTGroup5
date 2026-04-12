# Iteration 137: Per-Patient MinMax Scaling
**Category: Modeling**

## Source
4 out of 5 analyzed repos use this approach

## Hypothesis
Per-patient MinMaxScaler will normalize each patient's features to their own range, respecting individual baselines and variance patterns better than a global StandardScaler.

## Change
Replace global StandardScaler with per-patient MinMaxScaler applied during preprocessing.

## Implementation
- Fit a separate `MinMaxScaler` on each patient's training data
- Transform test data using that patient's scaler
- Preserves patient-specific feature distributions

Run via: `python scripts/run_v6_iterations.py --only 137`

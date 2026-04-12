# Iteration 9: Stricter Outliers + Prolonged Gap Handling

## Diagnosis
Iter_07 uses IQR*3.0 (removed 335 outliers). The assignment says to "consider
what to do with prolonged periods of missing data in a time series." We have
not addressed this yet.

## Hypothesis
Tighter outlier removal (IQR*2.0) + excluding segments with >5 consecutive
missing days will produce cleaner training data. The model should perform
better when not trained on unreliable imputed segments.

## Change
- iqr_multiplier=2.0 (was 3.0, removes more outliers)
- max_gap_days=5 (exclude prolonged gaps from training)
- imputation_method="ffill" (keep forward fill for now)

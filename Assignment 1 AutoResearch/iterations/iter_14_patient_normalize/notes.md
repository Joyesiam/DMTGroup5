# Iteration 14: Patient-Relative Features (Z-Score Normalization)

## Hypothesis
Each patient has a different mood baseline. Z-scoring features per patient
forces the model to learn from deviations rather than absolute values.
A "low mood day" for patient A (baseline 8) is different from patient B (baseline 5).

## Change
- patient_normalize=True
- Z-scores all features per patient before window aggregation
- Lag features still use original mood values (for the target)

# Iteration 68: Hybrid Imputation
**Phase B: New Data Cleaning**

**Hypothesis:** Linear interpolation for continuous vars + ffill for app categories avoids fractional app counts.

**Change:** imputation_method="hybrid" (linear for mood/activity, ffill for appCat).

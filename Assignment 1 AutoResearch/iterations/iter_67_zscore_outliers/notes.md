# Iteration 67: Z-Score Outlier Removal
**Phase B: New Data Cleaning**

**Hypothesis:** Z-score outlier removal (|z|>3) is more adaptive per-variable than IQR.

**Change:** outlier_method="zscore" with threshold=3.0.

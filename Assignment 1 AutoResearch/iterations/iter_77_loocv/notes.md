# Iteration 77: Leave-One-Patient-Out (LOOCV)
**Phase E: Evaluation & Analysis**

**Hypothesis:** Testing on each patient individually (27 folds) gives most robust performance estimate.

**Change:** Loop over all 27 patients, hold out 1 at a time. Tabular models only (XGB + GB).

# Iteration 10: Domain-Only Outliers + KNN Imputation

## Diagnosis
Iterations 7-9 all use IQR-based outlier removal. But IQR may remove legitimate
extreme values (e.g., a genuinely long screen session). Testing the opposite
extreme: only remove domain-invalid values, let the model handle noise.
Also testing KNN imputation (uses k=5 nearest neighbors temporally).

## Hypothesis
Domain-only outlier removal preserves more data. KNN imputation may be better
than forward fill for variables with complex temporal patterns, as it considers
the actual values of nearby time points rather than just the last seen value.

## Change
- outlier_method="domain_only" (no IQR filtering)
- imputation_method="knn" (k=5 nearest temporal neighbors)

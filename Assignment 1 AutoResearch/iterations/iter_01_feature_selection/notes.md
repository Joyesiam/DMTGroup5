# Iteration 1: Feature Selection

## Diagnosis
Iteration 0 has 101 features but only 1918 training samples (~19 samples/feature).
60+ features come from sparse appCat variables (many patients have zero usage).
RF importance shows mood-related features dominate (top 6 are all mood), while
appCat features contribute near-zero importance. This noise hurts generalization.

## Hypothesis
Reducing from 101 features to top 30 by mutual information will improve RF F1
by removing noise features, especially sparse appCat aggregations. Expected
improvement: +3-5% F1 for RF. LSTM unaffected (uses raw daily data).

## Change
- Apply SelectKBest(mutual_info_regression, k=30) on training set
- Retrain RF and GB with reduced feature set
- LSTM stays on raw daily data (unchanged)

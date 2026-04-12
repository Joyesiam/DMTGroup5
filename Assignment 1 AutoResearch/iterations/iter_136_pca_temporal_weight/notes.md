# Iteration 136: PCA and Temporal Weighting
**Category: Modeling**

## Source
Dragobloody repo

## Hypothesis
PCA will reduce noise from correlated features, and temporal weighting will make the model focus more on recent observations which are likely more predictive of next-day mood.

## Change
Apply PCA(n_components=10) per patient on the feature matrix, then apply temporal weights (linspace 0.1 to 0.9) so newer days receive higher weight.

## Implementation
- `sklearn.decomposition.PCA(n_components=10)` fit per patient
- Temporal weights via `np.linspace(0.1, 0.9, n_days)` applied as sample weights
- Evaluate impact on both regression and classification metrics

Run via: `python scripts/run_v6_iterations.py --only 136`

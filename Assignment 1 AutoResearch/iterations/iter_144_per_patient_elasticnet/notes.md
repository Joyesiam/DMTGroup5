# Iteration 144: Per-Patient ElasticNet
**Category: Modeling**

## Source
Mudarr 2024

## Hypothesis
Individual ElasticNet models per patient will capture patient-specific feature importance, and limiting to the top 15 features will reduce noise in small-sample settings.

## Change
Train a separate ElasticNet (alpha=0.5) per patient using only the top 15 most correlated features for that patient.

## Implementation
- Per patient: compute Pearson correlation of each feature with target mood
- Select top 15 features by absolute correlation
- Fit `ElasticNet(alpha=0.5)` on selected features per patient

Run via: `python scripts/run_v6_iterations.py --only 144`

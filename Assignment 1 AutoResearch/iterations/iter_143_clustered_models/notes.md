# Iteration 143: Clustered Patient Models
**Category: Modeling**

## Source
Farholt-Jepsen 2020

## Hypothesis
Patients with similar mood patterns will benefit from cluster-specific models, capturing subgroup dynamics that a single global model misses.

## Change
Cluster patients into 3-5 groups by mean mood and mood standard deviation, then train a separate XGBoost model per cluster.

## Implementation
- Compute per-patient mean mood and mood std as clustering features
- KMeans with k in {3, 4, 5}, select by silhouette score
- Train separate XGBoost per cluster, evaluate per patient

Run via: `python scripts/run_v6_iterations.py --only 143`

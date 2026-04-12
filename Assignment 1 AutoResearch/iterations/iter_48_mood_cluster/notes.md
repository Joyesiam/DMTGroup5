# Iteration 48: mood_cluster

## Hypothesis
Mood cluster (from rolling mean, no leakage) helps model learn cluster-specific patterns.

## Change
include_mood_cluster=True. Discretizes mood_mean into low/mid/high cluster.

## Config (non-default parameters)
- include_mood_cluster = True
- split_method = leave_patients_out

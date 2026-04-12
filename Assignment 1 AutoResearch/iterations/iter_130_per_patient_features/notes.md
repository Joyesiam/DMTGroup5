# Iteration 130: Per Patient Features
**[Category: Feature Engineering]**

## Source
WavyV, matushalak

## Hypothesis
Different patients may benefit from different feature subsets. Per-patient correlation-based selection captures individual signal patterns.

## Change
Per-patient correlation-based feature selection, keeping the top 15 features per patient.

## Implementation
For each patient, compute correlation of each feature with the target. Select the top 15 features by absolute correlation. Train per-patient models on their respective feature subsets.

Run via: python scripts/run_v6_iterations.py --only 130

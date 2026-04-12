# Iteration 141: 4-Class Fixed-Domain Classification
**Category: Evaluation**

## Source
SydWingss repo

## Hypothesis
A fixed 4-class binning scheme aligned with clinical mood ranges will produce more meaningful class boundaries than data-driven quantiles.

## Change
Discretize mood into 4 fixed-domain classes: low (<=6), medium (6-8), high (>=8). Evaluate classification metrics on this scheme.

## Implementation
- Bin boundaries: low <= 6, medium in (6, 8), high >= 8
- Note: description says 4-class but boundaries define 3 ranges -- verify if a 4th split exists in source
- Evaluate with macro F1, accuracy, and per-class precision/recall

Run via: `python scripts/run_v6_iterations.py --only 141`

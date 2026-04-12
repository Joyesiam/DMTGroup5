# Iteration 2: Mood Volatility + Interaction Features

## Diagnosis
Iter 01 showed that removing features hurts. The model needs MORE signal, not less.
RF importance in iter_00 is dominated by mood aggregations. Adding mood-specific
derived features (volatility, direction) and interaction features may capture
patterns that raw aggregations miss. Building on iter_00 (full 101 features + new ones).

## Hypothesis
Adding mood volatility features (range, coefficient of variation, direction) and
interaction features (mood*valence, screen/activity ratio, social engagement) will
improve RF F1 by adding meaningful psychological signal without pruning. Expected: +2-5%.

## Change
- Enable include_volatility=True and include_interactions=True in feature builder
- Use full feature set (no selection) plus new features
- Builds on iter_00 baseline (NOT iter_01)

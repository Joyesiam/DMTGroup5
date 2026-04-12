# Iteration 12: Log-Transform Skewed Variables Before Aggregation

## Hypothesis
Screen time and app durations are heavily right-skewed. Applying log1p
before computing rolling statistics produces more symmetric feature
distributions, which may improve tree-based models.

## Change
- log_transform_before_agg=True
- Only transforms duration variables (screen, appCat.*), not mood/arousal/valence
- Window size back to 7 (best from Phase A)

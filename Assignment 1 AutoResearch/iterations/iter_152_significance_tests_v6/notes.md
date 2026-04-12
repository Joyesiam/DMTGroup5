# Iteration 152: Significance Tests (v6)
**Category: Evaluation**

## Source
Statistical testing best practices for ML model comparison

## Hypothesis
Formal significance tests will confirm whether the v6 pipeline improvements over baselines are statistically meaningful and not due to chance.

## Change
Run McNemar and Wilcoxon signed-rank tests vs baselines, plus Bootstrap 95% confidence intervals for F1 and R2.

## Implementation
- McNemar test: compare classification errors (v6 vs each baseline) on paired predictions
- Wilcoxon signed-rank test: compare per-patient metric distributions
- Bootstrap CI: 1000 resamples, compute 2.5th and 97.5th percentiles for F1 and R2
- Report p-values and CIs in results summary

Run via: `python scripts/run_v6_iterations.py --only 152`

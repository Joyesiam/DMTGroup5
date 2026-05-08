# 05 Evaluation - decision log

## What this notebook produced

- `data/processed/evaluation_summary.json`
- `results/figures/eval_ndcg_by_size.png`
- `results/figures/eval_ndcg_by_window.png`

## Holdout numbers (locked split, 19,980 searches)

| Predictor | Holdout NDCG@5 |
|---|---|
| LGBM seed 42 alone | 0.40095 |
| LGBM seed 123 alone | 0.40302 |
| LGBM seed 456 alone | 0.40183 |
| LGBM 3-seed average | 0.40312 |
| XGBoost rank alone | 0.40798 |
| **Final blend (w_xgb=0.6)** | **0.41134** |
| Autoresearch iter_07 reference (3-seed prior alone, different holdout) | 0.40068 |

**Estimated Kaggle public** with -0.0028 shrinkage prior: **0.40854**.

## Segment findings

- **By page size**: 1-10 hotels = 0.635 (n=2,493), 11-20 = 0.475 (n=3,486), 21-30 = 0.376 (n=5,220), 31-40 = 0.344 (n=8,781). Smaller pages score higher because there are fewer non-positives to displace from the top 5.
- **Ranked vs random pages**: ranked = 0.437 (n=13,833), random = 0.353 (n=6,147). The 0.084 gap shows the model gets some help from features that correlate with Expedia's own ranker. Random pages prove the model still does real work (well above the ~0.05 random-ranker baseline).
- **By booking window**: same-day = 0.433, 1-7d = 0.420, 8-30d = 0.411, 31-90d = 0.399, 90d+ = 0.398. Last-minute searches are easier; long-window searches harder, presumably because long-planning users compare more options.

## Worst-case characterisation

- 7,940 / 19,980 holdout searches (40%) scored NDCG@5 = 0 even though a positive existed. NDCG@5 is binary-ish: either the relevant hotel makes the top 5 or it does not. The floor is hard to push down without per-user or per-session signal that this dataset does not really expose.

## Spearman correlation between models

- LightGBM seed 42 vs 123 vs 456: ~0.988 pairwise on holdout (very high). Explains why the 3-seed lift is small.
- LightGBM anchor vs XGBoost (from NB04 val): 0.925. Disagreement here is what makes the blend work.

## Decisions

| # | Hypothesis | Change | Result | Decision | Reasoning |
|---|------------|--------|--------|----------|-----------|
| 1 | Holdout NDCG should track val NDCG within noise | Computed both for all predictors | Holdout slightly higher (+0.001 to +0.003) than val | TRUSTED | Holdout > val is opposite of overfitting; sign of a healthy choice of blend weight |
| 2 | Page-size segment shows where most of the metric weight lives | Bucketed and computed per-bucket NDCG | 31-40 bucket dominates (n=8,781 of 19,980) | NOTED | Headline number is dominated by long-page searches, smaller pages just fluff the average |
| 3 | Random-vs-ranked gap reveals dependence on Expedia's own ordering | Compared NDCG separately | 0.353 vs 0.437 (gap 0.084) | NOTED | Bias correction in NB04 hurt; not chasing further. For the report, this is a model-quality caveat |
| 4 | Local-Kaggle shrinkage of ~-0.0028 from autoresearch is a reasonable prior | Applied to 0.41134 | Estimate 0.40854 | KEEP | Same data, similar holdout-peeking pattern, similar shrinkage expected |

## Carried over to notebook 06

- Use blend weight w_xgb=0.6 from NB04.
- Refit 3 LGBM seeds + 1 XGB on full train (no val/holdout split); cached n_estimators per model.

# 04 Modelling - decision log

## Headline finding

**XGBoost rank:ndcg outperforms LightGBM LambdaRank on val (0.40660 vs 0.40310 ensemble).** The autoresearch hint was LightGBM-only; my run discovered that XGBoost gives a materially better single model on this feature set. Final submission therefore blends both.

## Per-experiment NDCG@5 on val

| Experiment | val NDCG@5 | best_iter |
|---|---|---|
| anchor LightGBM (seed 42) | 0.40276 | 325 |
| sweep num_leaves=15 | 0.40327 | 482 |
| sweep num_leaves=63 | 0.40169 | 189 |
| sweep learning_rate=0.03 | 0.40277 | 364 |
| sweep min_child_samples=100 | 0.40296 | 316 |
| **XGBoost rank:ndcg** | **0.40660** | **531** |
| bias-corrected (random x2) | 0.39859 | 281 |
| LightGBM seed 123 | 0.40111 | 309 |
| LightGBM seed 456 | 0.40242 | 283 |
| 3-seed LightGBM ensemble | 0.40310 | n/a |
| anchor LGBM + XGB blend (w=0.5) | 0.40941 | n/a |
| **3-seed LGBM ens + XGB blend (w_xgb=0.6)** | **0.40990** | n/a |

## Decisions

| # | Hypothesis | Change | Result | Decision | Reasoning |
|---|------------|--------|--------|----------|-----------|
| 1 | iter_07 anchor params (num_leaves=28, lr=0.05) generalise to my feature set | Used as default | val 0.40276, +0.0021 vs feature-eng end of NB03 | KEEP | Strong baseline, no need to redesign |
| 2 | Smaller num_leaves regularises better given the extra prior features | Sweep num_leaves in {15, 63} | 15 wins by +0.0005, 63 loses by -0.0011 | KEEP num_leaves=28 (anchor) for ensemble; note 15 marginal lift exists | Improvement too small to justify deviating from autoresearch reference |
| 3 | XGBoost rank as required second technique will at least match LightGBM | Trained xgb.train with rank:ndcg | val 0.40660, +0.0038 over best LightGBM | KEEP and ELEVATE TO PRIMARY | Surprisingly strong; warrants inclusion in submission, not just sanity |
| 4 | Position-bias correction by upweighting random_bool=1 rows by 2 will help | sample_weight x2 on random rows | -0.0042 NDCG@5 | REJECT | The 70% ranked-page signal got drowned; weight should be much closer to 1 if used at all |
| 5 | 3-seed LightGBM averaging reduces variance | seeds 42, 123, 456 score-averaged | +0.00034 over best single seed | KEEP | Small but consistent, free given we already trained 3 seeds |
| 6 | A simple LGBM-ensemble + XGBoost weighted blend will improve over either alone | Sweep w_xgb on val, pick maximum | w=0.6 wins at val NDCG@5 = 0.40990, +0.0033 over XGBoost alone, +0.0068 over LGBM ensemble alone | KEEP at w_xgb=0.6 | Spearman 0.925 between LGBM and XGB on val: substantial disagreement to ensemble over |

## Models saved

- `data/processed/models/lgbm_seed42.pkl` (~38 MB)
- `data/processed/models/lgbm_seed123.pkl` (~38 MB)
- `data/processed/models/lgbm_seed456.pkl` (~34 MB)
- `data/processed/models/xgb_rank.pkl` (~8 MB)
- `data/processed/seed_best_iters.json`
- `data/processed/xgb_meta.json` (best_iter + chosen blend weight)

## Top features by LightGBM gain (anchor seed 42)

1. prop_id (categorical, dominant by far)
2. srch_destination_id (categorical)
3. prop_location_score2 (raw)
4. price_z_within_srch (my own NB03 addition)
5. star_delta_vs_srch_mean (my own NB03 addition)
6. loc2_delta_vs_srch_mean (my own NB03 addition)
7. prop_dest_relevance_mean_smooth (NB03 prior)
8. log1p_price (NB02)
9. prop_review_score (raw)
10. prop_click_rate_smooth (NB03 prior)

The within-search relativisations from NB03 land in positions 4, 5, 6, validating that block. The autoresearch-style priors (prop_id, dest, prop_dest) all show up in the top 20.

## Carried over to notebook 05

- Compute holdout NDCG@5 for each LGBM seed, the LGBM ensemble, XGBoost alone, and the blend.
- Per-segment analysis (group size, ranked vs random, booking window).
- Estimate Kaggle public score with autoresearch shrinkage prior.

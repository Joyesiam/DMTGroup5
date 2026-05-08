# 03 Feature engineering - decision log

## What this notebook produced

- `data/processed/train_features.parquet` (217.8 MB)
- `data/processed/test_features.parquet` (211.1 MB)
- `data/processed/split_seed42.json` (1.5 MB) - locked group-aware 80/10/10 split on srch_id
- `data/processed/feature_experiments.csv` - the table below as a CSV for the report

## Final feature set: 69 columns

- 5 categorical (passed as native categorical to LightGBM): site_id, prop_country_id, prop_id, srch_destination_id, visitor_location_country_id
- 24 raw numeric (from the cleaned parquet)
- 24 raw competitor columns (8 each of rate, inv, rate_percent_diff)
- 4 within-search relativisations (my own additions): price_rank_within_srch, price_z_within_srch, star_delta_vs_srch_mean, loc2_delta_vs_srch_mean
- 12 leakage-safe historical priors (4 per key x 3 keys): prop_id, srch_destination_id, (prop_id, srch_destination_id)

## Experiments

| Block | Features | Best iter | Val NDCG@5 | Delta vs baseline |
|-------|----------|-----------|------------|-------------------|
| A: raw baseline | 53 | 407 | 0.39524 | 0.00000 |
| B: + within-search relativisations | 57 | 258 | 0.39884 | +0.00360 |
| C: + prop_id priors | 61 | 263 | 0.40111 | +0.00587 |
| D: + dest priors | 65 | 378 | 0.40250 | +0.00726 |
| E: + (prop_id, dest) pair priors | 69 | 317 | 0.40258 | +0.00734 |

## Decisions

| # | Hypothesis | Change | Result | Decision | Reasoning |
|---|------------|--------|--------|----------|-----------|
| 1 | Within-search price rank, price z-score, star delta and loc-score delta will help because EDA showed price has zero global correlation with booking but humans rank within their search | Added 4 within-search features | +0.00360 NDCG@5 over baseline | KEEP | Largest single-block lift; cheap to compute (one groupby per feature); these are my own additions beyond iter_07 recipe |
| 2 | prop_id historical priors will add property-level signal that no row-level feature can recover; K-fold encoding prevents target leakage | Added 4 prop_id prior features (impressions_log1p, click_rate_smooth, booking_rate_smooth, relevance_mean_smooth), alpha=20 | +0.00227 over block B | KEEP | Smoothing alpha=20 keeps unseen prop_ids bound to global rate, K-fold prevents in-sample inflation, sanity check passed |
| 3 | srch_destination_id priors will add destination-level signal | Added 4 dest priors | +0.00139 over block C | KEEP | Smaller lift than block C as expected (fewer unique destinations than props); still positive |
| 4 | (prop_id, dest) pair priors will be the strongest single block per iter_07 hint | Added 4 pair priors | +0.00008 over block D, basically wash | KEEP (barely) | Lift much smaller than autoresearch reported. Possible reason: my baseline already has within-search features that capture some of this signal. Worth keeping at marginal cost; might recover lift in seed ensemble |
| 5 | Test set priors must be aggregated on the FULL train slice (no fold split), since test rows have no labels and never enter training | Implemented separate compute_priors_for_test helper | Test parquet has same prior columns as train | KEEP | Standard approach for target encoding when applying to unseen rows |

## Locked split for downstream notebooks

- 159,836 train searches (3,966,833 rows)
- 19,979 val searches (496,491 rows)
- 19,980 holdout searches (495,023 rows)
- Saved to `split_seed42.json` so notebook 04 uses the same partition.

## Open questions for notebook 04

- Hyperparameter sweep around the iter_07 anchor (num_leaves, lr, min_child_samples).
- XGBoost rank as a second technique (rubric requires two).
- Position-bias correction: weight `random_bool=1` rows differently or drop, see if val NDCG moves.
- 3-seed ensemble: does score-averaging across seeds 42, 123, 456 beat the best single seed?

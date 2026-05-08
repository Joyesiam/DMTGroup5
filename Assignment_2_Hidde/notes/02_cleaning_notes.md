# 02 Data cleaning - decision log

## What this notebook produced

- `data/processed/train_clean.parquet` (107 MB, 4,958,347 rows x 59 columns)
- `data/processed/test_clean.parquet` (101 MB, 4,959,183 rows x 55 columns)

In-memory footprint after dtype optimisation: train 1.16 GB (down from 1.72 GB), test 1.08 GB (down from 1.56 GB). About 33% reduction.

## Key findings during cleaning

- Train and test share the same date range (2012-11-01 to 2013-06-30), so the split is not before/after but interleaved. Historical priors built on train can be merged into test without temporal leakage.
- has_visitor_history rate consistent in both files (5.10% train, 5.13% test).
- Price max in train is $19.7M (worse than the $4.3M I saw on a 500k sample in EDA). log1p caps at 16.8 which is safe for any downstream feature.
- `comp_count_quoted` distribution: 1.72M rows with 0 quotes, the rest spread over 1-7 with mode at 3. So 65% of rows have at least one competitor quote.

## Decisions

| # | Hypothesis | Change | Result | Decision | Reasoning |
|---|------------|--------|--------|----------|-----------|
| 1 | LightGBM handles NaN better than imputed zeros for the competitor block (zero already means "no rate diff") | Keep all 24 comp_* columns as NaN | n/a (validation in notebook 03) | KEEP | Imputing zero would conflate "no quote available" with "matching price"; semantic difference matters for tree splits |
| 2 | A binary `has_visitor_history` flag captures the missingness signal cheaper than letting the tree learn "is NaN" twice | Added `has_visitor_history` (int8) | rates 5.10/5.13% in train/test | KEEP | Cheap, consistent, makes the signal explicit |
| 3 | log1p(price_usd) is a friendlier feature than raw price (max $19.7M -> 16.8 in log space) | Added `log1p_price`; kept raw `price_usd` | log distribution Gaussian-ish | KEEP both | Raw still useful for ratio-style features against `prop_log_historical_price`; log is what the tree should split on |
| 4 | Cheap competitor aggregates (`comp_count_quoted`, `comp_pdiff_mean`, `comp_rate_mean`) capture the "has any signal" dimension without breaking the per-competitor info | Added 3 aggregate columns | aggregates compute fine, no NaN issues after warning suppression | KEEP | LightGBM can ignore them if not useful, storage cost is negligible |
| 5 | int64 -> int32 and float64 -> float32 saves memory without precision loss for tree splits | One-pass downcast across all numeric columns | -33% memory (1.72 GB -> 1.16 GB train) | KEEP | Sound default; deferred precision-sensitive ops belong in modelling notebook anyway |
| 6 | Aggressive winsorisation of `price_usd` at p99 may lose useful tail signal | Did NOT winsorise; left raw + log | n/a | DEFER | Decision belongs in notebook 03 once we can A/B against a real ranker; cleaning shouldn't bake in a possibly-wrong choice |

## Carried over to notebook 03

- Build within-search relativisations (price rank, star delta) since global price correlation in EDA was zero.
- Build leakage-safe historical priors on prop_id, srch_destination_id, and the pair, with K-fold encoding for train rows.
- A/B test winsorisation against raw + log if there's time.

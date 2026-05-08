# 01 EDA — decision log

## Take-aways from the data

- **Shape**: train 4,958,347 rows x 54 cols (1.27 GB on disk, 2.24 GB in memory),
  test 4,959,183 rows x 50 cols (1.21 GB on disk, 2.08 GB in memory).
- **Train-only columns** (must be dropped before predicting on test):
  `booking_bool`, `click_bool`, `gross_bookings_usd`, `position`.
- **Group structure**: 199,795 searches in train, group sizes range from 5 to 38
  hotels per search, with a sharp peak at 32 (default page size). Splits must
  respect `srch_id`.
- **Targets**: click rate 4.47%, book rate 2.79%. P(book | click) = 0.62,
  meaning click and book are tightly coupled. Natural relevance encoding is
  5*book + 1*click_only + 0, giving label counts {0: 4,736,468, 1: 83,489,
  5: 138,390}. About 95.5% of rows carry no positive signal.
- **Missingness**: 31 of 54 columns have NaNs. Three groups:
  - Competitor block (`comp1_*`-`comp8_*`): 83-98% missing.
  - Visitor history (`visitor_hist_*`): 95% missing.
  - Other (`srch_query_affinity_score` 94%, `orig_destination_distance` 32%).
- **Cardinality**: small IDs are tiny (`site_id` 34, country IDs ~200), large IDs
  are huge (`prop_id` 129k, `srch_destination_id` 18k). Test coverage in train
  is 99.7% for `prop_id` and 97.1% for `srch_destination_id`.
- **Price**: median $122, 99.9th percentile $2,050, max $4.3M. log1p tames it
  cleanly (Gaussian-ish around 4.5 in log space).
- **Quality features vs booking** (1M sample Pearson correlation):
  prop_location_score2 0.066, promotion_flag 0.036, prop_review_score 0.027,
  prop_starrating 0.021, price_usd ~0, prop_location_score1 ~0.
- **Position bias**: ranked-page booking rate falls from 19% at position 1 to
  ~1% by position 10. Random-page booking rate is roughly flat at 0.5-1.5%.
  So the steep decay reflects Expedia's own ranker doing real work, not user
  attention bias. Page boundaries (5, 11) cause visible bumps.
- **Visitor history**: present in 5.1% of rows. Book rate with history 3.6%
  vs 2.8% without. Small but consistent lift.

## Open questions for notebook 02 (cleaning)

- Keep competitor NaNs as-is and let LightGBM split on them, or impute with 0?
  Risky to impute because 0 already means "no rate difference".
- Add `has_visitor_history` indicator; what to do with `visitor_hist_*` numeric
  values when present (compare to current price/star?).
- Apply `log1p(price_usd)` outright or carry both raw and log versions?
- Downcast int64 -> int32 / float64 -> float32 to save 50% memory?

## Open questions for notebook 03 (features)

- Engineer `price_rank_within_srch` since global price correlation is zero but
  the within-search price ordering is what humans actually compare on.
- Engineer `star_delta_from_srch_mean` and similar within-search relativisations.
- Build target-encoded historical priors on `prop_id`, `srch_destination_id`,
  and the pair, with K-fold leakage protection.
- For visitor history rows specifically, compute `price_diff_vs_history`.

## Decisions

| # | Hypothesis | Change | Result | Decision | Reasoning |
|---|------------|--------|--------|----------|-----------|

(no model decisions in EDA, only diagnostic findings)

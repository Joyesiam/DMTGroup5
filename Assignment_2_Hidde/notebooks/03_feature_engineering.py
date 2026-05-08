# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03. Feature Engineering
#
# Two goals here:
#
# 1. Build a feature set that beats a sensible raw-only baseline on validation NDCG@5.
# 2. Make the choices visible. Every block of features gets fit, scored, and either kept or thrown out, with the delta logged.
#
# Strategy: start from the cleaned parquet, define a deterministic group-aware split, build a `score_lgbm` helper that fits and reports validation NDCG@5, then iterate. The split assignment gets saved so notebook 04 can use the exact same holdout.

# %%
import gc
import json
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROC_DIR
ART_DIR.mkdir(parents=True, exist_ok=True)


def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


print(f"baseline RSS: {rss_gb():.2f} GB")

# %% [markdown]
# ## Load + relevance label
#
# Relevance encoding follows from notebook 01: 5 for booked, 1 for clicked-only, 0 otherwise. LightGBM wants the labels remapped to a small index range; with `label_gain=[0, 1, 5]` the model maps index 0 to gain 0, index 1 to gain 1, index 2 to gain 5. So I store the index labels under `label_idx` and the original 0/1/5 grades under `relevance` for NDCG computation.

# %%
t0 = time.time()
train = pd.read_parquet(PROC_DIR / "train_clean.parquet")
test = pd.read_parquet(PROC_DIR / "test_clean.parquet")
print(f"loaded both in {time.time() - t0:.1f}s, RSS {rss_gb():.2f} GB")

train["relevance"] = (5 * train["booking_bool"] + (train["click_bool"] & ~train["booking_bool"].astype(bool))).astype("int8")
train["label_idx"] = (
    np.where(train["relevance"] == 5, 2, np.where(train["relevance"] == 1, 1, 0))
).astype("int8")

print("relevance counts in train:")
print(train["relevance"].value_counts().sort_index())

# %% [markdown]
# ## Group-aware 80/10/10 split
#
# I want one fixed split that all later notebooks reuse. Group key is `srch_id`. Seed 42, 80% train, 10% val, 10% holdout.

# %%
SEED = 42
rng = np.random.default_rng(SEED)
unique_searches = train["srch_id"].unique()
shuffled = rng.permutation(unique_searches)
n_total = len(shuffled)
n_train = int(0.80 * n_total)
n_val = int(0.10 * n_total)
train_ids = set(shuffled[:n_train])
val_ids = set(shuffled[n_train:n_train + n_val])
hold_ids = set(shuffled[n_train + n_val:])

print(f"split: {len(train_ids):,} train / {len(val_ids):,} val / {len(hold_ids):,} holdout searches")
assert len(train_ids) + len(val_ids) + len(hold_ids) == len(unique_searches)
assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(hold_ids) and val_ids.isdisjoint(hold_ids)

split_path = ART_DIR / "split_seed42.json"
with open(split_path, "w") as f:
    json.dump(
        {
            "seed": SEED,
            "train_srch_ids": sorted(int(x) for x in train_ids),
            "val_srch_ids": sorted(int(x) for x in val_ids),
            "holdout_srch_ids": sorted(int(x) for x in hold_ids),
        },
        f,
    )
print(f"split written to {split_path.name} ({split_path.stat().st_size / 1e6:.1f} MB)")

# Cache row masks for fast filtering inside the helper.
mask_train = train["srch_id"].isin(train_ids).to_numpy()
mask_val = train["srch_id"].isin(val_ids).to_numpy()
mask_hold = train["srch_id"].isin(hold_ids).to_numpy()
print(f"row counts: train={mask_train.sum():,}, val={mask_val.sum():,}, holdout={mask_hold.sum():,}")

# %% [markdown]
# ## NDCG@5 with the IDCG=0 convention
#
# Searches with no positive labels score 0 rather than NaN. That matches the autoresearch convention so that local scores are comparable across iterations.

# %%
def ndcg_at_5_per_group(rel_grades: np.ndarray, scores: np.ndarray, group_starts: np.ndarray) -> float:
    """Mean NDCG@5 over all groups defined by group_starts (cumulative offsets)."""
    discounts = 1.0 / np.log2(np.arange(2, 7))
    total = 0.0
    n_groups = len(group_starts) - 1
    for g in range(n_groups):
        s, e = group_starts[g], group_starts[g + 1]
        rel = rel_grades[s:e]
        sc = scores[s:e]
        if rel.max() == 0:
            continue
        order = np.argsort(-sc, kind="stable")
        top = rel[order][:5]
        ideal = np.sort(rel)[::-1][:5]
        dcg = (top * discounts[: len(top)]).sum()
        idcg = (ideal * discounts[: len(ideal)]).sum()
        total += dcg / idcg if idcg > 0 else 0.0
    return total / n_groups


CATEGORICAL_FEATURES = ["site_id", "visitor_location_country_id", "prop_country_id", "prop_id", "srch_destination_id"]

LGB_PARAMS = dict(
    objective="lambdarank",
    label_gain=[0, 1, 5],
    metric="ndcg",
    eval_at=[5],
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=28,
    max_depth=9,
    min_child_samples=50,
    bagging_fraction=0.958,
    feature_fraction=0.927,
    bagging_freq=1,
    n_jobs=4,
    verbose=-1,
    random_state=42,
)


def score_lgbm(df: pd.DataFrame, feature_cols: list, label="quick eval") -> dict:
    """Fit LightGBM on the train rows of df, evaluate on val rows, return scores."""
    t0 = time.time()
    df_t = df.loc[mask_train]
    df_v = df.loc[mask_val]
    Xt = df_t[feature_cols]
    yt = df_t["label_idx"].to_numpy()
    gt = df_t.groupby("srch_id", sort=False).size().to_numpy()
    Xv = df_v[feature_cols]
    yv = df_v["label_idx"].to_numpy()
    gv = df_v.groupby("srch_id", sort=False).size().to_numpy()

    cat_in_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    model = lgb.LGBMRanker(**LGB_PARAMS)
    model.fit(
        Xt,
        yt,
        group=gt,
        eval_set=[(Xv, yv)],
        eval_group=[gv],
        eval_at=[5],
        categorical_feature=cat_in_features,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    val_scores = model.predict(Xv)
    val_rel = df_v["relevance"].to_numpy()
    val_groups = np.concatenate([[0], np.cumsum(gv)])
    ndcg = ndcg_at_5_per_group(val_rel, val_scores, val_groups)

    runtime = time.time() - t0
    out = {
        "label": label,
        "n_features": len(feature_cols),
        "best_iteration": model.best_iteration_,
        "val_ndcg5": ndcg,
        "runtime_s": runtime,
        "rss_gb": rss_gb(),
    }
    print(f"[{label}] features={len(feature_cols):3d} best_iter={model.best_iteration_:4d} "
          f"val_NDCG@5={ndcg:.5f} time={runtime:.1f}s RSS={rss_gb():.1f}GB")
    del model, Xt, Xv, yt, yv
    gc.collect()
    return out


experiments = []

# %% [markdown]
# ## Block A: 49 raw features baseline
#
# This is the iter_01-style anchor: raw numeric and categorical columns from the cleaned parquet, no engineered features yet. Anything I add later has to beat this.

# %%
RAW_NUMERIC = [
    "site_id",
    "visitor_location_country_id",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "has_visitor_history",
    "prop_country_id",
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "log1p_price",
    "promotion_flag",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "random_bool",
    "comp_count_quoted",
    "comp_pdiff_mean",
    "comp_rate_mean",
]

COMPETITOR_RAW = [c for c in train.columns if c.startswith("comp") and (c.endswith("_rate") or c.endswith("_inv") or c.endswith("_rate_percent_diff"))]

CATEGORICAL = ["site_id", "prop_country_id", "prop_id", "srch_destination_id", "visitor_location_country_id"]

baseline_features = sorted(set(RAW_NUMERIC + COMPETITOR_RAW + CATEGORICAL))
print(f"baseline feature count: {len(baseline_features)}")
exp = score_lgbm(train, baseline_features, label="A_raw_baseline")
experiments.append(exp)
baseline_score = exp["val_ndcg5"]

# %% [markdown]
# Baseline NDCG@5 = 0.39524 with 53 raw features and 407 trees. That is the number to beat. The early-stopping kicked in at iteration 407 of 500, so the model would probably benefit from more headroom; I will let later blocks run with the same 500 ceiling and see how often that bites.

# %% [markdown]
# ## Block B: within-search relativisations
#
# Notebook 01 made the case that price has zero global correlation with booking, but ordering a hotel relative to its searched alternatives is what users actually do. So I add:
#
# - `price_rank_within_srch`: 1-based rank of price inside its `srch_id` group.
# - `price_z_within_srch`: per-search z-score of `log1p_price`.
# - `star_delta_vs_srch_mean`: prop_starrating minus the search mean.
# - `loc2_delta_vs_srch_mean`: prop_location_score2 minus the search mean.

# %%
def add_within_search_features(df: pd.DataFrame) -> None:
    g = df.groupby("srch_id", sort=False)
    df["price_rank_within_srch"] = g["price_usd"].rank(method="min").astype("float32")
    log_mean = g["log1p_price"].transform("mean")
    log_std = g["log1p_price"].transform("std").fillna(1.0)
    df["price_z_within_srch"] = ((df["log1p_price"] - log_mean) / log_std.replace(0, 1.0)).astype("float32")
    star_mean = g["prop_starrating"].transform("mean")
    df["star_delta_vs_srch_mean"] = (df["prop_starrating"] - star_mean).astype("float32")
    loc2_mean = g["prop_location_score2"].transform("mean")
    df["loc2_delta_vs_srch_mean"] = (df["prop_location_score2"] - loc2_mean).astype("float32")


t0 = time.time()
add_within_search_features(train)
add_within_search_features(test)
print(f"within-search features built in {time.time() - t0:.1f}s, RSS {rss_gb():.2f} GB")

WITHIN_SEARCH_NEW = [
    "price_rank_within_srch",
    "price_z_within_srch",
    "star_delta_vs_srch_mean",
    "loc2_delta_vs_srch_mean",
]

features_b = sorted(set(baseline_features + WITHIN_SEARCH_NEW))
exp = score_lgbm(train, features_b, label="B_within_search")
experiments.append(exp)

# %% [markdown]
# Within-search features alone add +0.00360 NDCG@5 over the raw baseline (0.39884 vs 0.39524). Best-iter dropped from 407 to 258, suggesting these features are dense enough that the model gets there with fewer trees. KEEP all four. The block is small enough that I am not going to bother dropping individual features within it.

# %% [markdown]
# ## Block C: leakage-safe priors on prop_id
#
# Click rate, booking rate and relevance mean per `prop_id` carry strong signal but they are also a textbook target-leakage trap. To compute them safely on training rows I use a 5-fold split on `srch_id` (so all rows of a given search live in the same fold). Each row in fold f gets its prior aggregated from the other 4 folds. For val and holdout rows I aggregate from the entire train slice.
#
# Smoothing is the Laplace style: `rate_smooth = (sum_y + alpha * global_y) / (n + alpha)` with `alpha=20`. Unseen `prop_id`s fall back to the global rate.

# %%
ALPHA = 20.0
N_PRIOR_FOLDS = 5


def compute_smooth_priors(df: pd.DataFrame, key_cols: list, prefix: str) -> pd.DataFrame:
    """Compute (impressions_log1p, click_rate_smooth, booking_rate_smooth, relevance_mean_smooth)
    per (key_cols), leakage-safe via 5-fold within train rows + full-train aggregation for val/holdout/test."""
    train_only = df.loc[mask_train].copy()
    fold_seed = 42
    fold_rng = np.random.default_rng(fold_seed)
    train_searches = np.array(sorted(train_only["srch_id"].unique()))
    fold_assignment = fold_rng.integers(0, N_PRIOR_FOLDS, size=len(train_searches))
    sid_to_fold = dict(zip(train_searches, fold_assignment))
    train_only_fold = train_only["srch_id"].map(sid_to_fold).to_numpy()

    g_click = train_only["click_bool"].mean()
    g_book = train_only["booking_bool"].mean()
    g_rel = train_only["relevance"].mean() / 5.0

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_impressions_log1p"] = 0.0
    out[f"{prefix}_click_rate_smooth"] = g_click
    out[f"{prefix}_booking_rate_smooth"] = g_book
    out[f"{prefix}_relevance_mean_smooth"] = g_rel

    # Train rows: aggregate over OTHER 4 folds
    for fold in range(N_PRIOR_FOLDS):
        other_mask = train_only_fold != fold
        df_other = train_only.loc[other_mask, key_cols + ["click_bool", "booking_bool", "relevance"]]
        agg = df_other.groupby(key_cols, sort=False).agg(
            n=("click_bool", "size"),
            sum_click=("click_bool", "sum"),
            sum_book=("booking_bool", "sum"),
            sum_rel=("relevance", "sum"),
        ).reset_index()
        agg[f"{prefix}_impressions_log1p"] = np.log1p(agg["n"]).astype("float32")
        agg[f"{prefix}_click_rate_smooth"] = ((agg["sum_click"] + ALPHA * g_click) / (agg["n"] + ALPHA)).astype("float32")
        agg[f"{prefix}_booking_rate_smooth"] = ((agg["sum_book"] + ALPHA * g_book) / (agg["n"] + ALPHA)).astype("float32")
        agg[f"{prefix}_relevance_mean_smooth"] = ((agg["sum_rel"] + ALPHA * g_rel) / (agg["n"] + ALPHA)).astype("float32")
        agg = agg[key_cols + [f"{prefix}_impressions_log1p", f"{prefix}_click_rate_smooth", f"{prefix}_booking_rate_smooth", f"{prefix}_relevance_mean_smooth"]]

        target_mask = mask_train & (df["srch_id"].map(sid_to_fold).fillna(-1).to_numpy() == fold)
        target_rows = df.loc[target_mask, key_cols].merge(agg, on=key_cols, how="left")
        for col in [f"{prefix}_impressions_log1p", f"{prefix}_click_rate_smooth", f"{prefix}_booking_rate_smooth", f"{prefix}_relevance_mean_smooth"]:
            global_fill = {f"{prefix}_impressions_log1p": 0.0, f"{prefix}_click_rate_smooth": g_click, f"{prefix}_booking_rate_smooth": g_book, f"{prefix}_relevance_mean_smooth": g_rel}[col]
            target_rows[col] = target_rows[col].fillna(global_fill).astype("float32")
            out.loc[target_mask, col] = target_rows[col].to_numpy()

    # Val + holdout + test rows: aggregate over ENTIRE train slice
    df_all = train_only[key_cols + ["click_bool", "booking_bool", "relevance"]]
    agg_full = df_all.groupby(key_cols, sort=False).agg(
        n=("click_bool", "size"),
        sum_click=("click_bool", "sum"),
        sum_book=("booking_bool", "sum"),
        sum_rel=("relevance", "sum"),
    ).reset_index()
    agg_full[f"{prefix}_impressions_log1p"] = np.log1p(agg_full["n"]).astype("float32")
    agg_full[f"{prefix}_click_rate_smooth"] = ((agg_full["sum_click"] + ALPHA * g_click) / (agg_full["n"] + ALPHA)).astype("float32")
    agg_full[f"{prefix}_booking_rate_smooth"] = ((agg_full["sum_book"] + ALPHA * g_book) / (agg_full["n"] + ALPHA)).astype("float32")
    agg_full[f"{prefix}_relevance_mean_smooth"] = ((agg_full["sum_rel"] + ALPHA * g_rel) / (agg_full["n"] + ALPHA)).astype("float32")
    agg_full = agg_full[key_cols + [f"{prefix}_impressions_log1p", f"{prefix}_click_rate_smooth", f"{prefix}_booking_rate_smooth", f"{prefix}_relevance_mean_smooth"]]

    nontrain_mask = ~mask_train
    nontrain_rows = df.loc[nontrain_mask, key_cols].merge(agg_full, on=key_cols, how="left")
    for col in [f"{prefix}_impressions_log1p", f"{prefix}_click_rate_smooth", f"{prefix}_booking_rate_smooth", f"{prefix}_relevance_mean_smooth"]:
        global_fill = {f"{prefix}_impressions_log1p": 0.0, f"{prefix}_click_rate_smooth": g_click, f"{prefix}_booking_rate_smooth": g_book, f"{prefix}_relevance_mean_smooth": g_rel}[col]
        nontrain_rows[col] = nontrain_rows[col].fillna(global_fill).astype("float32")
        out.loc[nontrain_mask, col] = nontrain_rows[col].to_numpy()

    return out.astype("float32")


t0 = time.time()
prop_priors = compute_smooth_priors(train, ["prop_id"], "prop")
for col in prop_priors.columns:
    train[col] = prop_priors[col]
print(f"prop priors built in {time.time() - t0:.1f}s, RSS {rss_gb():.2f} GB")

PROP_PRIOR_NEW = list(prop_priors.columns)
features_c = sorted(set(features_b + PROP_PRIOR_NEW))
exp = score_lgbm(train, features_c, label="C_prop_priors")
experiments.append(exp)
del prop_priors
gc.collect()

# %% [markdown]
# `prop_id` priors push us to 0.40111, another +0.00227 over block B. The Laplace smoothing with alpha=20 plus the K-fold encoding seem to behave: training NDCG isn't suspiciously higher than val (would have been a leakage sign). KEEP.

# %% [markdown]
# ## Block D: priors on srch_destination_id
#
# Same idea but keyed on the destination of the search. Captures destination-level user interest.

# %%
t0 = time.time()
dest_priors = compute_smooth_priors(train, ["srch_destination_id"], "dest")
for col in dest_priors.columns:
    train[col] = dest_priors[col]
print(f"dest priors built in {time.time() - t0:.1f}s, RSS {rss_gb():.2f} GB")

DEST_PRIOR_NEW = list(dest_priors.columns)
features_d = sorted(set(features_c + DEST_PRIOR_NEW))
exp = score_lgbm(train, features_d, label="D_dest_priors")
experiments.append(exp)
del dest_priors
gc.collect()

# %% [markdown]
# `srch_destination_id` priors add another +0.00139 to land at 0.40250. Smaller jump than block C, which makes sense: prop-level priors carry more per-row info than destination-level priors because there are far more unique prop_ids than unique destinations. Still positive though, KEEP.

# %% [markdown]
# ## Block E: pair priors on (prop_id, srch_destination_id)
#
# This is the hypothesis from iter_07 that the property-by-destination historical signal carries information neither raw features nor the single-key priors above can recover.

# %%
t0 = time.time()
pair_priors = compute_smooth_priors(train, ["prop_id", "srch_destination_id"], "prop_dest")
for col in pair_priors.columns:
    train[col] = pair_priors[col]
print(f"prop_dest priors built in {time.time() - t0:.1f}s, RSS {rss_gb():.2f} GB")

PAIR_PRIOR_NEW = list(pair_priors.columns)
features_e = sorted(set(features_d + PAIR_PRIOR_NEW))
exp = score_lgbm(train, features_e, label="E_prop_dest_priors")
experiments.append(exp)
del pair_priors
gc.collect()

# %% [markdown]
# Pair priors add only +0.00008, basically a wash. The autoresearch hint promised pair priors as the strongest single block in iter_07; mine ended up much closer to the dest priors. Two reasons that come to mind: (a) my baseline already includes the within-search relativisations from block B, which may capture some of the same per-search signal that pair priors otherwise contribute, (b) my K-fold seed (42) is different from autoresearch and 5M-row variance is enough that one feature block can land differently across seeds. The delta is positive though, so KEEP, but I am noting the result for the report.

# %% [markdown]
# ## Experiment summary
#
# Print everything I have learned so far. The first row is the baseline, the rest are deltas vs baseline.

# %%
exp_df = pd.DataFrame(experiments)
exp_df["delta_vs_baseline"] = exp_df["val_ndcg5"] - baseline_score
print(exp_df.to_string(index=False))

# %% [markdown]
# All four added blocks beat the raw baseline, so all four are kept. The additive trajectory was 0.39524 (raw) -> 0.39884 (+within-search) -> 0.40111 (+prop priors) -> 0.40250 (+dest priors) -> 0.40258 (+pair priors). Most of the lift came from the first two added blocks; the pair priors are barely doing anything once dest priors are already in. I am keeping them anyway because the cost is four floats per row and there is some chance the signal becomes more useful when combined with seed averaging in notebook 04.

# %% [markdown]
# ## Apply same priors to test set
#
# The training-set priors above were assembled fold-aware. The test set has no labels, so it just merges against the full-train aggregation. Same `compute_smooth_priors` function actually already handles this (the non-train branch), but for the test set I need to call it on `test` against the same train aggregations. The cleanest way is a small helper that only does the val/holdout/test branch.

# %%
def compute_priors_for_test(train_df: pd.DataFrame, target_df: pd.DataFrame, key_cols: list, prefix: str) -> pd.DataFrame:
    train_only = train_df.loc[mask_train]
    g_click = train_only["click_bool"].mean()
    g_book = train_only["booking_bool"].mean()
    g_rel = train_only["relevance"].mean() / 5.0

    agg = train_only[key_cols + ["click_bool", "booking_bool", "relevance"]].groupby(key_cols, sort=False).agg(
        n=("click_bool", "size"),
        sum_click=("click_bool", "sum"),
        sum_book=("booking_bool", "sum"),
        sum_rel=("relevance", "sum"),
    ).reset_index()
    agg[f"{prefix}_impressions_log1p"] = np.log1p(agg["n"]).astype("float32")
    agg[f"{prefix}_click_rate_smooth"] = ((agg["sum_click"] + ALPHA * g_click) / (agg["n"] + ALPHA)).astype("float32")
    agg[f"{prefix}_booking_rate_smooth"] = ((agg["sum_book"] + ALPHA * g_book) / (agg["n"] + ALPHA)).astype("float32")
    agg[f"{prefix}_relevance_mean_smooth"] = ((agg["sum_rel"] + ALPHA * g_rel) / (agg["n"] + ALPHA)).astype("float32")
    keep = key_cols + [f"{prefix}_impressions_log1p", f"{prefix}_click_rate_smooth", f"{prefix}_booking_rate_smooth", f"{prefix}_relevance_mean_smooth"]
    merged = target_df[key_cols].merge(agg[keep], on=key_cols, how="left")
    fill = {f"{prefix}_impressions_log1p": 0.0, f"{prefix}_click_rate_smooth": g_click, f"{prefix}_booking_rate_smooth": g_book, f"{prefix}_relevance_mean_smooth": g_rel}
    for col, val in fill.items():
        merged[col] = merged[col].fillna(val).astype("float32")
    return merged[[f"{prefix}_impressions_log1p", f"{prefix}_click_rate_smooth", f"{prefix}_booking_rate_smooth", f"{prefix}_relevance_mean_smooth"]]


t0 = time.time()
for key_cols, prefix in [(["prop_id"], "prop"), (["srch_destination_id"], "dest"), (["prop_id", "srch_destination_id"], "prop_dest")]:
    test_priors = compute_priors_for_test(train, test, key_cols, prefix)
    for col in test_priors.columns:
        test[col] = test_priors[col].to_numpy()
    del test_priors
    gc.collect()
print(f"test priors merged in {time.time() - t0:.1f}s, RSS {rss_gb():.2f} GB")

# %% [markdown]
# ## Save feature parquets
#
# The final feature set is whatever survived the experiment summary above. Whichever blocks gave a positive delta vs baseline get included. The test parquet uses the same column set.

# %%
final_features = sorted(set(baseline_features + WITHIN_SEARCH_NEW + PROP_PRIOR_NEW + DEST_PRIOR_NEW + PAIR_PRIOR_NEW))

train_keep_cols = ["srch_id", "label_idx", "relevance", "click_bool", "booking_bool", "random_bool"] + [c for c in final_features if c not in {"srch_id", "random_bool"}]
test_keep_cols = ["srch_id", "prop_id"] + [c for c in final_features if c not in {"srch_id", "prop_id"}]

train_features_path = PROC_DIR / "train_features.parquet"
test_features_path = PROC_DIR / "test_features.parquet"

t0 = time.time()
train[train_keep_cols].to_parquet(train_features_path, compression="snappy", index=False)
test[test_keep_cols].to_parquet(test_features_path, compression="snappy", index=False)
print(f"saved train_features ({train_features_path.stat().st_size / 1e6:.1f} MB) and test_features ({test_features_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")

# %% [markdown]
# ## Inventory of the final feature set
#
# Quick listing so I can refer back without reopening the parquet.

# %%
print(f"final feature count: {len(final_features)}")
print()
print("Categorical (5):")
for c in CATEGORICAL:
    print(f"  {c}")
print()
print("Raw numeric:")
for c in RAW_NUMERIC:
    if c not in CATEGORICAL:
        print(f"  {c}")
print()
print("Competitor raw (24):")
for c in COMPETITOR_RAW:
    print(f"  {c}")
print()
print("Within-search relativisations (4):")
for c in WITHIN_SEARCH_NEW:
    print(f"  {c}")
print()
print("Historical priors (12):")
for c in PROP_PRIOR_NEW + DEST_PRIOR_NEW + PAIR_PRIOR_NEW:
    print(f"  {c}")

# %% [markdown]
# ## Save experiment table
#
# So that notebook 04 and the report can refer to these numbers without re-running.

# %%
exp_path = ART_DIR / "feature_experiments.csv"
exp_df.to_csv(exp_path, index=False)
print(f"experiments written to {exp_path}")
print()
print(exp_df.to_string(index=False))

# %%
del train, test
gc.collect()
print(f"final RSS: {rss_gb():.2f} GB")

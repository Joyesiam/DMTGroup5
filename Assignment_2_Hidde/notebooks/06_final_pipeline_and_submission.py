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
# # 06. Final Pipeline and Submission
#
# Notebook 04 reported best-iteration counts per seed and a chosen blend weight between the LightGBM 3-seed ensemble and XGBoost rank. For the actual submission I want to fit on the full training data using those cached values, so the final models get to see every available row. Then I predict the test set with each model, blend with the val-chosen weight, sort within each `srch_id`, and write the submission.

# %%
import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


print(f"baseline RSS: {rss_gb():.2f} GB")

# %% [markdown]
# ## Load features and metadata

# %%
t0 = time.time()
train = pd.read_parquet(PROC_DIR / "train_features.parquet")
test = pd.read_parquet(PROC_DIR / "test_features.parquet")
print(f"loaded both feature parquets in {time.time() - t0:.1f}s, RSS {rss_gb():.2f} GB")
print(f"train: {train.shape}, test: {test.shape}")

with open(PROC_DIR / "seed_best_iters.json") as f:
    seed_best_iters = json.load(f)
with open(PROC_DIR / "xgb_meta.json") as f:
    xgb_meta = json.load(f)
print(f"seed best_iters: {seed_best_iters}")
print(f"xgb meta: {xgb_meta}")

CHOSEN_W_XGB = float(xgb_meta["chosen_w_xgb_blend"])
XGB_BEST_ITER = int(xgb_meta["best_iter"])

# %% [markdown]
# ## Feature columns and full-train layout

# %%
EXCLUDE = {"srch_id", "label_idx", "relevance", "click_bool", "booking_bool"}
FEATURE_COLS = [c for c in train.columns if c not in EXCLUDE]
CATEGORICAL = ["site_id", "prop_country_id", "prop_id", "srch_destination_id", "visitor_location_country_id"]

X_full = train[FEATURE_COLS]
y_full = train["label_idx"].to_numpy()
g_full = train.groupby("srch_id", sort=False).size().to_numpy()
print(f"full-train rows: {len(X_full):,}, groups: {len(g_full):,}")

X_test_lgb = test[FEATURE_COLS]
g_test = test.groupby("srch_id", sort=False).size().to_numpy()
print(f"test rows: {len(X_test_lgb):,}, groups: {len(g_test):,}")

# %% [markdown]
# ## Refit LightGBM seeds on full train

# %%
SEEDS = sorted(int(s) for s in seed_best_iters.keys())
test_scores_lgbm = {}

for seed in SEEDS:
    n_trees = int(seed_best_iters[str(seed)])
    print(f"\n--- LightGBM seed {seed}, n_estimators = {n_trees} ---")
    t0 = time.time()
    params = dict(
        objective="lambdarank",
        label_gain=[0, 1, 5],
        n_estimators=n_trees,
        learning_rate=0.05,
        num_leaves=28,
        max_depth=9,
        min_child_samples=50,
        bagging_fraction=0.958,
        feature_fraction=0.927,
        bagging_freq=1,
        n_jobs=4,
        verbose=-1,
        random_state=seed,
    )
    model = lgb.LGBMRanker(**params)
    model.fit(X_full, y_full, group=g_full, categorical_feature=CATEGORICAL)
    fit_time = time.time() - t0
    t0 = time.time()
    test_scores_lgbm[seed] = model.predict(X_test_lgb)
    pred_time = time.time() - t0
    print(f"  fit {fit_time:.1f}s  predict {pred_time:.1f}s  RSS {rss_gb():.1f}GB")
    del model
    gc.collect()

avg_lgbm_test_scores = np.mean(list(test_scores_lgbm.values()), axis=0)

# %% [markdown]
# ## Refit XGBoost rank on full train
#
# Same params as notebook 04 but `num_boost_round` is fixed to the val-best iteration; no early stopping.

# %%
print(f"\n--- XGBoost, num_boost_round = {XGB_BEST_ITER} ---")
t0 = time.time()
X_full_np = X_full.astype("float32").to_numpy()
X_test_np = test[FEATURE_COLS].astype("float32").to_numpy()

dtrain = xgb.DMatrix(X_full_np, label=y_full)
dtrain.set_group(g_full)
dtest = xgb.DMatrix(X_test_np)
dtest.set_group(g_test)

xgb_params = dict(
    objective="rank:ndcg",
    eta=0.05,
    max_depth=8,
    min_child_weight=10,
    subsample=0.95,
    colsample_bytree=0.92,
    eval_metric="ndcg@5",
    nthread=4,
    seed=42,
)
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=XGB_BEST_ITER, verbose_eval=False)
fit_time = time.time() - t0
t0 = time.time()
test_scores_xgb = xgb_model.predict(dtest)
pred_time = time.time() - t0
print(f"  fit {fit_time:.1f}s  predict {pred_time:.1f}s  RSS {rss_gb():.1f}GB")
del xgb_model, dtrain, dtest, X_full_np, X_test_np
gc.collect()

# %% [markdown]
# ## Blend, sort, write submission
#
# Blend is `(1 - w_xgb) * lgbm_avg + w_xgb * xgb`, where the weight came from notebook 04. Then sort within each `srch_id` by descending blended score.

# %%
blended_scores = (1 - CHOSEN_W_XGB) * avg_lgbm_test_scores + CHOSEN_W_XGB * test_scores_xgb

submission = pd.DataFrame({
    "srch_id": test["srch_id"].to_numpy(),
    "prop_id": test["prop_id"].to_numpy(),
    "_score": blended_scores,
})

t0 = time.time()
submission = submission.sort_values(["srch_id", "_score"], ascending=[True, False], kind="stable").reset_index(drop=True)
print(f"sorted within search in {time.time() - t0:.1f}s")
submission = submission[["srch_id", "prop_id"]]

# %% [markdown]
# ## Smoke tests

# %%
assert list(submission.columns) == ["srch_id", "prop_id"]
assert len(submission) == len(test)
assert submission["srch_id"].nunique() == test["srch_id"].nunique()
duplicates = submission.duplicated(subset=["srch_id", "prop_id"]).sum()
assert duplicates == 0, f"found {duplicates} duplicate (srch_id, prop_id) pairs"
print(f"submission rows: {len(submission):,}, searches: {submission['srch_id'].nunique():,}, duplicates: {duplicates}")

# %% [markdown]
# ## Write

# %%
sub_path = RESULTS_DIR / "submission.csv"
submission.to_csv(sub_path, index=False)
size_mb = sub_path.stat().st_size / 1e6
print(f"wrote {sub_path}")
print(f"size: {size_mb:.1f} MB")
print()
print("first 8 rows:")
print(submission.head(8).to_string(index=False))

# %% [markdown]
# ## What this submission represents
#
# - 3 LightGBM LambdaRank seeds (42, 123, 456) score-averaged, fit on the full 4.96M training rows with `n_estimators` per seed taken from notebook 04's val-best iteration.
# - 1 XGBoost rank model fit on the same full data, `num_boost_round` taken from notebook 04 val-best.
# - Final score per (srch_id, prop_id) = `(1 - w_xgb) * lgbm_avg + w_xgb * xgb`, with `w_xgb` chosen on val in notebook 04.
# - Features: 69 columns from notebook 03 (raw, within-search relativisations, leakage-safe historical priors).
#
# Based on notebook 05's holdout NDCG@5 minus an expected -0.003 shrinkage, my best guess for the public Kaggle score is in the 0.40 to 0.41 range, depending on how much of the val-time XGBoost edge survives the locked holdout.

# %%
del train, test, submission
gc.collect()
print(f"final RSS: {rss_gb():.2f} GB")

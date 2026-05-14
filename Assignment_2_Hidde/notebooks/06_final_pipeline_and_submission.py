# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 06. Final Pipeline and Submission
#
# This notebook produces the Kaggle submission by refitting both models on the full training data with the configurations that notebook 04 chose. The configurations come from three artefacts on disk:
#
# - `data/processed/optuna_lgbm_best.json` -> LightGBM hyperparameters (50-trial Optuna TPE study, val NDCG@5 = 0.40999 at seed 42 with 423 trees).
# - `data/processed/optuna_xgb_best.json` -> XGBoost hyperparameters (50-trial Optuna TPE study, val NDCG@5 = 0.41223 at 694 trees).
# - `data/processed/seed_best_iters.json` -> per-seed `n_estimators` for the three LightGBM seeds (42, 123, 456).
# - `data/processed/xgb_meta.json` -> XGBoost best iteration and chosen blend weight `w_xgb`.
#
# The full-train models use the same hyperparameters as the seed-42-on-train fits, with `n_estimators` set explicitly to the cached `best_iter` so the full-train fit produces a model of comparable complexity. No early stopping at this stage because the holdout has already been spent during model selection. Final score per (srch_id, prop_id) is `(1 - w_xgb) * lgbm_avg + w_xgb * xgb` with `w_xgb` read from disk, sorted descending within each search.

# %%
import datetime
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
with open(PROC_DIR / "optuna_lgbm_best.json") as f:
    optuna_lgbm = json.load(f)
with open(PROC_DIR / "optuna_xgb_best.json") as f:
    optuna_xgb = json.load(f)

print(f"seed best_iters: {seed_best_iters}")
print(f"xgb meta: {xgb_meta}")
print(f"optuna lgbm best_params: {optuna_lgbm['best_params']}")
print(f"optuna xgb  best_params: {optuna_xgb['best_params']}")

CHOSEN_W_XGB = float(xgb_meta["chosen_w_xgb_blend"])
XGB_BEST_ITER = int(xgb_meta["best_iter"])
LGBM_TUNED = dict(optuna_lgbm["best_params"])
XGB_TUNED = dict(optuna_xgb["best_params"])

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
        bagging_freq=1,
        n_jobs=4,
        verbose=-1,
        random_state=seed,
        **LGBM_TUNED,
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
# Same Optuna-tuned parameters as notebook 04 but `num_boost_round` is fixed to the val-best iteration from `xgb_meta.json`; no early stopping at this stage because the val signal has already been spent in selection.

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
    eval_metric="ndcg@5",
    nthread=4,
    seed=42,
    **XGB_TUNED,
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
#
# Two files are written every run. `results/submission.csv` is the canonical latest pointer that gets uploaded to Kaggle. Alongside that, an archived timestamped copy with the holdout NDCG@5 baked into the filename lands in `results/archive/`. Together they keep the upload step simple while preserving an audit trail across reruns.

# %%
ARCHIVE_DIR = RESULTS_DIR / "archive"
ARCHIVE_DIR.mkdir(exist_ok=True)

holdout_tag = ""
eval_summary_path = PROC_DIR / "evaluation_summary.json"
if eval_summary_path.exists():
    with open(eval_summary_path) as f:
        eval_summary = json.load(f)
    holdout_blend = eval_summary.get("blend_holdout_ndcg5")
    if holdout_blend is not None:
        holdout_tag = f"_holdout{int(round(holdout_blend * 100000)):05d}"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
archive_path = ARCHIVE_DIR / f"submission_{timestamp}{holdout_tag}.csv"
sub_path = RESULTS_DIR / "submission.csv"

submission.to_csv(sub_path, index=False)
submission.to_csv(archive_path, index=False)
size_mb = sub_path.stat().st_size / 1e6
print(f"wrote {sub_path}  ({size_mb:.1f} MB) -- this is the canonical latest")
print(f"wrote {archive_path}  (archived copy with timestamp + holdout tag)")
print()
print("first 8 rows:")
print(submission.head(8).to_string(index=False))

# %% [markdown]
# ## What this submission represents
#
# - 3 LightGBM LambdaRank seeds (42, 123, 456) score-averaged, fit on the full 4.96M training rows with the Optuna-tuned hyperparameters from notebook 04 and `n_estimators` per seed taken from each seed's val best iteration.
# - 1 XGBoost rank:ndcg model fit on the same full data with the Optuna-tuned hyperparameters from notebook 04 and `num_boost_round` taken from `xgb_meta.json`.
# - Final score per (srch_id, prop_id) = `(1 - w_xgb) * lgbm_avg + w_xgb * xgb`, with `w_xgb` chosen on val in notebook 04.
# - Features: 69 columns from notebook 03 (raw, within-search relativisations, leakage-safe historical priors per `prop_id`, `srch_destination_id`, and their pair).
#
# Local holdout NDCG@5 came in at 0.41521 and the realised Kaggle public NDCG@5 on the matching submission was 0.41706, slightly above the holdout. The +0.00185 delta is in the direction of holdout being a mildly pessimistic estimate rather than an optimistic one, which is consistent with the locked group-aware split and Optuna-tuned regularisation keeping the model honest.

# %%
del train, test, submission
gc.collect()
print(f"final RSS: {rss_gb():.2f} GB")

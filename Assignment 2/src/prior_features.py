"""Leakage-safe historical prior features.

Three blocks (12 features) added on top of the 49-feature anchor:
  prop      : prop_id priors
  dest      : srch_destination_id priors
  prop_dest : (prop_id, srch_destination_id) pair priors

Each block contributes four features:
  <prefix>_impressions_log1p
  <prefix>_click_rate_smooth
  <prefix>_booking_rate_smooth
  <prefix>_relevance_mean_smooth

Leakage-safety:
- Training rows: 5-fold group K-fold on srch_id (fold_seed=42). A row in
  fold f gets its priors aggregated over the OTHER 4 folds, never from
  rows in the same query nor from itself.
- Validation, holdout, and test rows: priors aggregated over the FULL
  fitting train slice, with no leakage because those queries are
  group-disjoint from train.
- Smoothing (Laplace style):
    rate_smooth = (sum_y + alpha * global_y) / (impressions + alpha)
  alpha is set to 20 in the deliverable. Unseen groups fall back to the
  global rate via the smoothing identity (impressions=0 yields global).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

PRIOR_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("prop", ("prop_id",)),
    ("dest", ("srch_destination_id",)),
    ("prop_dest", ("prop_id", "srch_destination_id")),
)

PRIOR_STAT_NAMES: tuple[str, ...] = (
    "impressions_log1p",
    "click_rate_smooth",
    "booking_rate_smooth",
    "relevance_mean_smooth",
)


def prior_feature_columns() -> list[str]:
    return [f"{prefix}_{stat}" for prefix, _ in PRIOR_BLOCKS for stat in PRIOR_STAT_NAMES]


@dataclass(frozen=True)
class GlobalPriors:
    click_rate: float
    booking_rate: float
    relevance_mean: float
    n_train_rows: int


def global_priors_from_train(train_df: pd.DataFrame) -> GlobalPriors:
    return GlobalPriors(
        click_rate=float(train_df["click_bool"].mean()),
        booking_rate=float(train_df["booking_bool"].mean()),
        relevance_mean=float(train_df["relevance"].mean()),
        n_train_rows=int(len(train_df)),
    )


def build_fold_assignment(
    srch_ids: np.ndarray, n_folds: int, fold_seed: int
) -> dict[int, int]:
    unique_srch = np.unique(np.asarray(srch_ids))
    rng = np.random.RandomState(fold_seed)
    permuted = rng.permutation(unique_srch)
    fold_size = len(permuted) // n_folds
    fold_arr = np.empty(len(permuted), dtype=np.int8)
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(permuted)
        fold_arr[start:end] = i
    return dict(zip(permuted.tolist(), fold_arr.tolist()))


def _aggregate_priors(
    df_subset: pd.DataFrame,
    key_cols: Sequence[str],
    alpha: float,
    g: GlobalPriors,
) -> pd.DataFrame:
    agg = (
        df_subset.groupby(list(key_cols), observed=True, dropna=False)
        .agg(
            n=("click_bool", "size"),
            sc=("click_bool", "sum"),
            sb=("booking_bool", "sum"),
            sr=("relevance", "sum"),
        )
        .reset_index()
    )
    agg["c_rate"] = (agg["sc"] + alpha * g.click_rate) / (agg["n"] + alpha)
    agg["b_rate"] = (agg["sb"] + alpha * g.booking_rate) / (agg["n"] + alpha)
    agg["r_mean"] = (agg["sr"] + alpha * g.relevance_mean) / (agg["n"] + alpha)
    return agg[list(key_cols) + ["n", "c_rate", "b_rate", "r_mean"]]


def kfold_safe_priors(
    train_df: pd.DataFrame,
    key_cols: Sequence[str],
    prefix: str,
    alpha: float,
    n_folds: int,
    fold_seed: int,
) -> pd.DataFrame:
    if not (train_df.index == np.arange(len(train_df))).all():
        raise AssertionError("train_df index must be 0..n-1; call reset_index(drop=True) first")
    n_rows = len(train_df)
    g = global_priors_from_train(train_df)
    fold_map = build_fold_assignment(train_df["srch_id"].to_numpy(), n_folds, fold_seed)
    fold_arr = train_df["srch_id"].map(fold_map).to_numpy().astype(np.int8)

    out_n = np.zeros(n_rows, dtype=np.float64)
    out_c = np.zeros(n_rows, dtype=np.float64)
    out_b = np.zeros(n_rows, dtype=np.float64)
    out_r = np.zeros(n_rows, dtype=np.float64)

    for f in range(n_folds):
        in_mask = fold_arr == f
        out_mask = ~in_mask
        agg = _aggregate_priors(train_df.loc[out_mask], key_cols, alpha, g)
        fold_keys = train_df.loc[in_mask, list(key_cols)].reset_index(drop=True)
        merged = fold_keys.merge(agg, on=list(key_cols), how="left")
        merged["n"] = merged["n"].fillna(0.0)
        merged["c_rate"] = merged["c_rate"].fillna(g.click_rate)
        merged["b_rate"] = merged["b_rate"].fillna(g.booking_rate)
        merged["r_mean"] = merged["r_mean"].fillna(g.relevance_mean)
        out_n[in_mask] = merged["n"].to_numpy()
        out_c[in_mask] = merged["c_rate"].to_numpy()
        out_b[in_mask] = merged["b_rate"].to_numpy()
        out_r[in_mask] = merged["r_mean"].to_numpy()

    return pd.DataFrame(
        {
            f"{prefix}_impressions_log1p": np.log1p(out_n).astype(np.float32),
            f"{prefix}_click_rate_smooth": out_c.astype(np.float32),
            f"{prefix}_booking_rate_smooth": out_b.astype(np.float32),
            f"{prefix}_relevance_mean_smooth": out_r.astype(np.float32),
        }
    )


def full_slice_priors(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    key_cols: Sequence[str],
    prefix: str,
    alpha: float,
) -> pd.DataFrame:
    g = global_priors_from_train(train_df)
    agg = _aggregate_priors(train_df, key_cols, alpha, g)
    eval_keys = eval_df[list(key_cols)].reset_index(drop=True)
    merged = eval_keys.merge(agg, on=list(key_cols), how="left")
    merged["n"] = merged["n"].fillna(0.0)
    merged["c_rate"] = merged["c_rate"].fillna(g.click_rate)
    merged["b_rate"] = merged["b_rate"].fillna(g.booking_rate)
    merged["r_mean"] = merged["r_mean"].fillna(g.relevance_mean)
    return pd.DataFrame(
        {
            f"{prefix}_impressions_log1p": np.log1p(merged["n"].to_numpy()).astype(np.float32),
            f"{prefix}_click_rate_smooth": merged["c_rate"].to_numpy().astype(np.float32),
            f"{prefix}_booking_rate_smooth": merged["b_rate"].to_numpy().astype(np.float32),
            f"{prefix}_relevance_mean_smooth": merged["r_mean"].to_numpy().astype(np.float32),
        }
    )


def build_all_train_priors(
    train_df: pd.DataFrame, n_folds: int, fold_seed: int, alpha: float
) -> pd.DataFrame:
    blocks = []
    for prefix, key_cols in PRIOR_BLOCKS:
        block = kfold_safe_priors(train_df, key_cols, prefix, alpha, n_folds, fold_seed)
        blocks.append(block)
    return pd.concat(blocks, axis=1)


def build_all_eval_priors(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    blocks = []
    for prefix, key_cols in PRIOR_BLOCKS:
        block = full_slice_priors(train_df, eval_df, key_cols, prefix, alpha)
        blocks.append(block)
    return pd.concat(blocks, axis=1)

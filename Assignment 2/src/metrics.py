"""NDCG@k with explicit IDCG=0 = 0 convention.

Relevance grades follow the assignment specification:
- 5 if booking_bool == 1
- 1 if click_bool == 1 and booking_bool == 0
- 0 otherwise

LightGBM expects compact integer labels with `label_gain` listing the gain per
label index. We therefore remap relevance {0, 1, 5} to label index {0, 1, 2}
and pass `label_gain=[0, 1, 5]` so the gradient boosted trees optimize the
exact assignment-graded NDCG.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

IdcgZeroPolicy = Literal["zero", "exclude"]


def relevance_from_labels(click: np.ndarray, book: np.ndarray) -> np.ndarray:
    click = np.asarray(click)
    book = np.asarray(book)
    rel = np.where(book.astype(bool), 5, np.where(click.astype(bool), 1, 0))
    return rel.astype(np.int8)


def remap_relevance_to_label_index(rel: np.ndarray) -> np.ndarray:
    rel = np.asarray(rel, dtype=np.int64)
    out = np.zeros_like(rel)
    out[rel == 1] = 1
    out[rel == 5] = 2
    return out.astype(np.int8)


def dcg_at_k_single(relevances_in_predicted_order: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances_in_predicted_order, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k_dataframe(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "relevance",
    group_col: str = "srch_id",
    k: int = 5,
    idcg_zero: IdcgZeroPolicy = "zero",
) -> float:
    if df.empty:
        return 0.0
    work = df[[group_col, score_col, label_col]].copy()
    work = work.sort_values([group_col, score_col], ascending=[True, False], kind="stable")
    work["__rank"] = work.groupby(group_col, sort=False).cumcount()
    work_topk = work[work["__rank"] < k].copy()
    work_topk["__discount"] = 1.0 / np.log2(work_topk["__rank"].to_numpy() + 2)
    work_topk["__gain"] = work_topk[label_col].astype(float) * work_topk["__discount"]
    dcg = work_topk.groupby(group_col, sort=False)["__gain"].sum()

    ideal = df[[group_col, label_col]].copy()
    ideal = ideal.sort_values([group_col, label_col], ascending=[True, False], kind="stable")
    ideal["__rank"] = ideal.groupby(group_col, sort=False).cumcount()
    ideal_topk = ideal[ideal["__rank"] < k].copy()
    ideal_topk["__discount"] = 1.0 / np.log2(ideal_topk["__rank"].to_numpy() + 2)
    ideal_topk["__gain"] = ideal_topk[label_col].astype(float) * ideal_topk["__discount"]
    idcg = ideal_topk.groupby(group_col, sort=False)["__gain"].sum()

    aligned = pd.concat([dcg.rename("dcg"), idcg.rename("idcg")], axis=1).fillna(0.0)
    mask_zero = aligned["idcg"] <= 0.0
    if idcg_zero == "zero":
        ndcg = np.where(mask_zero, 0.0, aligned["dcg"] / aligned["idcg"].replace(0.0, np.nan))
        ndcg = np.nan_to_num(ndcg, nan=0.0, posinf=0.0, neginf=0.0)
    elif idcg_zero == "exclude":
        valid = aligned[~mask_zero]
        if valid.empty:
            return 0.0
        ndcg = (valid["dcg"] / valid["idcg"]).to_numpy()
    else:
        raise ValueError(f"Unknown idcg_zero policy: {idcg_zero}")
    return float(np.mean(ndcg))

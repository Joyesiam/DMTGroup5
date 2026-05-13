"""Anchor feature builder: 49 columns.

Composition:
  19 raw numeric columns
  24 raw competitor columns (comp1..8 rate / inv / rate_percent_diff)
   5 categorical id columns (site_id, visitor_location_country_id,
     prop_country_id, prop_id, srch_destination_id)
   1 derived flag (has_visitor_history)

Train-only columns position, click_bool, booking_bool, gross_bookings_usd
are dropped before the model sees the matrix. LightGBM treats the five
id columns as categorical via the categorical_feature argument.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.metrics import relevance_from_labels, remap_relevance_to_label_index

NUMERIC_COLS: list[str] = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
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
]

COMP_COLS: list[str] = (
    [f"comp{i}_rate" for i in range(1, 9)]
    + [f"comp{i}_inv" for i in range(1, 9)]
    + [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
)

CATEGORICAL_COLS: list[str] = [
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "prop_id",
    "srch_destination_id",
]

DERIVED_COLS: list[str] = ["has_visitor_history"]

TRAIN_ONLY_COLS: list[str] = ["position", "click_bool", "booking_bool", "gross_bookings_usd"]


class AnchorFeatures(NamedTuple):
    X: pd.DataFrame
    label_index: np.ndarray
    relevance: np.ndarray
    srch_id: np.ndarray
    feature_columns: list[str]
    categorical_features: list[str]


def expected_feature_columns() -> list[str]:
    return NUMERIC_COLS + COMP_COLS + CATEGORICAL_COLS + DERIVED_COLS


def _add_has_visitor_history(df: pd.DataFrame) -> pd.Series:
    a = df["visitor_hist_starrating"].notna() if "visitor_hist_starrating" in df.columns else False
    b = df["visitor_hist_adr_usd"].notna() if "visitor_hist_adr_usd" in df.columns else False
    return pd.Series((a & b).astype("int8"), index=df.index, name="has_visitor_history")


def build_anchor_features(df: pd.DataFrame, *, has_labels: bool) -> AnchorFeatures:
    if has_labels:
        click = df["click_bool"].astype("Int8").fillna(0).to_numpy().astype(int)
        book = df["booking_bool"].astype("Int8").fillna(0).to_numpy().astype(int)
        rel_raw = relevance_from_labels(click, book)
        label_idx = remap_relevance_to_label_index(rel_raw)
    else:
        rel_raw = np.zeros(len(df), dtype=np.int8)
        label_idx = np.zeros(len(df), dtype=np.int8)

    work = df.drop(columns=[c for c in TRAIN_ONLY_COLS if c in df.columns], errors="ignore").copy()
    work["has_visitor_history"] = _add_has_visitor_history(work).values

    expected = expected_feature_columns()
    missing = [c for c in expected if c not in work.columns]
    if missing:
        raise AssertionError(f"missing anchor columns: {missing}")

    X = work[expected].copy()
    for col in NUMERIC_COLS + COMP_COLS:
        X[col] = X[col].astype("float32")
    for col in CATEGORICAL_COLS:
        if pd.api.types.is_extension_array_dtype(X[col]):
            X[col] = X[col].astype("Int32")
    X["has_visitor_history"] = X["has_visitor_history"].astype("int8")

    feature_columns = list(X.columns)
    categorical_features = [c for c in CATEGORICAL_COLS if c in feature_columns]

    srch_id = df["srch_id"].to_numpy()
    return AnchorFeatures(
        X=X,
        label_index=label_idx,
        relevance=rel_raw,
        srch_id=srch_id,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
    )


def group_sizes_from_srch_id(sorted_srch_id: np.ndarray) -> np.ndarray:
    if sorted_srch_id.size == 0:
        return np.array([], dtype=np.int64)
    s = pd.Series(sorted_srch_id)
    if not s.is_monotonic_increasing:
        raise ValueError("srch_id must be sorted ascending before computing group sizes")
    return s.value_counts(sort=False).reindex(s.unique()).to_numpy(dtype=np.int64)


def sort_for_ranker(
    X: pd.DataFrame,
    y: np.ndarray,
    srch_id: np.ndarray,
    *,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    extra_arrays = extra_arrays or {}
    order = np.argsort(srch_id, kind="stable")
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]
    srch_id_sorted = np.asarray(srch_id)[order]
    extra_sorted = {name: np.asarray(arr)[order] for name, arr in extra_arrays.items()}
    group_sizes = group_sizes_from_srch_id(srch_id_sorted)
    if int(group_sizes.sum()) != len(X_sorted):
        raise AssertionError(
            f"group_sizes.sum()={int(group_sizes.sum())} != len(X)={len(X_sorted)}"
        )
    return X_sorted, y_sorted, srch_id_sorted, group_sizes, extra_sorted

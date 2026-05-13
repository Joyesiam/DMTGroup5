"""LightGBM LambdaMART wrapper for the final pipeline.

Configuration matches the deliverable: Variant A label encoding
({0, 1, 5} -> {0, 1, 2}) with label_gain = [0, 1, 5]. Thread counts are
controlled, never -1. Categorical columns are passed explicitly via the
fit() call.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

LGBM_DEFAULT_PARAMS: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "label_gain": [0, 1, 5],
    "num_leaves": 28,
    "max_depth": 9,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "bagging_fraction": 0.958,
    "feature_fraction": 0.927,
    "min_child_samples": 50,
    "verbose": -1,
    "num_threads": 6,
    "force_col_wise": True,
}


@dataclass
class TrainedRanker:
    booster: lgb.LGBMRanker
    best_iteration: int


def train_lgbm_ranker(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    group_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    group_val: np.ndarray | None = None,
    *,
    seed: int,
    categorical_features: list[str] | None = None,
    n_estimators: int | None = None,
    early_stopping_rounds: int | None = 50,
    params: dict | None = None,
) -> TrainedRanker:
    cfg = dict(LGBM_DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    if n_estimators is not None:
        cfg["n_estimators"] = int(n_estimators)
    cfg["random_state"] = int(seed)

    ranker = lgb.LGBMRanker(**cfg)
    fit_kwargs: dict = {
        "X": X_train,
        "y": y_train,
        "group": group_train,
        "categorical_feature": categorical_features or "auto",
    }
    if X_val is not None and y_val is not None and group_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [group_val]
        if early_stopping_rounds is not None:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=0),
            ]
    ranker.fit(**fit_kwargs)
    best_iter = int(getattr(ranker, "best_iteration_", None) or cfg["n_estimators"])
    gc.collect()
    return TrainedRanker(booster=ranker, best_iteration=best_iter)


def predict(ranker: lgb.LGBMRanker, X: pd.DataFrame) -> np.ndarray:
    return ranker.predict(X, num_iteration=ranker.best_iteration_).astype(np.float32)

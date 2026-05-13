"""Final pipeline CLI for the historical-prior 3-seed LightGBM submission.

Steps:
  1. Load train and test parquet files.
  2. Build the 49-feature anchor on train and test.
  3. Build the 12 leakage-safe historical priors on train (5-fold group
     K-fold) and on test (full-train aggregates). Add to the matrices.
  4. Fit one LightGBM LambdaMART model per seed on the FULL train slice
     with a fixed number of trees (best_iteration learned during the
     80% / 10% / 10% split phase). Score the test set per seed.
  5. Score-average across seeds (rank-equivalent: same monotone ordering
     within each search).
  6. Write the Kaggle submission CSV with header `srch_id,prop_id`,
     ordered from highest score to lowest score within each srch_id.

Reproducibility note: the deliverable uses the locked best_iteration
values discovered on the locked-holdout split phase. Those numbers are
recorded in `artifacts/seed_n_trees.json`.

Threading: num_threads is fixed to 6 for LightGBM. We never pass -1.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features import (
    build_anchor_features,
    sort_for_ranker,
)
from src.io_utils import read_json, read_parquet, sha256_of_file, write_json
from src.metrics import relevance_from_labels
from src.model import predict, train_lgbm_ranker
from src.prior_features import (
    build_all_eval_priors,
    build_all_train_priors,
    prior_feature_columns,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

DEFAULT_SEEDS: list[int] = [42, 123, 456]
DEFAULT_ALPHA: float = 20.0
DEFAULT_N_FOLDS: int = 5
DEFAULT_FOLD_SEED: int = 42


def _attach_priors(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    is_train: bool,
    alpha: float,
    n_folds: int,
    fold_seed: int,
) -> pd.DataFrame:
    if is_train:
        priors = build_all_train_priors(
            train_df.reset_index(drop=True),
            n_folds=n_folds,
            fold_seed=fold_seed,
            alpha=alpha,
        )
    else:
        priors = build_all_eval_priors(
            train_df.reset_index(drop=True),
            eval_df.reset_index(drop=True),
            alpha=alpha,
        )
    if len(priors) != len(eval_df):
        raise AssertionError(
            f"prior length mismatch: priors={len(priors)} eval_df={len(eval_df)}"
        )
    return priors.reset_index(drop=True)


def _add_relevance_column(df: pd.DataFrame) -> pd.DataFrame:
    if "relevance" in df.columns:
        return df
    df = df.copy()
    df["relevance"] = relevance_from_labels(
        df["click_bool"].astype("Int8").fillna(0).to_numpy().astype(int),
        df["booking_bool"].astype("Int8").fillna(0).to_numpy().astype(int),
    )
    return df


def train_and_predict(
    train_path: Path,
    test_path: Path,
    seeds: list[int],
    seed_n_trees_path: Path,
    submission_out: Path,
    alpha: float = DEFAULT_ALPHA,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_seed: int = DEFAULT_FOLD_SEED,
) -> Path:
    print(f"[pipeline] reading train: {train_path}")
    train_df = read_parquet(train_path)
    train_df = _add_relevance_column(train_df)

    print(f"[pipeline] reading test: {test_path}")
    test_df = read_parquet(test_path)

    seed_n_trees = read_json(seed_n_trees_path)
    if any(str(s) not in seed_n_trees for s in seeds):
        missing = [s for s in seeds if str(s) not in seed_n_trees]
        raise FileNotFoundError(
            f"seed_n_trees.json missing seeds {missing}; expected at {seed_n_trees_path}"
        )

    print("[pipeline] building train priors (K-fold leakage-safe)")
    train_priors = _attach_priors(
        train_df, train_df, is_train=True, alpha=alpha, n_folds=n_folds, fold_seed=fold_seed
    )
    print("[pipeline] building test priors (full-train aggregates)")
    test_priors = _attach_priors(
        train_df, test_df, is_train=False, alpha=alpha, n_folds=n_folds, fold_seed=fold_seed
    )

    print("[pipeline] building anchor features for train")
    anchor_train = build_anchor_features(train_df, has_labels=True)
    print("[pipeline] building anchor features for test")
    anchor_test = build_anchor_features(test_df, has_labels=False)

    X_train = pd.concat(
        [anchor_train.X.reset_index(drop=True), train_priors.reset_index(drop=True)], axis=1
    )
    X_test = pd.concat(
        [anchor_test.X.reset_index(drop=True), test_priors.reset_index(drop=True)], axis=1
    )

    expected_extra = prior_feature_columns()
    missing_in_train = [c for c in expected_extra if c not in X_train.columns]
    if missing_in_train:
        raise AssertionError(f"prior columns missing from X_train: {missing_in_train}")
    if list(X_train.columns) != list(X_test.columns):
        raise AssertionError("train/test feature columns disagree")

    X_train_sorted, y_train_sorted, _, group_train, _ = sort_for_ranker(
        X_train, anchor_train.label_index, anchor_train.srch_id
    )
    X_test_sorted, _, srch_id_test_sorted, _, extra_test = sort_for_ranker(
        X_test,
        np.zeros(len(X_test), dtype=np.int8),
        anchor_test.srch_id,
        extra_arrays={"prop_id": test_df["prop_id"].to_numpy()},
    )

    cat_cols = anchor_train.categorical_features
    test_score_acc = np.zeros(len(X_test_sorted), dtype=np.float64)
    for seed in seeds:
        n_trees = int(seed_n_trees[str(seed)])
        print(f"[pipeline] seed={seed} n_trees={n_trees}")
        ranker = train_lgbm_ranker(
            X_train_sorted,
            y_train_sorted,
            group_train,
            seed=seed,
            categorical_features=cat_cols,
            n_estimators=n_trees,
            early_stopping_rounds=None,
        )
        seed_scores = predict(ranker.booster, X_test_sorted).astype(np.float64)
        test_score_acc += seed_scores
        del ranker
        gc.collect()
    test_score_avg = test_score_acc / float(len(seeds))

    sub = pd.DataFrame(
        {
            "srch_id": srch_id_test_sorted,
            "prop_id": extra_test["prop_id"],
            "score": test_score_avg,
        }
    )
    sub = sub.sort_values(["srch_id", "score"], ascending=[True, False], kind="stable")
    sub[["srch_id", "prop_id"]].to_csv(submission_out, index=False)
    digest = sha256_of_file(submission_out)
    meta = {
        "submission_path": str(submission_out),
        "sha256": digest,
        "n_rows": int(len(sub)),
        "n_unique_srch_ids": int(sub["srch_id"].nunique()),
        "seeds": seeds,
        "alpha": alpha,
        "n_folds": n_folds,
        "fold_seed": fold_seed,
        "seed_n_trees": {str(s): int(seed_n_trees[str(s)]) for s in seeds},
    }
    write_json(meta, submission_out.with_suffix(".meta.json"))
    print(f"[pipeline] wrote {submission_out} sha256={digest}")
    return submission_out


def validate_submission(path: Path, expected_n_rows: int | None = None) -> dict:
    df = pd.read_csv(path)
    if list(df.columns) != ["srch_id", "prop_id"]:
        raise AssertionError(f"unexpected columns {list(df.columns)}")
    if df.isnull().any().any():
        raise AssertionError("submission contains NaN")
    if df.duplicated(["srch_id", "prop_id"]).any():
        raise AssertionError("submission contains duplicate (srch_id, prop_id) pairs")
    info = {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_unique_srch_ids": int(df["srch_id"].nunique()),
        "sha256": sha256_of_file(path),
    }
    if expected_n_rows is not None and info["n_rows"] != int(expected_n_rows):
        raise AssertionError(
            f"row count {info['n_rows']} != expected {expected_n_rows}"
        )
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Final pipeline for VU-DM-2026-Group-5")
    p.add_argument("--train", action="store_true", help="train and predict end-to-end")
    p.add_argument("--predict", action="store_true", help="alias for --train (kept for clarity)")
    p.add_argument("--validate", type=Path, default=None, help="validate an existing CSV instead")
    p.add_argument("--train-parquet", type=Path, default=DATA_DIR / "processed" / "train_clean.parquet")
    p.add_argument("--test-parquet", type=Path, default=DATA_DIR / "processed" / "test_clean.parquet")
    p.add_argument("--seed-n-trees", type=Path, default=ARTIFACTS_DIR / "seed_n_trees.json")
    p.add_argument(
        "--submission-out",
        type=Path,
        default=RESULTS_DIR / "submit_final.csv",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = p.parse_args(argv)

    if args.validate is not None:
        info = validate_submission(args.validate)
        print(info)
        return 0

    if not (args.train or args.predict):
        p.print_help()
        return 1

    out = train_and_predict(
        args.train_parquet,
        args.test_parquet,
        seeds=list(args.seeds),
        seed_n_trees_path=args.seed_n_trees,
        submission_out=args.submission_out,
    )
    info = validate_submission(out)
    print(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

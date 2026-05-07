"""Group-aware train/val/holdout split on srch_id.

The holdout is locked: same srch_id set across seeds. Only the train/val
rotation changes per seed. The locked split JSON used in the deliverable
has sha256 prefix bc8ea6f6 with sizes 159,835 / 19,980 / 19,980.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.io_utils import read_json, write_json


def make_group_split(
    srch_ids: Iterable[int],
    val_frac: float = 0.10,
    holdout_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[int]]:
    unique_ids = np.unique(np.asarray(list(srch_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_ids)
    n_total = len(unique_ids)
    n_holdout = int(round(n_total * holdout_frac))
    n_val = int(round(n_total * val_frac))
    holdout_ids = unique_ids[:n_holdout]
    val_ids = unique_ids[n_holdout : n_holdout + n_val]
    train_ids = unique_ids[n_holdout + n_val :]
    return {
        "train": [int(x) for x in train_ids.tolist()],
        "val": [int(x) for x in val_ids.tolist()],
        "holdout": [int(x) for x in holdout_ids.tolist()],
    }


def load_locked_holdout(holdout_path: Path | str) -> dict[str, set]:
    raw = read_json(holdout_path)
    return {k: set(v) for k, v in raw.items()}


def assert_split_disjoint(split: dict[str, list[int] | set]) -> None:
    train = set(split["train"])
    val = set(split["val"])
    holdout = set(split["holdout"])
    if train & val:
        raise AssertionError("train and val share srch_ids")
    if train & holdout:
        raise AssertionError("train and holdout share srch_ids")
    if val & holdout:
        raise AssertionError("val and holdout share srch_ids")

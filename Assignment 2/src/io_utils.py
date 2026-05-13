"""Parquet, JSON, and hashing helpers for the final pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path | str, compression: str = "snappy") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression=compression, index=False)


def read_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def write_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def sha256_of_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

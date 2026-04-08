"""
Shared utility functions for DMT Assignment 1 (Advanced).
Mental health smartphone dataset - mood prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Ensure figures directory exists
FIGURES_DIR.mkdir(exist_ok=True)

# Dataset variable categories
MOOD_VARS = ["mood"]
SELF_REPORT_VARS = ["circumplex.arousal", "circumplex.valence"]
SENSOR_VARS = ["activity", "screen", "call", "sms"]
APP_VARS = [
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel", "appCat.unknown",
    "appCat.utilities", "appCat.weather",
]
ALL_VARS = MOOD_VARS + SELF_REPORT_VARS + SENSOR_VARS + APP_VARS


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset and parse timestamps."""
    df = pd.read_csv(DATA_DIR / "dataset_mood_smartphone.csv", index_col=0)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    return df


def pivot_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format data to daily wide format.
    Aggregates per patient per day:
    - mood, arousal, valence, activity: daily mean
    - screen, appCat.*: daily sum
    - call, sms: daily count
    """
    mean_vars = MOOD_VARS + SELF_REPORT_VARS + ["activity"]
    sum_vars = ["screen"] + APP_VARS
    count_vars = ["call", "sms"]

    daily_frames = []

    for agg_func, var_list in [("mean", mean_vars), ("sum", sum_vars), ("count", count_vars)]:
        subset = df[df["variable"].isin(var_list)]
        if subset.empty:
            continue
        pivoted = (
            subset.groupby(["id", "date", "variable"])["value"]
            .agg(agg_func)
            .reset_index()
            .pivot_table(index=["id", "date"], columns="variable", values="value")
            .reset_index()
        )
        daily_frames.append(pivoted.set_index(["id", "date"]))

    daily = pd.concat(daily_frames, axis=1).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["id", "date"]).reset_index(drop=True)


def save_figure(fig, name: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to the figures directory."""
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / name}.png")

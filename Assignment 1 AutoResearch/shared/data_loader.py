"""
Data loading, cleaning, and daily aggregation pipeline.
Fully parameterized: outlier method, imputation method, gap handling.
Every choice is a parameter so iterations can compare strategies.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import (
    RAW_DATA_FILE, MOOD_VARS, SELF_REPORT_VARS, SENSOR_VARS, APP_VARS,
    ALL_VARS, MEAN_VARS, SUM_VARS, COUNT_VARS, ID_COL, DATE_COL
)


def load_raw_data() -> pd.DataFrame:
    """Load the raw long-format dataset and parse timestamps."""
    df = pd.read_csv(RAW_DATA_FILE, index_col=0)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    return df


def pivot_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format data to daily wide format.
    Aggregation rules:
    - mood, arousal, valence, activity: daily mean
    - screen, appCat.*: daily sum (duration)
    - call, sms: daily count
    """
    daily_frames = []
    for agg_func, var_list in [("mean", MEAN_VARS), ("sum", SUM_VARS), ("count", COUNT_VARS)]:
        subset = df[df["variable"].isin(var_list)]
        if subset.empty:
            continue
        pivoted = (
            subset.groupby([ID_COL, "date", "variable"])["value"]
            .agg(agg_func)
            .reset_index()
            .pivot_table(index=[ID_COL, "date"], columns="variable", values="value")
            .reset_index()
        )
        daily_frames.append(pivoted.set_index([ID_COL, "date"]))

    daily = pd.concat(daily_frames, axis=1).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values([ID_COL, "date"]).reset_index(drop=True)


def fill_date_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    """Ensure continuous date range per patient (fill missing dates with NaN rows)."""
    frames = []
    for pid, group in daily.groupby(ID_COL):
        date_range = pd.date_range(group[DATE_COL].min(), group[DATE_COL].max(), freq="D")
        full = pd.DataFrame({DATE_COL: date_range, ID_COL: pid})
        merged = full.merge(group, on=[ID_COL, DATE_COL], how="left")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


def add_morning_evening_mood(daily: pd.DataFrame) -> pd.DataFrame:
    """Add morning and evening mood as separate columns from raw data."""
    raw = load_raw_data()
    mood_raw = raw[raw["variable"] == "mood"].copy()
    mood_raw["hour"] = mood_raw["time"].dt.hour

    # Morning: 8-12, Evening: 18-22
    morning = mood_raw[mood_raw["hour"].between(8, 12)].groupby([ID_COL, "date"])["value"].mean()
    evening = mood_raw[mood_raw["hour"].between(18, 22)].groupby([ID_COL, "date"])["value"].mean()

    morning_df = morning.reset_index().rename(columns={"value": "mood_morning"})
    evening_df = evening.reset_index().rename(columns={"value": "mood_evening"})

    morning_df["date"] = pd.to_datetime(morning_df["date"])
    evening_df["date"] = pd.to_datetime(evening_df["date"])

    df = daily.merge(morning_df, on=[ID_COL, DATE_COL], how="left")
    df = df.merge(evening_df, on=[ID_COL, DATE_COL], how="left")

    # Mood slope within day (evening - morning)
    df["mood_intraday_slope"] = df["mood_evening"] - df["mood_morning"]

    print(f"    Added mood_morning, mood_evening, mood_intraday_slope")
    return df


SPARSE_APP_COLS = [
    "appCat.weather", "appCat.game", "appCat.finance",
   "appCat.office", "appCat.travel", "appCat.utilities",
]


def drop_sparse_apps(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop app categories with >80% missing values."""
    cols_to_drop = [c for c in SPARSE_APP_COLS if c in daily.columns]
    df = daily.drop(columns=cols_to_drop)
    print(f"    Dropped {len(cols_to_drop)} sparse app columns (>80% missing)")
    return df


# === OUTLIER REMOVAL (parameterized) ===

def remove_domain_outliers(daily: pd.DataFrame) -> pd.DataFrame:
    """Remove values outside domain-valid ranges. Always applied."""
    df = daily.copy()
    if "mood" in df.columns:
        df.loc[~df["mood"].between(1, 10), "mood"] = np.nan
    for col in ["circumplex.arousal", "circumplex.valence"]:
        if col in df.columns:
            df.loc[~df[col].between(-2, 2), col] = np.nan
    if "activity" in df.columns:
        df.loc[~df["activity"].between(0, 1), "activity"] = np.nan
    non_negative_cols = ["screen", "call", "sms"] + APP_VARS
    for col in non_negative_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    return df


def remove_iqr_outliers(daily: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """Replace IQR-based outliers with NaN for sensor/app columns."""
    df = daily.copy()
    iqr_cols = SENSOR_VARS + APP_VARS
    n_removed = 0
    for col in iqr_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = ~df[col].between(lower, upper) & df[col].notna()
        n_removed += mask.sum()
        df.loc[mask, col] = np.nan
    print(f"    IQR outliers removed (multiplier={multiplier}): {n_removed}")
    return df


def remove_zscore_outliers(daily: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Replace z-score outliers with NaN for sensor/app columns."""
    df = daily.copy()
    iqr_cols = SENSOR_VARS + APP_VARS
    n_removed = 0
    for col in iqr_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        mean, std = vals.mean(), vals.std()
        if std == 0:
            continue
        mask = (((df[col] - mean) / std).abs() > threshold) & df[col].notna()
        n_removed += mask.sum()
        df.loc[mask, col] = np.nan
    print(f"    Z-score outliers removed (threshold={threshold}): {n_removed}")
    return df


# === IMPUTATION (parameterized) ===

def impute_forward_fill(daily: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill per patient, then backward-fill remaining."""
    df = daily.copy()
    feature_cols = [c for c in df.columns if c not in [ID_COL, DATE_COL]]
    df[feature_cols] = df.groupby(ID_COL)[feature_cols].transform(
        lambda x: x.ffill().bfill()
    )
    return df


def impute_linear(daily: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation per patient (time-aware)."""
    df = daily.copy()
    feature_cols = [c for c in df.columns if c not in [ID_COL, DATE_COL]]
    df[feature_cols] = df.groupby(ID_COL)[feature_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    return df


def impute_knn(daily: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """KNN imputation using k nearest temporal neighbors per patient."""
    from sklearn.impute import KNNImputer
    df = daily.copy()
    feature_cols = [c for c in df.columns if c not in [ID_COL, DATE_COL]]

    # Pre-fill all-NaN columns with 0 (KNNImputer drops them otherwise)
    for col in feature_cols:
        if df[col].isna().all():
            df[col] = 0

    imputer = KNNImputer(n_neighbors=min(k, 3))
    frames = []
    for pid, group in df.groupby(ID_COL):
        g = group.copy()
        data = g[feature_cols].values.copy()
        # Fill columns that are all-NaN for this patient with 0
        all_nan_cols = np.isnan(data).all(axis=0)
        data[:, all_nan_cols] = 0
        # Only impute if there are non-NaN values
        if np.isnan(data).any():
            try:
                imputed = imputer.fit_transform(data)
                for j, col in enumerate(feature_cols):
                    g[col] = imputed[:, j]
            except Exception:
                # Fallback to forward fill if KNN fails for this patient
                for col in feature_cols:
                    g[col] = g[col].ffill().bfill().fillna(0)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


def impute_hybrid(daily: pd.DataFrame) -> pd.DataFrame:
    """Hybrid imputation: linear interpolation for continuous vars, ffill for app categories."""
    df = daily.copy()
    continuous_cols = [c for c in MOOD_VARS + SELF_REPORT_VARS + SENSOR_VARS if c in df.columns]
    categorical_cols = [c for c in APP_VARS if c in df.columns]

    # Linear interpolation for continuous variables
    if continuous_cols:
        df[continuous_cols] = df.groupby(ID_COL)[continuous_cols].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
    # Forward fill for app category durations
    if categorical_cols:
        df[categorical_cols] = df.groupby(ID_COL)[categorical_cols].transform(
            lambda x: x.ffill().bfill()
        )
    return df


# === GAP HANDLING ===

def mark_prolonged_gaps(daily: pd.DataFrame, max_gap_days: int = None) -> pd.DataFrame:
    """
    If max_gap_days is set, mark rows that fall within prolonged gaps
    (>max_gap_days consecutive NaN for mood) as unreliable.
    These rows will have mood=NaN and should be excluded from training instances.
    """
    if max_gap_days is None:
        return daily

    df = daily.copy()
    n_excluded = 0
    for pid, group in df.groupby(ID_COL):
        mood_null = group["mood"].isna() if "mood" in group.columns else pd.Series(False, index=group.index)
        # Find consecutive NaN streaks
        streak = 0
        gap_indices = []
        for idx in group.index:
            if mood_null.loc[idx]:
                streak += 1
                gap_indices.append(idx)
            else:
                if streak > max_gap_days:
                    # Mark the gap rows
                    for gi in gap_indices:
                        df.loc[gi, "mood"] = np.nan
                    n_excluded += len(gap_indices)
                streak = 0
                gap_indices = []
        # Check final streak
        if streak > max_gap_days:
            for gi in gap_indices:
                df.loc[gi, "mood"] = np.nan
            n_excluded += len(gap_indices)

    if n_excluded > 0:
        print(f"    Prolonged gaps (>{max_gap_days} days): {n_excluded} rows excluded")
    return df


# === MAIN PARAMETERIZED PIPELINE ===

def load_and_clean(
    outlier_method: str = "iqr",
    iqr_multiplier: float = 3.0,
    imputation_method: str = "ffill",
    max_gap_days: int = None,
    log_transform_durations: bool = False,
    add_morning_evening: bool = False,
    drop_sparse: bool = False,
    save_path: Path = None,
) -> pd.DataFrame:
    """
    Full parameterized pipeline: raw -> daily -> cleaned -> imputed.

    Parameters
    ----------
    outlier_method : "iqr", "domain_only", "zscore"
    iqr_multiplier : float, IQR multiplier (only for "iqr" method)
    imputation_method : "ffill", "linear", "knn"
    max_gap_days : int or None, exclude segments with gaps longer than this
    log_transform_durations : bool, apply log1p to duration variables
    save_path : Path or None, save cleaned CSV here
    """
    print("  Phase 1: Data Cleaning")
    raw = load_raw_data()
    daily = pivot_to_daily(raw)
    daily = fill_date_gaps(daily)
    print(f"    Raw daily: {daily.shape[0]} rows, {daily.shape[1]} columns, {daily[ID_COL].nunique()} patients")

    # Outlier removal (domain-based always applied first)
    daily = remove_domain_outliers(daily)
    if outlier_method == "iqr":
        daily = remove_iqr_outliers(daily, multiplier=iqr_multiplier)
    elif outlier_method == "zscore":
        daily = remove_zscore_outliers(daily, threshold=iqr_multiplier)
    elif outlier_method == "domain_only":
        print("    Outlier method: domain-only (no statistical removal)")
    else:
        raise ValueError(f"Unknown outlier_method: {outlier_method}")

    # Handle prolonged gaps (before imputation)
    daily = mark_prolonged_gaps(daily, max_gap_days=max_gap_days)

    # Imputation
    if imputation_method == "ffill":
        daily = impute_forward_fill(daily)
        print("    Imputation: forward fill + backward fill")
    elif imputation_method == "linear":
        daily = impute_linear(daily)
        print("    Imputation: linear interpolation")
    elif imputation_method == "knn":
        daily = impute_knn(daily, k=5)
        print("    Imputation: KNN (k=5)")
    elif imputation_method == "hybrid":
        daily = impute_hybrid(daily)
        print("    Imputation: hybrid (linear for continuous, ffill for categorical)")
    else:
        raise ValueError(f"Unknown imputation_method: {imputation_method}")

    # Log-transform duration variables
    if log_transform_durations:
        duration_cols = ["screen"] + APP_VARS
        for col in duration_cols:
            if col in daily.columns:
                daily[col] = np.log1p(daily[col].clip(lower=0))
        print("    Applied log1p to duration variables")

    # Add morning/evening mood (before filling NaN)
    if add_morning_evening:
        daily = add_morning_evening_mood(daily)

    # Drop sparse app categories
    if drop_sparse:
        daily = drop_sparse_apps(daily)

    # Fill remaining NaN with 0 (sparse app categories)
    feature_cols = [c for c in daily.columns if c not in [ID_COL, DATE_COL]]
    daily[feature_cols] = daily[feature_cols].fillna(0)

    print(f"    Final cleaned: {daily.shape[0]} rows, {daily.shape[1]} cols, {daily[ID_COL].nunique()} patients")

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(save_path, index=False)
        print(f"    Saved: {save_path}")

    return daily


# === v6 DATA CLEANING EXTENSIONS ===

def group_app_categories(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Iter 107: Group 12 app categories into 4 semantic super-categories.
    social_communication = communication + social + builtin
    entertainment_leisure = entertainment + game + travel + weather
    productivity_work = office + finance + utilities
    miscellaneous = other + unknown
    """
    df = daily.copy()
    groups = {
        "appCat.social_communication": ["appCat.communication", "appCat.social", "appCat.builtin"],
        "appCat.entertainment_leisure": ["appCat.entertainment", "appCat.game", "appCat.travel", "appCat.weather"],
        "appCat.productivity_work": ["appCat.office", "appCat.finance", "appCat.utilities"],
        "appCat.miscellaneous": ["appCat.other", "appCat.unknown"],
    }
    for new_col, source_cols in groups.items():
        existing = [c for c in source_cols if c in df.columns]
        if existing:
            df[new_col] = df[existing].sum(axis=1)
    # Drop original app columns
    orig_app_cols = [c for c in APP_VARS if c in df.columns]
    df = df.drop(columns=orig_app_cols)
    print(f"    Grouped 12 app categories into 4 super-categories")
    return df

def merge_other_unknown(daily: pd.DataFrame) -> pd.DataFrame:
    """Merge appCat.unknown into appCat.other and drop appCat.unknown."""
    df = daily.copy()
    if "appCat.unknown" in df.columns:
        df["appCat.other"] = df["appCat.other"].fillna(0) + df["appCat.unknown"].fillna(0)
        df = df.drop(columns=["appCat.unknown"])
        print("    Merged appCat.unknown into appCat.other")
    return df


def density_based_sparse_merge(daily: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """
    Iter 108: Per-patient density-based sparse merging.
    Any app column with <threshold non-zero rows gets merged into appCat.other.
    """
    df = daily.copy()
    app_cols = [c for c in APP_VARS if c in df.columns]
    n_merged = 0
    for pid, group in df.groupby(ID_COL):
        for col in app_cols:
            if col == "appCat.other":
                continue
            density = (group[col] > 0).mean()
            if density < threshold:
                df.loc[group.index, "appCat.other"] = df.loc[group.index, "appCat.other"] + df.loc[group.index, col]
                df.loc[group.index, col] = 0
                n_merged += 1
    print(f"    Density-based merge: {n_merged} patient-column pairs merged (threshold={threshold})")
    return df


def winsorize_percentile(daily: pd.DataFrame, lower: float = 5, upper: float = 95) -> pd.DataFrame:
    """
    Iter 109: Winsorize at lower/upper percentile for sensor/app columns.
    Clips instead of removing.
    """
    df = daily.copy()
    clip_cols = SENSOR_VARS + APP_VARS
    n_clipped = 0
    for col in clip_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        lo = np.percentile(vals, lower)
        hi = np.percentile(vals, upper)
        mask = df[col].notna()
        before = df.loc[mask, col].copy()
        df.loc[mask, col] = df.loc[mask, col].clip(lo, hi)
        n_clipped += (before != df.loc[mask, col]).sum()
    print(f"    Winsorized at {lower}th/{upper}th percentile: {n_clipped} values clipped")
    return df


def delete_long_mood_gaps(daily: pd.DataFrame, max_consecutive: int = 2) -> pd.DataFrame:
    """
    Iter 110: Delete stretches of >max_consecutive consecutive missing mood days.
    More aggressive than mark_prolonged_gaps - actually removes rows.
    """
    df = daily.copy()
    rows_to_drop = []
    for pid, group in df.groupby(ID_COL):
        mood_null = group["mood"].isna() if "mood" in group.columns else pd.Series(False, index=group.index)
        streak_indices = []
        for idx in group.index:
            if mood_null.loc[idx]:
                streak_indices.append(idx)
            else:
                if len(streak_indices) > max_consecutive:
                    rows_to_drop.extend(streak_indices)
                streak_indices = []
        if len(streak_indices) > max_consecutive:
            rows_to_drop.extend(streak_indices)
    if rows_to_drop:
        df = df.drop(index=rows_to_drop).reset_index(drop=True)
    print(f"    Deleted {len(rows_to_drop)} rows in long mood gaps (>{max_consecutive} consecutive)")
    return df


def cap_app_durations(daily: pd.DataFrame, max_seconds: float = 10800) -> pd.DataFrame:
    """
    Iter 111: Cap app durations at max_seconds (default 3 hours = 10800s).
    Domain-based capping.
    """
    df = daily.copy()
    n_capped = 0
    for col in APP_VARS:
        if col not in df.columns:
            continue
        mask = df[col] > max_seconds
        n_capped += mask.sum()
        df.loc[mask, col] = max_seconds
    print(f"    Capped app durations at {max_seconds}s: {n_capped} values capped")
    return df


def remove_all_negatives(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Iter 112: Remove all negative values except circumplex arousal/valence.
    More thorough than domain-only which only checks specific columns.
    """
    df = daily.copy()
    exempt = ["circumplex.arousal", "circumplex.valence"]
    n_removed = 0
    for col in df.columns:
        if col in [ID_COL, DATE_COL] or col in exempt:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            mask = df[col] < 0
            n_removed += mask.sum()
            df.loc[mask, col] = np.nan
    print(f"    Removed {n_removed} negative values (except circumplex)")
    return df


def conditional_zero_fill(daily: pd.DataFrame, min_active_cols: int = 4) -> pd.DataFrame:
    """
    Iter 113: For appCat/call/sms NaN values, only fill with 0 if the patient
    was active that day (>= min_active_cols other columns non-null).
    If patient wasn't using phone, leave as NaN (truly missing).
    """
    df = daily.copy()
    zero_fill_cols = [c for c in APP_VARS + ["call", "sms"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in [ID_COL, DATE_COL] + zero_fill_cols]
    n_filled = 0
    for idx in df.index:
        active_count = df.loc[idx, other_cols].notna().sum()
        if active_count >= min_active_cols:
            for col in zero_fill_cols:
                if pd.isna(df.loc[idx, col]):
                    df.loc[idx, col] = 0
                    n_filled += 1
    print(f"    Conditional zero-fill: {n_filled} values filled (min_active={min_active_cols})")
    return df


def load_and_clean_v6(
    outlier_method: str = "iqr",
    iqr_multiplier: float = 3.0,
    imputation_method: str = "ffill",
    max_gap_days: int = None,
    log_transform_durations: bool = False,
    add_morning_evening: bool = False,
    drop_sparse: bool = False,
    # v6 cleaning options
    app_grouping: bool = False,
    density_merge: bool = False,
    density_threshold: float = 0.25,
    winsorize: bool = False,
    winsorize_lower: float = 5,
    winsorize_upper: float = 95,
    delete_mood_gaps: bool = False,
    max_consecutive_gaps: int = 2,
    cap_app_hours: bool = False,
    cap_max_seconds: float = 10800,
    remove_negatives: bool = False,
    conditional_fill: bool = False,
    conditional_fill_min: int = 4,
    merge_other_unknown: bool = False,
    save_path: Path = None,
) -> pd.DataFrame:
    """
    Extended v6 pipeline with all research-driven cleaning options.
    """
    print("  Phase 1: Data Cleaning (v6)")
    raw = load_raw_data()
    daily = pivot_to_daily(raw)
    daily = fill_date_gaps(daily)
    print(f"    Raw daily: {daily.shape[0]} rows, {daily.shape[1]} columns, {daily[ID_COL].nunique()} patients")

    # v6: Remove all negatives (iter 112) -- before other cleaning
    if remove_negatives:
        daily = remove_all_negatives(daily)

    # Outlier removal
    daily = remove_domain_outliers(daily)
    if winsorize:
        daily = winsorize_percentile(daily, winsorize_lower, winsorize_upper)
    elif outlier_method == "iqr":
        daily = remove_iqr_outliers(daily, multiplier=iqr_multiplier)
    elif outlier_method == "zscore":
        daily = remove_zscore_outliers(daily, threshold=iqr_multiplier)
    elif outlier_method == "domain_only":
        print("    Outlier method: domain-only (no statistical removal)")

    # v6: Cap app durations (iter 111)
    if cap_app_hours:
        daily = cap_app_durations(daily, max_seconds=cap_max_seconds)

    # v6: Delete long mood gaps (iter 110)
    if delete_mood_gaps:
        daily = delete_long_mood_gaps(daily, max_consecutive=max_consecutive_gaps)

    # Handle prolonged gaps
    daily = mark_prolonged_gaps(daily, max_gap_days=max_gap_days)

    # v6: Conditional zero-fill before imputation (iter 113)
    if conditional_fill:
        daily = conditional_zero_fill(daily, min_active_cols=conditional_fill_min)

    # Imputation
    if imputation_method == "ffill":
        daily = impute_forward_fill(daily)
        print("    Imputation: forward fill + backward fill")
    elif imputation_method == "linear":
        daily = impute_linear(daily)
        print("    Imputation: linear interpolation")
    elif imputation_method == "knn":
        daily = impute_knn(daily, k=5)
        print("    Imputation: KNN (k=5)")
    elif imputation_method == "hybrid":
        daily = impute_hybrid(daily)
        print("    Imputation: hybrid (linear for continuous, ffill for categorical)")

    # Log-transform duration variables
    if log_transform_durations:
        duration_cols = ["screen"] + APP_VARS
        for col in duration_cols:
            if col in daily.columns:
                daily[col] = np.log1p(daily[col].clip(lower=0))
        print("    Applied log1p to duration variables")

    # Add morning/evening mood
    if add_morning_evening:
        daily = add_morning_evening_mood(daily)

    # v6: App category grouping (iter 107) -- after imputation, before drop_sparse
    if app_grouping:
        daily = group_app_categories(daily)

    # v6: Density-based sparse merge (iter 108)
    if density_merge:
        daily = density_based_sparse_merge(daily, threshold=density_threshold)


    # in the function body, after imputation:
    if merge_other_unknown:
        daily = merge_other_unknown(daily)

    # Drop sparse app categories
    if drop_sparse and not app_grouping:
        daily = drop_sparse_apps(daily)

    # Fill remaining NaN with 0
    feature_cols = [c for c in daily.columns if c not in [ID_COL, DATE_COL]]
    daily[feature_cols] = daily[feature_cols].fillna(0)

    print(f"    Final cleaned: {daily.shape[0]} rows, {daily.shape[1]} cols, {daily[ID_COL].nunique()} patients")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(save_path, index=False)
        print(f"    Saved: {save_path}")

    return daily


def get_first_last_mood(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Iter 117: Extract first and last mood report of each day from raw data.
    """
    raw = load_raw_data()
    mood_raw = raw[raw["variable"] == "mood"].copy()
    mood_raw = mood_raw.sort_values([ID_COL, "time"])

    first = mood_raw.groupby([ID_COL, "date"]).first()["value"].reset_index().rename(columns={"value": "mood_first_daily"})
    last = mood_raw.groupby([ID_COL, "date"]).last()["value"].reset_index().rename(columns={"value": "mood_last_daily"})

    first["date"] = pd.to_datetime(first["date"])
    last["date"] = pd.to_datetime(last["date"])

    df = daily.merge(first, on=[ID_COL, DATE_COL], how="left")
    df = df.merge(last, on=[ID_COL, DATE_COL], how="left")
    print(f"    Added mood_first_daily, mood_last_daily")
    return df


def get_bed_wake_times(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Iter 116: Extract bed time, wake-up time, and sleep duration from raw timestamps.
    bed_time: max hour in [18,24) union [0,5) with +24 wrapping
    wakeup_time: min hour in [5,13)
    sleep_duration: derived from consecutive days
    """
    raw = load_raw_data()
    raw = raw.sort_values([ID_COL, "time"])
    raw["hour"] = raw["time"].dt.hour

    bed_times = []
    wake_times = []

    for pid, group in raw.groupby(ID_COL):
        for date, day_group in group.groupby("date"):
            hours = day_group["hour"].values
            # Bed time: latest activity in evening/night
            evening_hours = hours[(hours >= 18) | (hours < 5)]
            if len(evening_hours) > 0:
                # Wrap hours < 5 to 24+
                wrapped = np.where(evening_hours < 5, evening_hours + 24, evening_hours)
                bed_time = float(wrapped.max())
            else:
                bed_time = np.nan

            # Wake time: earliest activity in morning
            morning_hours = hours[(hours >= 5) & (hours < 13)]
            if len(morning_hours) > 0:
                wake_time = float(morning_hours.min())
            else:
                wake_time = np.nan

            bed_times.append({ID_COL: pid, DATE_COL: pd.Timestamp(date), "bed_time": bed_time})
            wake_times.append({ID_COL: pid, DATE_COL: pd.Timestamp(date), "wakeup_time": wake_time})

    bed_df = pd.DataFrame(bed_times)
    wake_df = pd.DataFrame(wake_times)

    df = daily.merge(bed_df, on=[ID_COL, DATE_COL], how="left")
    df = df.merge(wake_df, on=[ID_COL, DATE_COL], how="left")

    # Sleep duration: (24 - bed_time[prev]) + wakeup_time[current]
    df = df.sort_values([ID_COL, DATE_COL])
    df["sleep_duration"] = np.nan
    for pid, group in df.groupby(ID_COL):
        idx = group.index
        for j in range(1, len(idx)):
            prev_bed = df.loc[idx[j - 1], "bed_time"]
            curr_wake = df.loc[idx[j], "wakeup_time"]
            if pd.notna(prev_bed) and pd.notna(curr_wake) and prev_bed >= 15:
                duration = (24 - prev_bed) + curr_wake
                if 0 < duration < 16:
                    df.loc[idx[j], "sleep_duration"] = duration

    print(f"    Added bed_time, wakeup_time, sleep_duration")
    return df


def get_night_day_split(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Iter 126: Separate daytime (7am-10pm) and nighttime (10pm-7am) for screen/activity.
    """
    raw = load_raw_data()
    raw["hour"] = raw["time"].dt.hour

    for var_name in ["screen", "activity"]:
        var_data = raw[raw["variable"] == var_name].copy()
        if var_data.empty:
            continue
        day_data = var_data[(var_data["hour"] >= 7) & (var_data["hour"] < 22)]
        night_data = var_data[(var_data["hour"] < 7) | (var_data["hour"] >= 22)]

        agg_func = "sum" if var_name == "screen" else "mean"
        day_agg = day_data.groupby([ID_COL, "date"])["value"].agg(agg_func).reset_index()
        night_agg = night_data.groupby([ID_COL, "date"])["value"].agg(agg_func).reset_index()

        day_agg = day_agg.rename(columns={"value": f"{var_name}_day"})
        night_agg = night_agg.rename(columns={"value": f"{var_name}_night"})

        day_agg["date"] = pd.to_datetime(day_agg["date"])
        night_agg["date"] = pd.to_datetime(night_agg["date"])

        daily = daily.merge(day_agg, on=[ID_COL, DATE_COL], how="left")
        daily = daily.merge(night_agg, on=[ID_COL, DATE_COL], how="left")

    print(f"    Added screen_day, screen_night, activity_day, activity_night")
    return daily


# === SPLIT STRATEGIES (parameterized) ===

def get_temporal_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Chronological split by global date cutoff."""
    dates = sorted(df[DATE_COL].unique())
    cutoff_idx = int(len(dates) * (1 - test_fraction))
    cutoff_date = dates[cutoff_idx]
    train = df[df[DATE_COL] < cutoff_date].copy()
    test = df[df[DATE_COL] >= cutoff_date].copy()
    return train, test


def get_leave_patients_out_split(df: pd.DataFrame, n_holdout: int = 5, seed: int = 42):
    """Hold out n complete patients as test set."""
    rng = np.random.RandomState(seed)
    patients = df[ID_COL].unique()
    holdout = rng.choice(patients, size=n_holdout, replace=False)
    train = df[~df[ID_COL].isin(holdout)].copy()
    test = df[df[ID_COL].isin(holdout)].copy()
    print(f"    Leave-patients-out: {n_holdout} patients held out: {list(holdout)}")
    return train, test


def get_sliding_window_splits(df: pd.DataFrame, window_days: int = 14, step_days: int = 7):
    """
    Generate multiple (train, test) splits using a sliding test window.
    Returns list of (train_df, test_df) tuples.
    """
    dates = sorted(df[DATE_COL].unique())
    splits = []
    # Start test window after first 60% of dates
    start_idx = int(len(dates) * 0.6)
    for i in range(start_idx, len(dates) - window_days, step_days):
        test_start = dates[i]
        test_end = dates[min(i + window_days, len(dates) - 1)]
        train = df[df[DATE_COL] < test_start].copy()
        test = df[(df[DATE_COL] >= test_start) & (df[DATE_COL] <= test_end)].copy()
        if len(test) >= 10 and len(train) >= 100:
            splits.append((train, test))
    print(f"    Sliding window: {len(splits)} splits (window={window_days}d, step={step_days}d)")
    return splits


def get_split(df, method="chronological", test_fraction=0.2, n_holdout_patients=5, seed=42):
    """Unified split interface. Returns (train, test) or list of (train, test) for sliding window."""
    if method == "chronological":
        return get_temporal_split(df, test_fraction)
    elif method == "leave_patients_out":
        return get_leave_patients_out_split(df, n_holdout_patients, seed)
    elif method == "sliding_window":
        return get_sliding_window_splits(df)
    else:
        raise ValueError(f"Unknown split method: {method}")

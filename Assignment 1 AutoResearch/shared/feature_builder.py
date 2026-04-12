"""
Parameterized feature engineering pipeline.
Builds instance-based dataset from daily time-series using sliding windows.
All choices are parameters so iterations can compare strategies.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from pathlib import Path
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import ID_COL, DATE_COL, TARGET_COL, ALL_VARS, N_LAGS, DEFAULT_WINDOW_SIZE, APP_VARS


def _compute_trend(series):
    """Linear slope over a window (trend direction)."""
    if len(series) < 2 or series.isna().all():
        return 0.0
    x = np.arange(len(series))
    valid = ~series.isna()
    if valid.sum() < 2:
        return 0.0
    slope, _, _, _, _ = stats.linregress(x[valid], series[valid])
    return slope


def _compute_skew(series):
    """Skewness of values in window."""
    vals = series.dropna()
    if len(vals) < 3:
        return 0.0
    return float(vals.skew())


def _compute_kurtosis(series):
    """Kurtosis of values in window."""
    vals = series.dropna()
    if len(vals) < 4:
        return 0.0
    return float(vals.kurtosis())


# Map of aggregation function names to callables
AGG_FUNCTIONS = {
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "std": lambda s: s.std(),
    "min": lambda s: s.min(),
    "max": lambda s: s.max(),
    "trend": _compute_trend,
    "skew": _compute_skew,
    "kurtosis": _compute_kurtosis,
}


def build_features(
    daily_df: pd.DataFrame,
    window_sizes: list = None,
    n_lags: int = N_LAGS,
    agg_functions: list = None,
    include_interactions: bool = False,
    include_volatility: bool = False,
    include_momentum: bool = False,
    include_lagged_valence: bool = False,
    include_mood_cluster: bool = False,
    include_study_day: bool = False,
    include_weekend_distance: bool = False,
    include_ema: bool = False,
    include_day_changes: bool = False,
    include_ratios: bool = False,
    include_autocorrelation: bool = False,
    ema_weighted_agg: bool = False,
    include_tomorrow_phone: bool = False,
    patient_normalize: bool = False,
    log_transform_before_agg: bool = False,
    predict_mood_change: bool = False,
    # v6 feature flags
    include_emotion_geometry: bool = False,
    include_circumplex_quadrant: bool = False,
    include_short_volatility: bool = False,
    include_ewm_all: bool = False,
    include_adaptive_direction: bool = False,
    include_app_diversity: bool = False,
    include_productive_ratio: bool = False,
    include_app_entropy: bool = False,
    include_rmssd: bool = False,
    include_cv_agg: bool = False,
    include_missingness_flag: bool = False,
    save_path: Path = None,
) -> pd.DataFrame:
    """
    Build instance-based dataset from daily data using sliding windows.

    Parameters
    ----------
    daily_df : DataFrame with columns [id, date, mood, ...features...]
    window_sizes : list of ints, e.g. [7]. Default: [DEFAULT_WINDOW_SIZE]
    n_lags : number of mood lag features
    agg_functions : list of str, e.g. ["mean", "std", "min", "max", "trend"]
    include_interactions : add interaction features
    include_volatility : add mood volatility features
    patient_normalize : z-score normalize features per patient
    log_transform_before_agg : apply log1p to duration vars before window aggregation
    save_path : Path to save CSV (train or test)
    """
    print("  Phase 2: Feature Engineering")

    if window_sizes is None:
        window_sizes = [DEFAULT_WINDOW_SIZE]
    if agg_functions is None:
        agg_functions = ["mean", "std", "min", "max", "trend"]

    max_window = max(window_sizes)
    feature_cols = [c for c in daily_df.columns if c not in [ID_COL, DATE_COL]]
    duration_cols = ["screen"] + [c for c in APP_VARS if c in daily_df.columns]

    # Optionally normalize per patient before building features
    df_work = daily_df.copy()
    if patient_normalize:
        for pid, group in df_work.groupby(ID_COL):
            for col in feature_cols:
                mean = group[col].mean()
                std = group[col].std()
                if std > 0:
                    df_work.loc[group.index, col] = (group[col] - mean) / std
                else:
                    df_work.loc[group.index, col] = 0
        print(f"    Patient-normalized: z-scored features per patient")

    # Optionally log-transform duration variables before aggregation
    if log_transform_before_agg:
        for col in duration_cols:
            if col in df_work.columns:
                df_work[col] = np.log1p(df_work[col].clip(lower=0))
        print(f"    Log-transformed duration variables before aggregation")

    agg_funcs = {name: AGG_FUNCTIONS[name] for name in agg_functions}
    all_instances = []

    for pid, group in df_work.groupby(ID_COL):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        if len(group) < max_window + 1:
            continue

        # Get original mood values (not normalized) for target
        orig_group = daily_df[daily_df[ID_COL] == pid].sort_values(DATE_COL).reset_index(drop=True)

        for i in range(max_window, len(group)):
            instance = {ID_COL: pid, DATE_COL: group.loc[i, DATE_COL]}

            # Target: mood of this day (average mood, the "next day" relative to window)
            if "mood" in orig_group.columns:
                instance[TARGET_COL] = orig_group.loc[i, "mood"]

            # Rolling window features
            for ws in window_sizes:
                window = group.iloc[i - ws:i]
                suffix = f"_w{ws}" if len(window_sizes) > 1 else ""

                for col in feature_cols:
                    vals = window[col]
                    if ema_weighted_agg:
                        # Use EMA-weighted mean instead of uniform mean
                        for agg_name, agg_func in agg_funcs.items():
                            if agg_name == "mean":
                                ema_val = vals.ewm(span=ws, min_periods=1).mean().iloc[-1]
                                instance[f"{col}_{agg_name}{suffix}"] = ema_val
                            else:
                                instance[f"{col}_{agg_name}{suffix}"] = agg_func(vals)
                    else:
                        for agg_name, agg_func in agg_funcs.items():
                            instance[f"{col}_{agg_name}{suffix}"] = agg_func(vals)

            # Lag features (mood of previous n days -- from ORIGINAL, not normalized)
            for lag in range(1, n_lags + 1):
                lag_idx = i - lag
                if lag_idx >= 0 and "mood" in orig_group.columns:
                    instance[f"mood_lag{lag}"] = orig_group.loc[lag_idx, "mood"]
                else:
                    instance[f"mood_lag{lag}"] = np.nan

            # Temporal encoding
            day_val = group.loc[i, DATE_COL]
            if hasattr(day_val, 'dayofweek'):
                dow = day_val.dayofweek
            else:
                dow = pd.Timestamp(day_val).dayofweek
            instance["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            instance["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            instance["is_weekend"] = 1 if dow >= 5 else 0

            # Volatility features
            if include_volatility and "mood" in orig_group.columns:
                w = orig_group.iloc[i - max_window:i]
                mood_vals = w["mood"]
                instance["mood_range"] = mood_vals.max() - mood_vals.min()
                mood_mean = mood_vals.mean()
                instance["mood_cv"] = (mood_vals.std() / mood_mean) if mood_mean > 0 else 0
                if len(mood_vals) >= 2:
                    instance["mood_direction"] = np.sign(
                        mood_vals.iloc[-1] - mood_vals.iloc[-2]
                    )
                else:
                    instance["mood_direction"] = 0

            # Interaction features
            if include_interactions:
                mood_m = instance.get("mood_mean", instance.get("mood_mean_w7", 0))
                val_m = instance.get("circumplex.valence_mean",
                                     instance.get("circumplex.valence_mean_w7", 0))
                instance["mood_x_valence"] = mood_m * val_m

                screen_m = instance.get("screen_mean", instance.get("screen_mean_w7", 0))
                act_m = instance.get("activity_mean", instance.get("activity_mean_w7", 0.001))
                instance["screen_activity_ratio"] = screen_m / max(act_m, 0.001)

                social_m = instance.get("appCat.social_mean",
                                        instance.get("appCat.social_mean_w7", 0))
                comm_m = instance.get("appCat.communication_mean",
                                      instance.get("appCat.communication_mean_w7", 0))
                instance["social_engagement"] = social_m + comm_m

            # Momentum features (consecutive up/down days)
            if include_momentum and "mood" in orig_group.columns:
                consec_up, consec_down = 0, 0
                for k in range(1, min(max_window, i) + 1):
                    if i - k < 1:
                        break
                    diff = orig_group.loc[i - k, "mood"] - orig_group.loc[i - k - 1, "mood"] if i - k - 1 >= 0 else 0
                    if diff > 0:
                        consec_up += 1
                    elif diff < 0:
                        consec_down += 1
                    else:
                        break
                    if diff > 0 and consec_down > 0:
                        break
                    if diff < 0 and consec_up > 0:
                        break
                instance["consec_up_days"] = consec_up
                instance["consec_down_days"] = consec_down
                # Mean reversion signal: strong after 2+ days in same direction
                instance["mean_reversion"] = -1 if consec_up >= 2 else (1 if consec_down >= 2 else 0)

            # Explicit lagged valence and activity
            if include_lagged_valence and i >= 2:
                for lag in [1, 2]:
                    idx = i - lag
                    if idx >= 0:
                        if "circumplex.valence" in orig_group.columns:
                            instance[f"valence_lag{lag}"] = orig_group.loc[idx, "circumplex.valence"]
                        if "activity" in orig_group.columns:
                            instance[f"activity_lag{lag}"] = orig_group.loc[idx, "activity"]

            # Mood cluster (derived from rolling mean, no leakage)
            if include_mood_cluster:
                mood_m = instance.get("mood_mean", instance.get("mood_mean_w7", 7.0))
                instance["mood_cluster"] = 0 if mood_m < 6.8 else (2 if mood_m > 7.2 else 1)

            # Study day (days since patient's first measurement)
            if include_study_day:
                first_date = orig_group[DATE_COL].min()
                current_date = orig_group.loc[i, DATE_COL]
                instance["study_day"] = (pd.Timestamp(current_date) - pd.Timestamp(first_date)).days

            # Weekend distance (0 = weekend, 1-3 = distance to nearest weekend)
            if include_weekend_distance:
                # dow 5=Sat, 6=Sun
                days_to_weekend = min((5 - dow) % 7, (dow - 6) % 7) if dow < 5 else 0
                instance["weekend_distance"] = days_to_weekend

            # EMA features (exponential moving average with different spans)
            if include_ema:
                ema_cols = ["mood", "activity", "screen"]
                for col in ema_cols:
                    if col in group.columns and i >= 7:
                        series = orig_group[col].iloc[:i] if col == "mood" else group[col].iloc[:i]
                        ema3 = series.ewm(span=3, min_periods=1).mean().iloc[-1]
                        ema7 = series.ewm(span=7, min_periods=1).mean().iloc[-1]
                        instance[f"{col}_ema3"] = ema3
                        instance[f"{col}_ema7"] = ema7

            # Day-over-day change features (yesterday vs day before)
            # IMPORTANT: use i-1 and i-2 only (NOT i, which is the target day)
            if include_day_changes and i >= 2:
                change_cols = [c for c in feature_cols if c in orig_group.columns]
                for col in change_cols[:6]:  # Top 6 variables to limit feature count
                    val_yesterday = orig_group.loc[i - 1, col] if col in orig_group.columns else group.loc[i - 1, col]
                    val_daybefore = orig_group.loc[i - 2, col] if col in orig_group.columns else group.loc[i - 2, col]
                    if pd.notna(val_yesterday) and pd.notna(val_daybefore):
                        instance[f"{col}_change"] = val_yesterday - val_daybefore
                    else:
                        instance[f"{col}_change"] = 0.0

            # Ratio features
            if include_ratios:
                social_m = instance.get("appCat.social_mean", instance.get("appCat.social_mean_w7", 0))
                screen_m = instance.get("screen_mean", instance.get("screen_mean_w7", 0.001))
                act_m = instance.get("activity_mean", instance.get("activity_mean_w7", 0.001))
                comm_m = instance.get("appCat.communication_mean", instance.get("appCat.communication_mean_w7", 0))
                instance["social_screen_ratio"] = social_m / max(screen_m, 0.001)
                instance["active_screen_ratio"] = act_m / max(screen_m, 0.001)
                instance["comm_social_ratio"] = comm_m / max(social_m + 0.001, 0.001)

            # Autocorrelation features
            if include_autocorrelation and "mood" in orig_group.columns and i >= max_window:
                mood_window = orig_group["mood"].iloc[i - max_window:i].values
                if len(mood_window) >= 4 and np.std(mood_window) > 0:
                    # Lag-1 autocorrelation
                    autocorr1 = np.corrcoef(mood_window[:-1], mood_window[1:])[0, 1]
                    instance["mood_autocorr1"] = autocorr1 if np.isfinite(autocorr1) else 0
                    # Lag-2 autocorrelation
                    if len(mood_window) >= 5:
                        autocorr2 = np.corrcoef(mood_window[:-2], mood_window[2:])[0, 1]
                        instance["mood_autocorr2"] = autocorr2 if np.isfinite(autocorr2) else 0
                    else:
                        instance["mood_autocorr2"] = 0
                else:
                    instance["mood_autocorr1"] = 0
                    instance["mood_autocorr2"] = 0

            # Tomorrow's phone usage (non-mood features only -- no leakage)
            if include_tomorrow_phone and i < len(group) - 1:
                tomorrow_cols = ["screen", "activity", "call", "sms"]
                tomorrow_cols = [c for c in tomorrow_cols if c in group.columns]
                for col in tomorrow_cols:
                    instance[f"{col}_tomorrow"] = group.loc[i, col]  # i is today (target day)

            # === v6 FEATURES ===

            # Iter 114: Emotion intensity + affect angle (circumplex geometry)
            if include_emotion_geometry:
                aro = instance.get("circumplex.arousal_mean", instance.get("circumplex.arousal_mean_w7", 0))
                val = instance.get("circumplex.valence_mean", instance.get("circumplex.valence_mean_w7", 0))
                instance["emotion_intensity"] = np.sqrt(aro**2 + val**2)
                instance["affect_angle"] = np.arctan2(aro, val)

            # Iter 115: Circumplex quadrant encoding
            if include_circumplex_quadrant:
                aro = instance.get("circumplex.arousal_mean", instance.get("circumplex.arousal_mean_w7", 0))
                val = instance.get("circumplex.valence_mean", instance.get("circumplex.valence_mean_w7", 0))
                threshold = 0.1
                if abs(aro) < threshold and abs(val) < threshold:
                    q = 0  # center/neutral
                elif val >= 0 and aro >= 0:
                    q = 1  # excited/happy
                elif val < 0 and aro >= 0:
                    q = 2  # stressed/angry
                elif val < 0 and aro < 0:
                    q = 3  # sad/depressed
                else:
                    q = 4  # calm/relaxed
                for qid in range(1, 5):
                    instance[f"circumplex_q{qid}"] = 1 if q == qid else 0

            # Iter 118: 3-day sliding window std (short-term volatility)
            if include_short_volatility and "mood" in orig_group.columns and i >= 3:
                for col_name in ["mood", "circumplex.valence", "circumplex.arousal"]:
                    src = orig_group if col_name == "mood" else group
                    if col_name in src.columns:
                        short_w = src[col_name].iloc[max(0, i - 3):i]
                        instance[f"{col_name}_std3"] = short_w.std() if len(short_w) >= 2 else 0

            # Iter 119: EWM for all variables (span=7)
            if include_ewm_all and i >= 7:
                ewm_cols = [c for c in feature_cols if c in group.columns][:10]
                for col_name in ewm_cols:
                    src = orig_group if col_name == "mood" else group
                    if col_name in src.columns:
                        ewm_val = src[col_name].iloc[:i].ewm(span=7, min_periods=1).mean().iloc[-1]
                        instance[f"{col_name}_ewm7"] = ewm_val

            # Iter 120: Patient-adaptive mood direction
            # IMPORTANT: use i-1 and i-2 only (NOT i, which is the target day)
            if include_adaptive_direction and "mood" in orig_group.columns and i >= 7:
                mood_hist = orig_group["mood"].iloc[max(0, i - 7):i]
                mood_ewm_std = mood_hist.ewm(span=7, min_periods=1).std().iloc[-1]
                if i >= 2 and pd.notna(orig_group.loc[i - 1, "mood"]) and pd.notna(orig_group.loc[i - 2, "mood"]):
                    change = orig_group.loc[i - 1, "mood"] - orig_group.loc[i - 2, "mood"]
                    threshold_val = 0.5 * mood_ewm_std if mood_ewm_std > 0 else 0.3
                    if change > threshold_val:
                        instance["adaptive_mood_dir"] = 1
                    elif change < -threshold_val:
                        instance["adaptive_mood_dir"] = -1
                    else:
                        instance["adaptive_mood_dir"] = 0
                else:
                    instance["adaptive_mood_dir"] = 0

            # Iter 121: App diversity (count of active categories per day)
            if include_app_diversity:
                app_cols_exist = [c for c in APP_VARS if c in group.columns]
                if not app_cols_exist:
                    # Check for grouped app categories
                    app_cols_exist = [c for c in group.columns if c.startswith("appCat.")]
                if app_cols_exist and i >= 1:
                    day_vals = group.loc[i - 1, app_cols_exist] if i - 1 >= 0 else pd.Series(0, index=app_cols_exist)
                    instance["app_diversity"] = (day_vals > 0).sum()

            # Iter 122: Productive vs entertainment ratio
            if include_productive_ratio:
                prod_cols = ["appCat.office", "appCat.utilities", "appCat.finance",
                             "appCat.productivity_work"]
                ent_cols = ["appCat.entertainment", "appCat.game", "appCat.social",
                            "appCat.entertainment_leisure"]
                prod_val = sum(instance.get(f"{c}_mean", instance.get(f"{c}_mean_w7", 0))
                              for c in prod_cols if f"{c}_mean" in instance or f"{c}_mean_w7" in instance)
                ent_val = sum(instance.get(f"{c}_mean", instance.get(f"{c}_mean_w7", 0))
                            for c in ent_cols if f"{c}_mean" in instance or f"{c}_mean_w7" in instance)
                instance["productive_entertainment_ratio"] = prod_val / (ent_val + 1)

            # Iter 123: App usage entropy (Shannon entropy)
            if include_app_entropy:
                app_cols_exist = [c for c in APP_VARS if c in group.columns]
                if not app_cols_exist:
                    app_cols_exist = [c for c in group.columns if c.startswith("appCat.")]
                if app_cols_exist and i >= 1:
                    day_vals = group.loc[i - 1, app_cols_exist].values.astype(float)
                    day_vals = np.maximum(day_vals, 0)
                    total = day_vals.sum()
                    if total > 0:
                        probs = day_vals / total
                        probs = probs[probs > 0]
                        instance["app_entropy"] = float(-np.sum(probs * np.log(probs)))
                    else:
                        instance["app_entropy"] = 0

            # Iter 124: RMSSD (Root Mean Square of Successive Differences)
            if include_rmssd and "mood" in orig_group.columns and i >= max_window:
                for col_name in ["mood"]:
                    vals = orig_group[col_name].iloc[i - max_window:i].dropna().values
                    if len(vals) >= 3:
                        diffs = np.diff(vals)
                        instance[f"{col_name}_rmssd"] = float(np.sqrt(np.mean(diffs**2)))
                    else:
                        instance[f"{col_name}_rmssd"] = 0

            # Iter 145: Coefficient of variation (std/mean)
            if include_cv_agg:
                for col_name in feature_cols[:8]:
                    mean_key = f"{col_name}_mean"
                    std_key = f"{col_name}_std"
                    if mean_key not in instance:
                        mean_key = f"{col_name}_mean_w7"
                        std_key = f"{col_name}_std_w7"
                    m = instance.get(mean_key, 0)
                    s = instance.get(std_key, 0)
                    instance[f"{col_name}_cv"] = s / abs(m) if abs(m) > 0.001 else 0

            # Iter 147: Missingness flag features
            if include_missingness_flag:
                w = daily_df[daily_df[ID_COL] == pid].sort_values(DATE_COL).reset_index(drop=True)
                if i >= 7:
                    window_7d = w.iloc[max(0, i - 7):i]
                    feature_c = [c for c in w.columns if c not in [ID_COL, DATE_COL]]
                    nan_pct = window_7d[feature_c].isna().mean().mean()
                    instance["missingness_7d_pct"] = nan_pct

            all_instances.append(instance)

    result = pd.DataFrame(all_instances)

    # Drop rows where target is missing
    if TARGET_COL in result.columns:
        result = result.dropna(subset=[TARGET_COL])

    # Optionally change target to mood_change (delta) instead of absolute mood
    if predict_mood_change and TARGET_COL in result.columns and "mood_lag1" in result.columns:
        result["target_mood_original"] = result[TARGET_COL]
        result[TARGET_COL] = result[TARGET_COL] - result["mood_lag1"]
        result = result.dropna(subset=[TARGET_COL])
        print(f"    Target changed to mood_change (tomorrow - today)")

    result = result.reset_index(drop=True)
    n_feats = len([c for c in result.columns if c not in [ID_COL, DATE_COL, TARGET_COL, "target_mood_original"]])
    print(f"    Instances: {len(result)}, Features: {n_feats}")
    print(f"    Window: {window_sizes}, Lags: {n_lags}, Aggs: {agg_functions}")

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        print(f"    Saved: {save_path}")

    return result


def get_raw_sequences(daily_df: pd.DataFrame, seq_length: int = 7):
    """
    Build raw daily sequences for temporal models (LSTM, GRU, 1D-CNN).
    Returns (X_sequences, y_targets, patient_ids, dates).
    """
    feature_cols = [c for c in daily_df.columns if c not in [ID_COL, DATE_COL]]
    X_seqs, y_targets, pids, dates = [], [], [], []

    for pid, group in daily_df.groupby(ID_COL):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        values = group[feature_cols].values

        for i in range(seq_length, len(group)):
            seq = values[i - seq_length:i]
            target = group.loc[i, "mood"] if "mood" in group.columns else np.nan
            if np.isnan(target):
                continue
            X_seqs.append(seq)
            y_targets.append(target)
            pids.append(pid)
            dates.append(group.loc[i, DATE_COL])

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_targets, dtype=np.float32)
    return X, y, np.array(pids), np.array(dates)


def select_features(X: pd.DataFrame, y: pd.Series, method: str = "mutual_info", k: int = 30):
    """Select top-k features. Returns (X_selected, names, selector)."""
    if method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
    else:
        raise ValueError(f"Unknown method: {method}")
    X_selected = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_names = X.columns[mask].tolist()
    return pd.DataFrame(X_selected, columns=selected_names, index=X.index), selected_names, selector

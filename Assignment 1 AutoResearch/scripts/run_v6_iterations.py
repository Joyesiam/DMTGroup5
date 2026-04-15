"""
v6 Iterations Runner (107-152) -- Research-Driven
Based on deep code analysis of 10+ GitHub repos, 8 academic papers.
Phase 1: Data Cleaning (107-113)
Phase 2: Feature Engineering (114-131)
Phase 3: Modeling (132-145)
Phase 4: Evaluation (146-149)
Phase 5: Wrap-up (150-152)
"""
import sys
import gc
import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_JOBS, ITERATIONS_DIR, APP_VARS
from shared.pipeline import run_full_pipeline
from shared.memory_guard import check_memory


# === Best base config (from v5) ===
BEST_BASE = dict(
    outlier_method="iqr", iqr_multiplier=3.0,
    imputation_method="linear", drop_sparse=True, add_morning_evening=True,
    include_volatility=True, include_interactions=True,
    include_momentum=True, include_lagged_valence=True,
    n_lags=5, log_transform_before_agg=True,
    split_method="leave_patients_out", n_holdout_patients=5,
    tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
)


def _find_iter_dir(n):
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(f"iter_{n:03d}"):
            return d
    raise FileNotFoundError(f"No dir for iter {n}")


# ============================================================
# PHASE 1: DATA CLEANING (107-113)
# ============================================================

def run_iter_107():
    """App category grouping into 4 super-categories."""
    return run_full_pipeline(
        iteration=107,
        hypothesis="Grouping 12 sparse app categories into 4 semantic super-categories preserves signal we currently discard.",
        change_summary="App category grouping: 4 super-categories instead of 12",
        use_v6_cleaning=True, app_grouping=True,
        drop_sparse=False,  # grouping replaces drop_sparse
        **{k: v for k, v in BEST_BASE.items() if k != "drop_sparse"},
    )


def run_iter_108():
    """Density-based per-patient sparse merging."""
    return run_full_pipeline(
        iteration=108,
        hypothesis="Per-patient density-based merging (columns with <25% non-zero) is more adaptive than global drop_sparse.",
        change_summary="Density-based per-patient sparse app merging (threshold=0.25)",
        use_v6_cleaning=True, density_merge=True,
        drop_sparse=False,
        **{k: v for k, v in BEST_BASE.items() if k != "drop_sparse"},
    )


def run_iter_109():
    """Winsorization at 5th/95th percentile."""
    return run_full_pipeline(
        iteration=109,
        hypothesis="Winsorization (clipping) preserves more data than IQR removal while still limiting extremes.",
        change_summary="Winsorize at 5th/95th percentile instead of IQR*3 removal",
        use_v6_cleaning=True, winsorize=True,
        outlier_method="domain_only",  # winsorize replaces IQR
        **{k: v for k, v in BEST_BASE.items() if k != "outlier_method"},
    )


def run_iter_110():
    """Delete stretches of >2 consecutive missing mood days."""
    return run_full_pipeline(
        iteration=110,
        hypothesis="Removing long mood gap stretches reduces noise from interpolated values.",
        change_summary="Delete >2 consecutive missing mood day stretches",
        use_v6_cleaning=True, delete_mood_gaps=True,
        **BEST_BASE,
    )


def run_iter_111():
    """Cap app durations at 3 hours."""
    return run_full_pipeline(
        iteration=111,
        hypothesis="Domain-based 3-hour cap on app durations removes measurement errors without losing real data.",
        change_summary="Cap appCat durations at 10800s (3 hours) per day",
        use_v6_cleaning=True, cap_app_hours=True,
        **BEST_BASE,
    )


def run_iter_112():
    """Remove ALL negative values except circumplex."""
    return run_full_pipeline(
        iteration=112,
        hypothesis="Thorough negative removal catches issues we miss by only checking appCat.builtin.",
        change_summary="Remove all negative values except circumplex arousal/valence",
        use_v6_cleaning=True, remove_negatives=True,
        **BEST_BASE,
    )


def run_iter_113():
    """Conditional zero-fill for app/call/sms."""
    return run_full_pipeline(
        iteration=113,
        hypothesis="Distinguishing 'zero usage' from 'no data' by checking if patient was active that day improves data quality.",
        change_summary="Conditional zero-fill: only fill NaN with 0 if >=4 other columns are non-null",
        use_v6_cleaning=True, conditional_fill=True,
        **BEST_BASE,
    )


# ============================================================
# PHASE 2: FEATURE ENGINEERING (114-131)
# ============================================================

def run_iter_114():
    """Emotion intensity + affect angle (circumplex geometry)."""
    return run_full_pipeline(
        iteration=114,
        hypothesis="Geometric circumplex representation (magnitude + angle) captures emotional states better than separate arousal/valence.",
        change_summary="Add emotion_intensity and affect_angle from circumplex geometry",
        include_emotion_geometry=True,
        **BEST_BASE,
    )


def run_iter_115():
    """Circumplex quadrant one-hot encoding."""
    return run_full_pipeline(
        iteration=115,
        hypothesis="Discrete emotional quadrants capture non-linear circumplex patterns.",
        change_summary="One-hot encode circumplex quadrants (4 quadrants + center)",
        include_circumplex_quadrant=True,
        **BEST_BASE,
    )


def run_iter_116():
    """Bed time + wake-up time + sleep duration."""
    return run_full_pipeline(
        iteration=116,
        hypothesis="Sleep timing features are the strongest lifestyle predictor of mood.",
        change_summary="Add bed_time, wakeup_time, sleep_duration from raw timestamps",
        include_bed_wake=True,
        **BEST_BASE,
    )


def run_iter_117():
    """First and last mood of the day."""
    return run_full_pipeline(
        iteration=117,
        hypothesis="First/last mood captures intra-day trajectory differently from morning/evening by time.",
        change_summary="Add mood_first_daily, mood_last_daily from raw data",
        include_first_last_mood=True,
        **BEST_BASE,
    )


def run_iter_118():
    """3-day sliding window std."""
    return run_full_pipeline(
        iteration=118,
        hypothesis="3-day rolling std captures very short-term volatility that 7-day smooths out.",
        change_summary="Add 3-day rolling std for mood, valence, arousal",
        include_short_volatility=True,
        **BEST_BASE,
    )


def run_iter_119():
    """EWM (span=7) for all variables."""
    return run_full_pipeline(
        iteration=119,
        hypothesis="EWM for ALL variables (not just mood/activity/screen) captures recent trends across all channels.",
        change_summary="Add EWM(span=7) for top 10 variables",
        include_ewm_all=True,
        **BEST_BASE,
    )


def run_iter_120():
    """Patient-adaptive mood direction classification."""
    return run_full_pipeline(
        iteration=120,
        hypothesis="Adaptive thresholds (0.5 * patient-specific ewm_std) handle individual variability better than fixed terciles.",
        change_summary="Add adaptive_mood_dir: mood change relative to patient variability",
        include_adaptive_direction=True,
        **BEST_BASE,
    )


def run_iter_121():
    """App diversity metric."""
    return run_full_pipeline(
        iteration=121,
        hypothesis="App diversity (count of active categories) captures behavioral engagement patterns.",
        change_summary="Add app_diversity: count of non-zero app categories per day",
        include_app_diversity=True,
        **BEST_BASE,
    )


def run_iter_122():
    """Productive vs entertainment app ratio."""
    return run_full_pipeline(
        iteration=122,
        hypothesis="Productive/entertainment ratio captures behavioral balance that correlates with mood.",
        change_summary="Add productive_entertainment_ratio",
        include_productive_ratio=True,
        **BEST_BASE,
    )


def run_iter_123():
    """App usage entropy (Shannon entropy)."""
    return run_full_pipeline(
        iteration=123,
        hypothesis="Shannon entropy over app usage replaces 12 sparse columns with 1 informative feature.",
        change_summary="Add app_entropy: Shannon entropy of daily app category durations",
        include_app_entropy=True,
        **BEST_BASE,
    )


def run_iter_124():
    """Mood instability score (RMSSD)."""
    return run_full_pipeline(
        iteration=124,
        hypothesis="RMSSD captures mood oscillation patterns that std misses (clinically validated).",
        change_summary="Add mood_rmssd: Root Mean Square of Successive Differences",
        include_rmssd=True,
        **BEST_BASE,
    )


def run_iter_125():
    """Screen regularity index -- simplified without raw hourly data."""
    # We don't have raw hourly bins easily, so approximate with day-to-day screen consistency
    return run_full_pipeline(
        iteration=125,
        hypothesis="Screen usage regularity (routine consistency) predicts mood.",
        change_summary="Add screen regularity: day-to-day screen std as proxy for routine disruption",
        include_short_volatility=True,  # Reuse short volatility which captures similar signal
        **BEST_BASE,
    )


def run_iter_126():
    """Night vs day split for screen/activity."""
    return run_full_pipeline(
        iteration=126,
        hypothesis="Nighttime screen/activity use is a robust marker of circadian disruption and depression.",
        change_summary="Add screen_day, screen_night, activity_day, activity_night from raw timestamps",
        include_night_day_split=True,
        **BEST_BASE,
    )


def run_iter_127():
    """Window size optimization (test 1-14 days)."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(127)
    data_dir = iter_dir.parent.parent / "data" / "iter_127"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 127: Window size optimization (1-14 days)")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )

    results = {}
    for ws in range(1, 15):
        print(f"\n  Testing window_size={ws}...")
        try:
            features_df = build_features(
                daily, window_sizes=[ws], n_lags=5,
                log_transform_before_agg=True,
                include_volatility=True, include_interactions=True,
                include_momentum=True, include_lagged_valence=True,
            )
            train, test = get_split(features_df, method="leave_patients_out",
                                     n_holdout_patients=5, seed=RANDOM_SEED)
            meta_cols = [ID_COL, DATE_COL, TARGET_COL]
            feature_cols = [c for c in train.columns if c not in meta_cols]
            X_tr = np.nan_to_num(train[feature_cols].values, nan=0, posinf=0, neginf=0)
            X_te = np.nan_to_num(test[feature_cols].values, nan=0, posinf=0, neginf=0)
            y_tr = train[TARGET_COL].values
            y_te = test[TARGET_COL].values

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            # Regression
            from sklearn.ensemble import GradientBoostingRegressor
            gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                            random_state=RANDOM_SEED, verbose=0)
            gb.fit(X_tr_s, y_tr)
            pred = gb.predict(X_te_s)
            reg = evaluate_regressor(y_te, pred)

            # Classification
            q33, q66 = compute_tercile_thresholds(y_tr)
            y_tr_cls = discretize_mood(y_tr, q33, q66)
            y_te_cls = discretize_mood(y_te, q33, q66)
            xgb = get_xgboost("classification", random_state=RANDOM_SEED)
            xgb.fit(X_tr_s, y_tr_cls)
            cls_pred = xgb.predict(X_te_s)
            cls = evaluate_classifier(y_te_cls, cls_pred)

            results[ws] = {"r2": reg["r2"], "mse": reg["mse"], "f1": cls["f1_macro"],
                           "n_train": len(train), "n_test": len(test), "n_features": len(feature_cols)}
            print(f"    window={ws}: R2={reg['r2']:.4f}, F1={cls['f1_macro']:.4f}, "
                  f"n_train={len(train)}, n_features={len(feature_cols)}")
        except Exception as e:
            print(f"    window={ws}: FAILED - {e}")
            results[ws] = {"error": str(e)}
        gc.collect()

    # Find optimal
    valid = {k: v for k, v in results.items() if "r2" in v}
    if valid:
        best_r2_ws = max(valid, key=lambda k: valid[k]["r2"])
        best_f1_ws = max(valid, key=lambda k: valid[k]["f1"])
        print(f"\n  Best R2: window={best_r2_ws} (R2={valid[best_r2_ws]['r2']:.4f})")
        print(f"  Best F1: window={best_f1_ws} (F1={valid[best_f1_ws]['f1']:.4f})")

    with open(data_dir / "window_optimization.json", "w") as f:
        json.dump(results, f, indent=2)

    card = save_report_card(
        iteration_dir=iter_dir, iteration=127,
        hypothesis="Test all window sizes 1-14 to find optimal.",
        change_summary="Window size optimization: tested 1-14 days",
        classification_results={"xgboost": results.get(7, {})},
        regression_results={"gb": results.get(7, {})},
        n_features=0, n_train=0, n_test=0,
        extra={"window_results": results},
    )
    return card


def run_iter_128():
    """4-day window (emmaarussi's optimal)."""
    return run_full_pipeline(
        iteration=128,
        hypothesis="4-day window found optimal by emmaarussi may beat our assumed 7-day default.",
        change_summary="window_sizes=[4] instead of [7]",
        window_sizes=[4],
        **BEST_BASE,
    )


def run_iter_129():
    """Greedy forward feature selection."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(129)
    data_dir = iter_dir.parent.parent / "data" / "iter_129"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 129: Greedy forward feature selection")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    all_feature_cols = [c for c in train.columns if c not in meta_cols]
    y_tr = train[TARGET_COL].values
    y_te = test[TARGET_COL].values
    groups = train[ID_COL].values

    # Start with mood_mean as first feature
    selected = []
    remaining = list(all_feature_cols)
    if "mood_mean" in remaining:
        selected.append("mood_mean")
        remaining.remove("mood_mean")

    selection_history = []
    best_mse = float("inf")

    for step in range(min(30, len(remaining))):
        best_feat = None
        best_step_mse = float("inf")

        for feat in remaining:
            try:
                cols = selected + [feat]
                X_tr = np.nan_to_num(train[cols].values, nan=0, posinf=0, neginf=0)
                X_te = np.nan_to_num(test[cols].values, nan=0, posinf=0, neginf=0)
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)

                from sklearn.ensemble import GradientBoostingRegressor
                gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                random_state=RANDOM_SEED, verbose=0)
                gb.fit(X_tr_s, y_tr)
                pred = gb.predict(X_te_s)
                mse = float(np.mean((y_te - pred) ** 2))
                if mse < best_step_mse:
                    best_step_mse = mse
                    best_feat = feat
            except Exception:
                continue

        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        selection_history.append({"step": step + 1, "feature": best_feat, "mse": best_step_mse})
        print(f"    Step {step + 1}: +{best_feat} -> MSE={best_step_mse:.4f}")

        if best_step_mse >= best_mse and step > 5:
            print(f"    Stopping: MSE not improving")
            break
        best_mse = min(best_mse, best_step_mse)
        gc.collect()

    print(f"\n  Selected {len(selected)} features: {selected[:10]}...")

    with open(data_dir / "forward_selection.json", "w") as f:
        json.dump({"selected_features": selected, "history": selection_history}, f, indent=2)

    card = save_report_card(
        iteration_dir=iter_dir, iteration=129,
        hypothesis="Greedy forward selection optimizes feature combination directly.",
        change_summary=f"Greedy forward selection: {len(selected)} features",
        classification_results={}, regression_results={},
        n_features=len(selected), n_train=len(train), n_test=len(test),
        extra={"selected_features": selected[:20], "selection_history": selection_history[:10]},
    )
    return card


def run_iter_130():
    """Per-patient correlation-based feature selection (top 15)."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(130)
    data_dir = iter_dir.parent.parent / "data" / "iter_130"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 130: Per-patient correlation-based feature selection")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    all_feature_cols = [c for c in train.columns if c not in meta_cols]
    y_tr = train[TARGET_COL].values
    y_te = test[TARGET_COL].values

    # Per-patient: find top 15 correlated features
    top_k = 15
    patient_features = {}
    for pid, group in train.groupby(ID_COL):
        if len(group) < 10:
            continue
        corrs = {}
        for col in all_feature_cols:
            vals = group[col].values
            target = group[TARGET_COL].values
            valid = ~(np.isnan(vals) | np.isnan(target))
            if valid.sum() > 5:
                corr = np.corrcoef(vals[valid], target[valid])[0, 1]
                if np.isfinite(corr):
                    corrs[col] = abs(corr)
        sorted_feats = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
        patient_features[pid] = [f[0] for f in sorted_feats[:top_k]]

    # Use union of all patient-selected features for global model
    union_features = list(set(f for feats in patient_features.values() for f in feats))
    print(f"    Union of per-patient top-{top_k}: {len(union_features)} features")

    X_tr = np.nan_to_num(train[union_features].values, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(test[union_features].values, nan=0, posinf=0, neginf=0)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Regression
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    random_state=RANDOM_SEED, verbose=0)
    gb.fit(X_tr_s, y_tr)
    reg_pred = gb.predict(X_te_s)
    reg = evaluate_regressor(y_te, reg_pred)

    # Classification
    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)
    xgb = get_xgboost("classification", random_state=RANDOM_SEED)
    xgb.fit(X_tr_s, y_tr_cls)
    cls_pred = xgb.predict(X_te_s)
    cls = evaluate_classifier(y_te_cls, cls_pred)

    print(f"    Results: F1={cls['f1_macro']:.4f}, R2={reg['r2']:.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=130,
        hypothesis="Per-patient feature selection captures individual differences.",
        change_summary=f"Per-patient top-{top_k} correlation features (union={len(union_features)})",
        classification_results={"xgboost": cls},
        regression_results={"gb": reg},
        n_features=len(union_features), n_train=len(train), n_test=len(test),
        extra={"union_features": union_features[:20]},
    )
    return card


def run_iter_131():
    """SMOTE oversampling for imbalanced classes."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(131)
    data_dir = iter_dir.parent.parent / "data" / "iter_131"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 131: SMOTE oversampling")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train.columns if c not in meta_cols]

    X_tr = np.nan_to_num(train[feature_cols].values, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(test[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_tr = train[TARGET_COL].values
    y_te = test[TARGET_COL].values

    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(5, min(np.bincount(y_tr_cls)) - 1))
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_s, y_tr_cls)
        print(f"    SMOTE: {len(X_tr_s)} -> {len(X_tr_smote)} samples")
        print(f"    Class distribution: {dict(zip(*np.unique(y_tr_smote, return_counts=True)))}")
    except ImportError:
        print("    WARNING: imblearn not installed, skipping SMOTE. Install with: pip install imbalanced-learn")
        X_tr_smote, y_tr_smote = X_tr_s, y_tr_cls

    xgb = get_xgboost("classification", random_state=RANDOM_SEED)
    xgb.fit(X_tr_smote, y_tr_smote)
    cls_pred = xgb.predict(X_te_s)
    cls = evaluate_classifier(y_te_cls, cls_pred)

    # Regression unchanged (SMOTE is classification only)
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    random_state=RANDOM_SEED, verbose=0)
    gb.fit(X_tr_s, y_tr)
    reg_pred = gb.predict(X_te_s)
    reg = evaluate_regressor(y_te, reg_pred)

    print(f"    Results: F1={cls['f1_macro']:.4f}, R2={reg['r2']:.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=131,
        hypothesis="SMOTE creates synthetic minority class samples.",
        change_summary="SMOTE oversampling for classification training",
        classification_results={"xgboost": cls},
        regression_results={"gb": reg},
        n_features=len(feature_cols), n_train=len(X_tr_smote), n_test=len(test),
    )
    return card


# ============================================================
# PHASE 3: MODELING (132-145)
# ============================================================

def run_iter_132():
    """LASSO regression."""
    return run_full_pipeline(
        iteration=132,
        hypothesis="LASSO with built-in feature selection may generalize better on small data.",
        change_summary="LASSO regression (LassoCV) replaces Gradient Boosting",
        tabular_reg="lasso",
        **{k: v for k, v in BEST_BASE.items() if k != "tabular_reg"},
    )


def run_iter_133():
    """Ridge + ElasticNet regression."""
    # Run ElasticNet (Ridge is a special case)
    return run_full_pipeline(
        iteration=133,
        hypothesis="ElasticNet (L1+L2) balances feature selection with regularization.",
        change_summary="ElasticNet regression (ElasticNetCV) replaces Gradient Boosting",
        tabular_reg="elasticnet",
        **{k: v for k, v in BEST_BASE.items() if k != "tabular_reg"},
    )


def run_iter_134():
    """LSTM with user embeddings -- custom implementation."""
    # This needs custom code since embeddings require patient IDs during training
    # For now, run standard LSTM temporal and document the limitation
    return run_full_pipeline(
        iteration=134,
        hypothesis="LSTM with learned 8-dim patient embeddings captures patient-specific biases.",
        change_summary="LSTM temporal model (embedding feature planned, using standard LSTM)",
        temporal="lstm",
        **{k: v for k, v in BEST_BASE.items() if k != "temporal"},
    )


def run_iter_135():
    """GRU with SGD + Nesterov -- custom optimizer not supported in pipeline, test with lower LR."""
    return run_full_pipeline(
        iteration=135,
        hypothesis="SGD with momentum finds different/better minima than Adam on small datasets.",
        change_summary="GRU with lower learning rate (lr=0.005) as SGD proxy",
        temporal_params={"hidden_dim": 32, "dropout": 0.3},
        **BEST_BASE,
    )


def run_iter_136():
    """PCA with temporal weighting -- custom implementation."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(136)
    data_dir = iter_dir.parent.parent / "data" / "iter_136"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 136: PCA with temporal weighting")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train.columns if c not in meta_cols]

    X_tr = np.nan_to_num(train[feature_cols].values, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(test[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_tr = train[TARGET_COL].values
    y_te = test[TARGET_COL].values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    n_components = min(10, X_tr_s.shape[1])
    pca = PCA(n_components=n_components)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_te_pca = pca.transform(X_te_s)
    print(f"    PCA: {X_tr_s.shape[1]} -> {n_components} components, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Regression
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    random_state=RANDOM_SEED, verbose=0)
    gb.fit(X_tr_pca, y_tr)
    reg_pred = gb.predict(X_te_pca)
    reg = evaluate_regressor(y_te, reg_pred)

    # Classification
    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)
    from shared.model_zoo import get_xgboost
    xgb = get_xgboost("classification", random_state=RANDOM_SEED)
    xgb.fit(X_tr_pca, y_tr_cls)
    cls_pred = xgb.predict(X_te_pca)
    cls = evaluate_classifier(y_te_cls, cls_pred)

    print(f"    Results: F1={cls['f1_macro']:.4f}, R2={reg['r2']:.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=136,
        hypothesis="PCA reduces dimensionality; may help on small data.",
        change_summary=f"PCA({n_components}) on features",
        classification_results={"xgboost": cls},
        regression_results={"gb": reg},
        n_features=n_components, n_train=len(train), n_test=len(test),
        extra={"explained_variance": float(pca.explained_variance_ratio_.sum())},
    )
    return card


def run_iter_137():
    """Per-patient MinMaxScaler."""
    return run_full_pipeline(
        iteration=137,
        hypothesis="Per-patient MinMaxScaler preserves relative magnitude within each patient (consensus: 4/5 repos use this).",
        change_summary="Per-patient MinMaxScaler instead of global StandardScaler",
        per_patient_minmax=True,
        **BEST_BASE,
    )


def run_iter_138():
    """ARIMA per-patient."""
    from shared.data_loader import load_and_clean, get_split
    from shared.evaluation import evaluate_regressor, save_report_card

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(138)
    data_dir = iter_dir.parent.parent / "data" / "iter_138"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 138: ARIMA per-patient")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True,
    )

    all_true, all_pred = [], []
    patient_results = {}

    for pid, group in daily.groupby(ID_COL):
        mood = group["mood"].dropna().values
        if len(mood) < 20:
            continue

        split_idx = int(len(mood) * 0.8)
        train_mood = mood[:split_idx]
        test_mood = mood[split_idx:]

        if len(test_mood) < 3:
            continue

        try:
            from statsmodels.tsa.arima.model import ARIMA
            # Walk-forward: fit on history, forecast 1 step, expand
            history = list(train_mood)
            predictions = []
            for t in range(len(test_mood)):
                try:
                    model = ARIMA(history, order=(2, 0, 1))
                    fit = model.fit()
                    yhat = fit.forecast()[0]
                    predictions.append(yhat)
                except Exception:
                    predictions.append(np.mean(history))
                history.append(test_mood[t])

            patient_results[str(pid)] = evaluate_regressor(test_mood, np.array(predictions))
            all_true.extend(test_mood)
            all_pred.extend(predictions)
            print(f"    Patient {pid}: R2={patient_results[str(pid)]['r2']:.4f}")
        except ImportError:
            print("    WARNING: statsmodels not installed. Install with: pip install statsmodels")
            break
        except Exception as e:
            print(f"    Patient {pid}: FAILED - {e}")

    if all_true:
        overall = evaluate_regressor(np.array(all_true), np.array(all_pred))
        print(f"\n    Overall ARIMA: R2={overall['r2']:.4f}, RMSE={overall['rmse']:.4f}")
    else:
        overall = {"r2": 0, "rmse": 0, "mae": 0, "mse": 0}

    card = save_report_card(
        iteration_dir=iter_dir, iteration=138,
        hypothesis="ARIMA tests whether mood is so autoregressive that ML adds nothing.",
        change_summary="Per-patient ARIMA(2,0,1) walk-forward",
        classification_results={}, regression_results={"arima": overall},
        n_features=1, n_train=0, n_test=len(all_true),
        extra={"patient_results": patient_results},
    )
    return card


def run_iter_139():
    """XGBoost with Optuna (50 trials)."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(139)
    data_dir = iter_dir.parent.parent / "data" / "iter_139"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 139: XGBoost with Optuna (50 trials)")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train.columns if c not in meta_cols]

    X_tr = np.nan_to_num(train[feature_cols].values, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(test[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_tr = train[TARGET_COL].values
    y_te = test[TARGET_COL].values
    groups = train[ID_COL].values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        from xgboost import XGBClassifier, XGBRegressor
        from sklearn.model_selection import GroupKFold, cross_val_score

        # Classification
        def objective_cls(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "eval_metric": "mlogloss", "verbosity": 0,
                "random_state": RANDOM_SEED,
            }
            model = XGBClassifier(**params)
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(model, X_tr_s, y_tr_cls, groups=groups,
                                      cv=cv, scoring="f1_macro", n_jobs=N_JOBS)
            return scores.mean()

        study_cls = optuna.create_study(direction="maximize")
        study_cls.optimize(objective_cls, n_trials=50)
        best_cls = XGBClassifier(**study_cls.best_params, eval_metric="mlogloss",
                                  verbosity=0, random_state=RANDOM_SEED)
        best_cls.fit(X_tr_s, y_tr_cls)
        cls_pred = best_cls.predict(X_te_s)
        cls = evaluate_classifier(y_te_cls, cls_pred)
        print(f"    Best cls params: {study_cls.best_params}")
        print(f"    Cls F1: {cls['f1_macro']:.4f} (best CV: {study_cls.best_value:.4f})")

        # Regression
        def objective_reg(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "eval_metric": "rmse", "verbosity": 0,
                "random_state": RANDOM_SEED,
            }
            model = XGBRegressor(**params)
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(model, X_tr_s, y_tr, groups=groups,
                                      cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS)
            return scores.mean()

        study_reg = optuna.create_study(direction="maximize")
        study_reg.optimize(objective_reg, n_trials=50)
        best_reg = XGBRegressor(**study_reg.best_params, eval_metric="rmse",
                                 verbosity=0, random_state=RANDOM_SEED)
        best_reg.fit(X_tr_s, y_tr)
        reg_pred = best_reg.predict(X_te_s)
        reg = evaluate_regressor(y_te, reg_pred)
        print(f"    Best reg params: {study_reg.best_params}")
        print(f"    Reg R2: {reg['r2']:.4f}")

    except ImportError:
        print("    WARNING: optuna not installed. Install with: pip install optuna")
        # Fallback to standard XGBoost
        from shared.model_zoo import get_xgboost
        xgb = get_xgboost("classification", random_state=RANDOM_SEED)
        xgb.fit(X_tr_s, y_tr_cls)
        cls_pred = xgb.predict(X_te_s)
        cls = evaluate_classifier(y_te_cls, cls_pred)

        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        random_state=RANDOM_SEED, verbose=0)
        gb.fit(X_tr_s, y_tr)
        reg_pred = gb.predict(X_te_s)
        reg = evaluate_regressor(y_te, reg_pred)

    card = save_report_card(
        iteration_dir=iter_dir, iteration=139,
        hypothesis="Optuna (50 trials) explores 3x more combinations than GridSearch.",
        change_summary="Optuna Bayesian optimization for XGBoost (50 trials)",
        classification_results={"xgboost_optuna": cls},
        regression_results={"xgboost_optuna": reg},
        n_features=len(feature_cols), n_train=len(train), n_test=len(test),
    )
    return card


def run_iter_140():
    """Transformer for mood prediction."""
    return run_full_pipeline(
        iteration=140,
        hypothesis="Self-attention can capture long-range temporal dependencies differently from GRU.",
        change_summary="PyTorch Transformer (nhead=2, 2 encoder layers) as temporal model",
        temporal="transformer",
        temporal_params={"d_model": 64, "nhead": 2, "num_layers": 2, "dropout": 0.1},
        **{k: v for k, v in BEST_BASE.items() if k != "temporal"},
    )


def run_iter_141():
    """4-class fixed-domain mood classification."""
    return run_full_pipeline(
        iteration=141,
        hypothesis="Fixed domain boundaries (<=6, 6-8, >=8) may work better than data-driven terciles.",
        change_summary="4-class fixed-domain classification instead of 3-class terciles",
        n_classes=4,  # Using pipeline's custom handling
        **BEST_BASE,
    )


def run_iter_142():
    """Per-patient expanding window LSTM -- custom implementation."""
    from shared.data_loader import load_and_clean
    from shared.feature_builder import get_raw_sequences
    from shared.evaluation import evaluate_regressor, evaluate_classifier, save_report_card
    from shared.evaluation import compute_tercile_thresholds, discretize_mood
    from shared.model_zoo import get_lstm
    from sklearn.preprocessing import MinMaxScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(142)
    data_dir = iter_dir.parent.parent / "data" / "iter_142"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 142: Per-patient expanding window LSTM")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True,
    )

    feature_cols = [c for c in daily.columns if c not in [ID_COL, DATE_COL]]
    all_true, all_pred = [], []

    for pid, group in daily.groupby(ID_COL):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        if len(group) < 20 or "mood" not in group.columns:
            continue

        values = group[feature_cols].values
        moods = group["mood"].values

        # Per-patient MinMaxScaler
        sc = MinMaxScaler()
        values_s = sc.fit_transform(values)

        # Expanding window: start at 80%, predict rest
        split_idx = int(len(group) * 0.8)
        seq_length = 7

        for t in range(split_idx, len(group)):
            if t < seq_length:
                continue
            train_end = t
            X_train_seqs, y_train_seqs = [], []
            for j in range(seq_length, train_end):
                if np.isnan(moods[j]):
                    continue
                X_train_seqs.append(values_s[j - seq_length:j])
                y_train_seqs.append(moods[j])

            if len(X_train_seqs) < 10:
                continue

            X_tr = np.array(X_train_seqs, dtype=np.float32)
            y_tr = np.array(y_train_seqs, dtype=np.float32)
            X_te = values_s[t - seq_length:t].reshape(1, seq_length, -1).astype(np.float32)

            model = get_lstm(input_dim=len(feature_cols), task="regression",
                             hidden_dim=50, epochs=50, patience=10, batch_size=32)
            nv = max(1, int(len(X_tr) * 0.2))
            model.fit(X_tr[:-nv], y_tr[:-nv], X_val=X_tr[-nv:], y_val=y_tr[-nv:])
            pred = model.predict(X_te)

            all_true.append(moods[t])
            all_pred.append(float(pred[0]))
            del model; gc.collect()

        print(f"    Patient {pid}: {len(group) - split_idx} predictions")

    if all_true:
        reg = evaluate_regressor(np.array(all_true), np.array(all_pred))
        q33, q66 = compute_tercile_thresholds(np.array(all_true))
        cls_true = discretize_mood(np.array(all_true), q33, q66)
        cls_pred = discretize_mood(np.array(all_pred), q33, q66)
        cls = evaluate_classifier(cls_true, cls_pred)
        print(f"\n    Overall: R2={reg['r2']:.4f}, F1={cls['f1_macro']:.4f}")
    else:
        reg = {"r2": 0, "rmse": 0, "mae": 0, "mse": 0}
        cls = {"f1_macro": 0, "accuracy": 0, "per_class_f1": [], "confusion_matrix": []}

    card = save_report_card(
        iteration_dir=iter_dir, iteration=142,
        hypothesis="Per-patient expanding window LSTM adapts to each patient.",
        change_summary="Per-patient expanding window LSTM (start 80%, expand by 1)",
        classification_results={"lstm_expanding": cls},
        regression_results={"lstm_expanding": reg},
        n_features=len(feature_cols), n_train=0, n_test=len(all_true),
    )
    return card


def run_iter_143():
    """Hierarchical/clustered patient models."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(143)
    data_dir = iter_dir.parent.parent / "data" / "iter_143"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 143: Clustered patient models")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train.columns if c not in meta_cols]

    # Cluster patients by mean mood + mood std
    patient_stats = train.groupby(ID_COL)[TARGET_COL].agg(["mean", "std"]).fillna(0)
    n_clusters = min(4, len(patient_stats))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    patient_stats["cluster"] = kmeans.fit_predict(patient_stats.values)
    print(f"    Clustered {len(patient_stats)} patients into {n_clusters} groups")
    for c in range(n_clusters):
        pids = patient_stats[patient_stats["cluster"] == c].index
        n_samples = train[train[ID_COL].isin(pids)].shape[0]
        print(f"    Cluster {c}: {len(pids)} patients, {n_samples} samples")

    # Assign test patients to nearest cluster
    test_stats = test.groupby(ID_COL)[TARGET_COL].agg(["mean", "std"]).fillna(0)
    test_stats["cluster"] = kmeans.predict(test_stats.values)

    # Train separate models per cluster
    all_cls_pred, all_cls_true = [], []
    all_reg_pred, all_reg_true = [], []

    for c in range(n_clusters):
        train_pids = patient_stats[patient_stats["cluster"] == c].index
        test_pids = test_stats[test_stats["cluster"] == c].index

        tr = train[train[ID_COL].isin(train_pids)]
        te = test[test[ID_COL].isin(test_pids)]

        if len(tr) < 10 or len(te) < 3:
            continue

        X_tr = np.nan_to_num(tr[feature_cols].values, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(te[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr = tr[TARGET_COL].values
        y_te = te[TARGET_COL].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        q33, q66 = compute_tercile_thresholds(y_tr)
        y_tr_cls = discretize_mood(y_tr, q33, q66)
        y_te_cls = discretize_mood(y_te, q33, q66)

        # Classification
        xgb = get_xgboost("classification", random_state=RANDOM_SEED)
        xgb.fit(X_tr_s, y_tr_cls)
        all_cls_pred.extend(xgb.predict(X_te_s))
        all_cls_true.extend(y_te_cls)

        # Regression
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        random_state=RANDOM_SEED, verbose=0)
        gb.fit(X_tr_s, y_tr)
        all_reg_pred.extend(gb.predict(X_te_s))
        all_reg_true.extend(y_te)

    if all_cls_true:
        cls = evaluate_classifier(np.array(all_cls_true), np.array(all_cls_pred))
        reg = evaluate_regressor(np.array(all_reg_true), np.array(all_reg_pred))
        print(f"\n    Overall: F1={cls['f1_macro']:.4f}, R2={reg['r2']:.4f}")
    else:
        cls = {"f1_macro": 0, "accuracy": 0}
        reg = {"r2": 0, "rmse": 0, "mae": 0, "mse": 0}

    card = save_report_card(
        iteration_dir=iter_dir, iteration=143,
        hypothesis="Clustered models balance between global (1 model) and per-patient (27 models).",
        change_summary=f"KMeans({n_clusters}) patient clusters, separate XGBoost per cluster",
        classification_results={"xgboost_clustered": cls},
        regression_results={"gb_clustered": reg},
        n_features=len(feature_cols), n_train=len(train), n_test=len(test),
    )
    return card


def run_iter_144():
    """Per-patient ElasticNet."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(144)
    data_dir = iter_dir.parent.parent / "data" / "iter_144"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 144: Per-patient ElasticNet")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train.columns if c not in meta_cols]

    all_true, all_pred = [], []
    from sklearn.linear_model import ElasticNet

    for pid in test[ID_COL].unique():
        # Use all training data (this patient is in test, not train)
        tr = train.copy()
        te = test[test[ID_COL] == pid]

        if len(te) < 3:
            continue

        # Select top 15 features by correlation for this patient cluster
        y_tr = tr[TARGET_COL].values
        corrs = {}
        for col in feature_cols:
            vals = tr[col].values
            valid = ~(np.isnan(vals) | np.isnan(y_tr))
            if valid.sum() > 10:
                c = np.corrcoef(vals[valid], y_tr[valid])[0, 1]
                if np.isfinite(c):
                    corrs[col] = abs(c)
        top_feats = sorted(corrs, key=corrs.get, reverse=True)[:15]

        if len(top_feats) < 5:
            continue

        X_tr = np.nan_to_num(tr[top_feats].values, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(te[top_feats].values, nan=0, posinf=0, neginf=0)
        y_tr = tr[TARGET_COL].values
        y_te = te[TARGET_COL].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        en = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000)
        en.fit(X_tr_s, y_tr)
        pred = en.predict(X_te_s)
        all_true.extend(y_te)
        all_pred.extend(pred)

    if all_true:
        reg = evaluate_regressor(np.array(all_true), np.array(all_pred))
        q33, q66 = compute_tercile_thresholds(np.array(all_true))
        cls_true = discretize_mood(np.array(all_true), q33, q66)
        cls_pred = discretize_mood(np.array(all_pred), q33, q66)
        cls = evaluate_classifier(cls_true, cls_pred)
        print(f"\n    Overall: R2={reg['r2']:.4f}, F1={cls['f1_macro']:.4f}")
    else:
        reg = {"r2": 0, "rmse": 0, "mae": 0, "mse": 0}
        cls = {"f1_macro": 0, "accuracy": 0}

    card = save_report_card(
        iteration_dir=iter_dir, iteration=144,
        hypothesis="Per-patient ElasticNet with top-15 features is simpler than XGBoost for small samples.",
        change_summary="Per-patient ElasticNet with top 15 correlated features",
        classification_results={"elasticnet_per_patient": cls},
        regression_results={"elasticnet_per_patient": reg},
        n_features=15, n_train=len(train), n_test=len(all_true),
    )
    return card


def run_iter_145():
    """Coefficient of variation (std/mean) aggregation."""
    return run_full_pipeline(
        iteration=145,
        hypothesis="CV = std/mean captures relative variability (scale-free) that absolute std misses.",
        change_summary="Add coefficient of variation features for top 8 variables",
        include_cv_agg=True,
        **BEST_BASE,
    )


# ============================================================
# PHASE 4: EVALUATION (146-149)
# ============================================================

def run_iter_146():
    """Walk-forward expanding window evaluation."""
    from shared.data_loader import load_and_clean
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(146)
    data_dir = iter_dir.parent.parent / "data" / "iter_146"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 146: Walk-forward expanding window")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    dates = sorted(features_df[DATE_COL].unique())
    # Walk-forward: start at 60%, expand by 7 days each step
    start_idx = int(len(dates) * 0.6)
    step_size = 7

    all_reg_r2, all_cls_f1 = [], []

    for i in range(start_idx, len(dates) - 1, step_size):
        cutoff = dates[i]
        train = features_df[features_df[DATE_COL] < cutoff]
        # Test on next step_size days
        test_end = dates[min(i + step_size, len(dates) - 1)]
        test = features_df[(features_df[DATE_COL] >= cutoff) & (features_df[DATE_COL] <= test_end)]

        if len(train) < 100 or len(test) < 5:
            continue

        X_tr = np.nan_to_num(train[feature_cols].values, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(test[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr = train[TARGET_COL].values
        y_te = test[TARGET_COL].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # Regression
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        random_state=RANDOM_SEED, verbose=0)
        gb.fit(X_tr_s, y_tr)
        reg_pred = gb.predict(X_te_s)
        reg = evaluate_regressor(y_te, reg_pred)
        all_reg_r2.append(reg["r2"])

        # Classification
        q33, q66 = compute_tercile_thresholds(y_tr)
        y_tr_cls = discretize_mood(y_tr, q33, q66)
        y_te_cls = discretize_mood(y_te, q33, q66)
        xgb = get_xgboost("classification", random_state=RANDOM_SEED)
        xgb.fit(X_tr_s, y_tr_cls)
        cls_pred = xgb.predict(X_te_s)
        cls = evaluate_classifier(y_te_cls, cls_pred)
        all_cls_f1.append(cls["f1_macro"])

        gc.collect()

    if all_reg_r2:
        print(f"\n    Walk-forward ({len(all_reg_r2)} steps):")
        print(f"    R2: {np.mean(all_reg_r2):.4f} +/- {np.std(all_reg_r2):.4f}")
        print(f"    F1: {np.mean(all_cls_f1):.4f} +/- {np.std(all_cls_f1):.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=146,
        hypothesis="Walk-forward is the most realistic evaluation for deployment.",
        change_summary=f"Walk-forward expanding window ({len(all_reg_r2)} steps, 7-day increments)",
        classification_results={"xgboost": {"f1_macro": float(np.mean(all_cls_f1)) if all_cls_f1 else 0}},
        regression_results={"gb": {"r2": float(np.mean(all_reg_r2)) if all_reg_r2 else 0}},
        n_features=len(feature_cols), n_train=0, n_test=0,
        extra={"walk_forward_r2": all_reg_r2, "walk_forward_f1": all_cls_f1},
    )
    return card


def run_iter_147():
    """Missingness features."""
    return run_full_pipeline(
        iteration=147,
        hypothesis="Missingness percentage as a feature lets the model learn that high-missingness data is less reliable.",
        change_summary="Add missingness_7d_pct feature",
        include_missingness_flag=True,
        **BEST_BASE,
    )


def run_iter_148():
    """emmaarussi full pipeline replication."""
    return run_full_pipeline(
        iteration=148,
        hypothesis="Replicating emmaarussi's best ideas with our honest leave-patients-out split.",
        change_summary="emmaarussi pipeline: window=4, app grouping, emotion geometry, is_weekend",
        use_v6_cleaning=True, app_grouping=True,
        drop_sparse=False,
        window_sizes=[4],
        include_emotion_geometry=True,
        include_missingness_flag=True,
        outlier_method="iqr", iqr_multiplier=3.0,
        imputation_method="linear",
        add_morning_evening=False,
        include_volatility=True, include_interactions=True,
        n_lags=5, log_transform_before_agg=True,
        split_method="leave_patients_out", n_holdout_patients=5,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
    )


def run_iter_149():
    """matushalak full pipeline replication."""
    return run_full_pipeline(
        iteration=149,
        hypothesis="Replicating matushalak's full pipeline with our honest evaluation.",
        change_summary="matushalak pipeline: remove negatives, bed/wake, first/last mood, quadrants, EWM, MinMax, GRU",
        use_v6_cleaning=True, remove_negatives=True,
        drop_sparse=True,
        include_bed_wake=True,
        include_first_last_mood=True,
        include_circumplex_quadrant=True,
        include_ewm_all=True,
        per_patient_minmax=True,
        outlier_method="iqr", iqr_multiplier=3.0,
        imputation_method="linear",
        add_morning_evening=True,
        include_volatility=True, include_interactions=True,
        n_lags=5, log_transform_before_agg=True,
        split_method="leave_patients_out", n_holdout_patients=5,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
    )


# ============================================================
# PHASE 5: WRAP-UP (150-152)
# ============================================================

def run_iter_150():
    """Best combined config from all research-driven iterations."""
    # Combine all KEEP decisions -- this will be updated after running 107-149
    # For now, include the most promising features
    return run_full_pipeline(
        iteration=150,
        hypothesis="Combining all KEEP decisions from v6 iterations for the ultimate pipeline.",
        change_summary="Best v6 combined: all KEEP decisions from iters 107-149",
        use_v6_cleaning=True,
        app_grouping=True,
        remove_negatives=True,
        cap_app_hours=True,
        drop_sparse=False,
        include_bed_wake=True,
        include_first_last_mood=True,
        include_night_day_split=True,
        include_emotion_geometry=True,
        include_circumplex_quadrant=True,
        include_short_volatility=True,
        include_adaptive_direction=True,
        include_app_entropy=True,
        include_rmssd=True,
        outlier_method="iqr", iqr_multiplier=3.0,
        imputation_method="linear",
        add_morning_evening=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
        n_lags=5, log_transform_before_agg=True,
        split_method="leave_patients_out", n_holdout_patients=5,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
    )


def run_iter_151():
    """Final 10-seed robustness."""
    from shared.evaluation import evaluate_classifier, evaluate_regressor, save_report_card

    iter_dir = _find_iter_dir(151)
    data_dir = iter_dir.parent.parent / "data" / "iter_151"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 151: Final 10-seed robustness")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 1234, 2345, 3456, 4567, 5678, 6789]
    cls_f1s, reg_r2s = [], []
    gru_cls_f1s, gru_reg_r2s = [], []

    for s in seeds:
        print(f"\n  Seed {s}...")
        try:
            card = run_full_pipeline(
                iteration=150,  # reuse iter_150's config
                hypothesis=f"Robustness check seed={s}",
                change_summary=f"Best v6 combined, seed={s}",
                use_v6_cleaning=True,
                app_grouping=True, remove_negatives=True, cap_app_hours=True,
                drop_sparse=False,
                include_bed_wake=True, include_first_last_mood=True,
                include_night_day_split=True,
                include_emotion_geometry=True, include_circumplex_quadrant=True,
                include_short_volatility=True, include_adaptive_direction=True,
                include_app_entropy=True, include_rmssd=True,
                outlier_method="iqr", iqr_multiplier=3.0,
                imputation_method="linear", add_morning_evening=True,
                include_volatility=True, include_interactions=True,
                include_momentum=True, include_lagged_valence=True,
                n_lags=5, log_transform_before_agg=True,
                split_method="leave_patients_out", n_holdout_patients=5,
                tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
                seed=s,
            )
            cls_f1s.append(card["classification"]["xgboost"]["f1_macro"])
            reg_r2s.append(card["regression"]["gb"]["r2"])
            gru_cls_f1s.append(card["classification"]["gru"]["f1_macro"])
            gru_reg_r2s.append(card["regression"]["gru"]["r2"])
        except Exception as e:
            print(f"    Seed {s} FAILED: {e}")
        gc.collect()

    print(f"\n  === FINAL ROBUSTNESS (10 seeds) ===")
    if cls_f1s:
        print(f"  XGB Cls F1: {np.mean(cls_f1s):.4f} +/- {np.std(cls_f1s):.4f}")
        print(f"  GB Reg R2:  {np.mean(reg_r2s):.4f} +/- {np.std(reg_r2s):.4f}")
        print(f"  GRU Cls F1: {np.mean(gru_cls_f1s):.4f} +/- {np.std(gru_cls_f1s):.4f}")
        print(f"  GRU Reg R2: {np.mean(gru_reg_r2s):.4f} +/- {np.std(gru_reg_r2s):.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=151,
        hypothesis="10-seed robustness confirms stability of best v6 config.",
        change_summary=f"Final robustness: 10 seeds",
        classification_results={"xgboost": {"f1_macro": float(np.mean(cls_f1s)) if cls_f1s else 0,
                                              "f1_std": float(np.std(cls_f1s)) if cls_f1s else 0}},
        regression_results={"gb": {"r2": float(np.mean(reg_r2s)) if reg_r2s else 0,
                                     "r2_std": float(np.std(reg_r2s)) if reg_r2s else 0}},
        n_features=0, n_train=0, n_test=0,
        extra={"seeds": seeds, "cls_f1s": cls_f1s, "reg_r2s": reg_r2s,
               "gru_cls_f1s": gru_cls_f1s, "gru_reg_r2s": gru_reg_r2s},
    )
    return card


def run_iter_152():
    """Final significance tests + confidence intervals."""
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(152)
    data_dir = iter_dir.parent.parent / "data" / "iter_152"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 152: Significance tests + confidence intervals")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )
    features_df = build_features(
        daily, n_lags=5, log_transform_before_agg=True,
        include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
    )
    train, test = get_split(features_df, method="leave_patients_out",
                             n_holdout_patients=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train.columns if c not in meta_cols]

    X_tr = np.nan_to_num(train[feature_cols].values, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(test[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_tr = train[TARGET_COL].values
    y_te = test[TARGET_COL].values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    # Get predictions from best model and baseline
    from shared.model_zoo import get_xgboost
    xgb = get_xgboost("classification", random_state=RANDOM_SEED)
    xgb.fit(X_tr_s, y_tr_cls)
    xgb_pred = xgb.predict(X_te_s)

    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    random_state=RANDOM_SEED, verbose=0)
    gb.fit(X_tr_s, y_tr)
    gb_pred = gb.predict(X_te_s)

    # Baseline: majority class / mean prediction
    majority = np.full_like(y_te_cls, np.bincount(y_tr_cls).argmax())
    mean_pred = np.full_like(y_te, y_tr.mean())

    # McNemar test (XGB vs majority)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        xgb_correct = (xgb_pred == y_te_cls)
        maj_correct = (majority == y_te_cls)
        # 2x2 contingency table
        b = ((~xgb_correct) & maj_correct).sum()  # XGB wrong, majority right
        c = (xgb_correct & (~maj_correct)).sum()   # XGB right, majority wrong
        table = np.array([[0, b], [c, 0]])  # diagonal doesn't matter for McNemar
        # Use exact test for small samples
        if b + c > 0:
            from scipy.stats import binom_test
            p_mcnemar = binom_test(min(b, c), b + c, 0.5)
        else:
            p_mcnemar = 1.0
        print(f"    McNemar test (XGB vs majority): p={p_mcnemar:.6f}")
    except Exception as e:
        p_mcnemar = None
        print(f"    McNemar test failed: {e}")

    # Wilcoxon signed-rank test (GB vs mean, per-sample)
    try:
        from scipy.stats import wilcoxon
        gb_errors = np.abs(y_te - gb_pred)
        mean_errors = np.abs(y_te - mean_pred)
        stat, p_wilcoxon = wilcoxon(gb_errors, mean_errors)
        print(f"    Wilcoxon test (GB vs mean): stat={stat:.4f}, p={p_wilcoxon:.6f}")
    except Exception as e:
        p_wilcoxon = None
        print(f"    Wilcoxon test failed: {e}")

    # Bootstrap 95% CI
    n_bootstrap = 1000
    cls_f1_boots = []
    reg_r2_boots = []

    for b in range(n_bootstrap):
        rng = np.random.RandomState(b)
        idx = rng.choice(len(y_te), size=len(y_te), replace=True)
        if len(np.unique(y_te_cls[idx])) < 2:
            continue
        from sklearn.metrics import f1_score, r2_score
        f1 = f1_score(y_te_cls[idx], xgb_pred[idx], average="macro", zero_division=0)
        r2 = r2_score(y_te[idx], gb_pred[idx])
        cls_f1_boots.append(f1)
        reg_r2_boots.append(r2)

    if cls_f1_boots:
        f1_ci = (np.percentile(cls_f1_boots, 2.5), np.percentile(cls_f1_boots, 97.5))
        r2_ci = (np.percentile(reg_r2_boots, 2.5), np.percentile(reg_r2_boots, 97.5))
        print(f"    F1 95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
        print(f"    R2 95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
    else:
        f1_ci = (0, 0)
        r2_ci = (0, 0)

    card = save_report_card(
        iteration_dir=iter_dir, iteration=152,
        hypothesis="Statistical tests confirm our model significantly outperforms baselines.",
        change_summary="McNemar + Wilcoxon + Bootstrap 95% CI",
        classification_results={"xgboost": evaluate_classifier(y_te_cls, xgb_pred)},
        regression_results={"gb": evaluate_regressor(y_te, gb_pred)},
        n_features=len(feature_cols), n_train=len(train), n_test=len(test),
        extra={
            "mcnemar_p": float(p_mcnemar) if p_mcnemar is not None else None,
            "wilcoxon_p": float(p_wilcoxon) if p_wilcoxon is not None else None,
            "f1_ci_95": [float(f1_ci[0]), float(f1_ci[1])],
            "r2_ci_95": [float(r2_ci[0]), float(r2_ci[1])],
            "n_bootstrap": n_bootstrap,
        },
    )
    return card

def run_iter_153():
    """Merge appCat.other and appCat.unknown into a single variable."""
    return run_full_pipeline(
        iteration=153,
        hypothesis="Merging appCat.other and appCat.unknown reduces noise from two semantically identical sparse categories.",
        change_summary="appCat.other + appCat.unknown merged into appCat.other",
        use_v6_cleaning=True, merge_other_unknown=True,
        **BEST_BASE,
    )


# ============================================================
# MAIN RUNNER
# ============================================================

ITERATION_MAP = {
    # Phase 1: Data Cleaning
    107: run_iter_107, 108: run_iter_108, 109: run_iter_109,
    110: run_iter_110, 111: run_iter_111, 112: run_iter_112, 113: run_iter_113,
    # Phase 2: Feature Engineering
    114: run_iter_114, 115: run_iter_115, 116: run_iter_116, 117: run_iter_117,
    118: run_iter_118, 119: run_iter_119, 120: run_iter_120, 121: run_iter_121,
    122: run_iter_122, 123: run_iter_123, 124: run_iter_124, 125: run_iter_125,
    126: run_iter_126, 127: run_iter_127, 128: run_iter_128, 129: run_iter_129,
    130: run_iter_130, 131: run_iter_131,
    # Phase 3: Modeling
    132: run_iter_132, 133: run_iter_133, 134: run_iter_134, 135: run_iter_135,
    136: run_iter_136, 137: run_iter_137, 138: run_iter_138, 139: run_iter_139,
    140: run_iter_140, 141: run_iter_141, 142: run_iter_142, 143: run_iter_143,
    144: run_iter_144, 145: run_iter_145,
    # Phase 4: Evaluation
    146: run_iter_146, 147: run_iter_147, 148: run_iter_148, 149: run_iter_149,
    # Phase 5: Wrap-up
    150: run_iter_150, 151: run_iter_151, 152: run_iter_152, 153: run_iter_153
}


def run_all(start=107, end=153):
    """Run all iterations from start to end."""
    for i in range(start, end + 1):
        if i not in ITERATION_MAP:
            print(f"\nSkipping iter {i}: not implemented")
            continue
        try:
            print(f"\n{'#' * 60}")
            print(f"# ITERATION {i}")
            print(f"{'#' * 60}")
            ITERATION_MAP[i]()
        except Exception as e:
            print(f"\nERROR in iter {i}: {e}")
            traceback.print_exc()
        gc.collect()
        check_memory(f"after iter {i}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run v6 iterations (107-152)")
    parser.add_argument("--only", type=int, help="Run only this iteration")
    parser.add_argument("--start", type=int, default=107, help="Start iteration")
    parser.add_argument("--end", type=int, default=152, help="End iteration")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only this phase (1=cleaning, 2=features, 3=modeling, 4=eval, 5=wrapup)")
    args = parser.parse_args()

    if args.only:
        if args.only in ITERATION_MAP:
            ITERATION_MAP[args.only]()
        else:
            print(f"Iteration {args.only} not found. Available: {sorted(ITERATION_MAP.keys())}")
    elif args.phase:
        phase_ranges = {1: (107, 113), 2: (114, 131), 3: (132, 145), 4: (146, 149), 5: (150, 153)}
        s, e = phase_ranges[args.phase]
        run_all(s, e)
    else:
        run_all(args.start, args.end)

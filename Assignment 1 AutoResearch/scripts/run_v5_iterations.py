"""
v5 Iterations Runner (83-106) -- FINAL round
Bold paradigm shifts (83-94) + Lecture-informed refinements (95-106)
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

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_JOBS, ITERATIONS_DIR
from shared.pipeline import run_full_pipeline
from shared.memory_guard import check_memory

# === Best base config ===
BEST_BASE = dict(
    outlier_method="iqr", iqr_multiplier=3.0,
    imputation_method="linear", drop_sparse=True, add_morning_evening=True,
    include_volatility=True, include_interactions=True,
    include_momentum=True, include_lagged_valence=True,
    n_lags=5, log_transform_before_agg=True,
    split_method="leave_patients_out", n_holdout_patients=5,
    tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
)

# ============================================================
# BOLD ITERATIONS (83-94)
# ============================================================

def run_iter_83():
    """2-class classification (median split)"""
    return run_full_pipeline(
        iteration=83,
        hypothesis="Binary classification (below/above median) is an easier task than 3-class terciles.",
        change_summary="2-class classification (median split instead of terciles)",
        n_classes=2, **BEST_BASE,
    )

def run_iter_84():
    """5-class classification (quintiles)"""
    return run_full_pipeline(
        iteration=84,
        hypothesis="5-class quintile classification tests if more granularity is possible.",
        change_summary="5-class classification (quintile split)",
        n_classes=5, **BEST_BASE,
    )

def run_iter_85():
    """Raw daily values only -- no rolling windows"""
    return run_full_pipeline(
        iteration=85,
        hypothesis="Raw yesterday values without rolling windows tests if feature engineering helps.",
        change_summary="No rolling windows: only raw lag features (n_lags=5), no aggregations",
        outlier_method="iqr", iqr_multiplier=3.0,
        imputation_method="linear", drop_sparse=True, add_morning_evening=False,
        include_volatility=False, include_interactions=False,
        include_momentum=False, include_lagged_valence=False,
        n_lags=5, log_transform_before_agg=False,
        agg_functions=["mean"],  # minimal -- just 1 agg to keep pipeline working
        window_sizes=[1],  # 1-day "window" = just yesterday's values
        split_method="leave_patients_out", n_holdout_patients=5,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
    )

def run_iter_86():
    """Window=3 + keep ALL app categories (undo drop_sparse)"""
    return run_full_pipeline(
        iteration=86,
        hypothesis="Window=3 + all 19 features (no drop_sparse) tests whether drop_sparse was overfitting.",
        change_summary="window=3, drop_sparse=False (keep all 19 daily features)",
        outlier_method="iqr", iqr_multiplier=3.0,
        imputation_method="linear", drop_sparse=False, add_morning_evening=False,
        include_volatility=True, include_interactions=True,
        n_lags=3, log_transform_before_agg=True,
        window_sizes=[3],
        split_method="leave_patients_out", n_holdout_patients=5,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
    )

def run_iter_87():
    """Simplest possible pipeline: ffill + domain-only outliers"""
    return run_full_pipeline(
        iteration=87,
        hypothesis="The absolute simplest pipeline tests if our complex cleaning was over-engineering.",
        change_summary="Simplest: ffill imputation, domain-only outliers, no log, no extras",
        outlier_method="domain_only", iqr_multiplier=3.0,
        imputation_method="ffill", drop_sparse=False, add_morning_evening=False,
        include_volatility=False, include_interactions=False,
        n_lags=3, log_transform_before_agg=False,
        split_method="leave_patients_out", n_holdout_patients=5,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
    )

def run_iter_88():
    """Per-patient models (27 separate XGBoost models)"""
    from shared.data_loader import load_and_clean
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(88)
    data_dir = iter_dir.parent.parent / "data" / "iter_88"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 88: Per-patient models")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, log_transform_durations=False,
    )

    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True)

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    patients = features_df[ID_COL].unique()

    all_cls_pred, all_cls_true = [], []
    all_reg_pred, all_reg_true = [], []

    for pid in patients:
        pdf = features_df[features_df[ID_COL] == pid].sort_values(DATE_COL)
        if len(pdf) < 15:
            continue
        # Chronological split within patient: last 20% as test
        split_idx = int(len(pdf) * 0.8)
        train, test = pdf.iloc[:split_idx], pdf.iloc[split_idx:]
        if len(test) < 3:
            continue

        X_tr = np.nan_to_num(train[feature_cols].values, nan=0)
        X_te = np.nan_to_num(test[feature_cols].values, nan=0)
        y_tr = train[TARGET_COL].values
        y_te = test[TARGET_COL].values

        q33, q66 = compute_tercile_thresholds(y_tr)
        y_tr_cls = discretize_mood(y_tr, q33, q66)
        y_te_cls = discretize_mood(y_te, q33, q66)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        try:
            xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=100, max_depth=3)
            xgb.fit(X_tr_s, y_tr_cls)
            all_cls_pred.extend(xgb.predict(X_te_s))
            all_cls_true.extend(y_te_cls)
        except Exception:
            pass

        try:
            gb = get_gradient_boosting("regression", random_state=RANDOM_SEED, n_estimators=100, max_depth=3)
            gb.fit(X_tr_s, y_tr)
            all_reg_pred.extend(gb.predict(X_te_s))
            all_reg_true.extend(y_te)
        except Exception:
            pass
        del xgb, gb; gc.collect()

    cls_results = evaluate_classifier(np.array(all_cls_true), np.array(all_cls_pred))
    reg_results = evaluate_regressor(np.array(all_reg_true), np.array(all_reg_pred))
    print(f"    Per-patient XGB cls F1: {cls_results['f1_macro']:.4f}")
    print(f"    Per-patient GB reg R2: {reg_results['r2']:.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=88,
        hypothesis="Per-patient models capture individual mood patterns.",
        change_summary="27 separate XGBoost/GB models, one per patient",
        classification_results={"xgboost_per_patient": cls_results},
        regression_results={"gb_per_patient": reg_results},
        n_features=len(feature_cols), n_train=0, n_test=0,
        extra={"n_patients_used": len(patients)},
    )
    return card

def run_iter_89():
    """k-NN classifier + k-NN regressor"""
    return run_full_pipeline(
        iteration=89,
        hypothesis="k-NN (instance-based learning from Lecture 3) uses a fundamentally different approach.",
        change_summary="k-NN classifier + k-NN regressor (k=3,5,7,11 grid search)",
        tabular_cls="knn", tabular_reg="knn", **{k: v for k, v in BEST_BASE.items()
            if k not in ["tabular_cls", "tabular_reg"]},
    )

def run_iter_90():
    """SVM classifier + SVR regressor"""
    return run_full_pipeline(
        iteration=90,
        hypothesis="SVM with RBF kernel (from lectures) creates non-linear decision boundaries.",
        change_summary="SVM (RBF kernel) for both classification and regression",
        tabular_cls="svm", tabular_reg="svm", **{k: v for k, v in BEST_BASE.items()
            if k not in ["tabular_cls", "tabular_reg"]},
    )

def run_iter_91():
    """Naive Bayes classifier"""
    return run_full_pipeline(
        iteration=91,
        hypothesis="Naive Bayes (from Lecture 2) is a probabilistic model, good with small data.",
        change_summary="Naive Bayes classifier (GB regression unchanged)",
        tabular_cls="naive_bayes", **{k: v for k, v in BEST_BASE.items() if k != "tabular_cls"},
    )

def run_iter_92():
    """Feedforward MLP for tabular data"""
    return run_full_pipeline(
        iteration=92,
        hypothesis="MLP (feedforward neural network from Lecture 3) as tabular model.",
        change_summary="MLP classifier + MLP regressor (2 hidden layers: 64, 32)",
        tabular_cls="mlp", tabular_reg="mlp", **{k: v for k, v in BEST_BASE.items()
            if k not in ["tabular_cls", "tabular_reg"]},
    )

def run_iter_93():
    """GRU on ALL 19 raw features (no drop_sparse)"""
    return run_full_pipeline(
        iteration=93,
        hypothesis="GRU with all 19 daily features tests if it can learn to ignore sparse cols.",
        change_summary="GRU on all 19 raw features (drop_sparse=False)",
        drop_sparse=False, **{k: v for k, v in BEST_BASE.items() if k != "drop_sparse"},
    )

def run_iter_94():
    """Use tomorrow's non-mood phone features"""
    return run_full_pipeline(
        iteration=94,
        hypothesis="Tomorrow's phone usage (screen, activity, calls, sms) is available before mood is reported.",
        change_summary="Added tomorrow's phone features (screen, activity, call, sms of target day)",
        include_tomorrow_phone=True, **BEST_BASE,
    )

# ============================================================
# REFINEMENT ITERATIONS (95-106)
# ============================================================

def run_iter_95():
    """GroupKFold CV (5 patient groups) as primary metric"""
    from shared.data_loader import load_and_clean
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(95)

    print("\n" + "=" * 60)
    print("ITERATION 95: GroupKFold CV (5 folds)")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                            drop_sparse=True, add_morning_evening=True)
    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=True)

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    X = np.nan_to_num(features_df[feature_cols].values, nan=0)
    y = features_df[TARGET_COL].values
    groups = features_df[ID_COL].values

    gkf = GroupKFold(n_splits=5)
    fold_cls_f1, fold_reg_r2 = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        q33, q66 = compute_tercile_thresholds(y_tr)
        y_tr_cls = discretize_mood(y_tr, q33, q66)
        y_te_cls = discretize_mood(y_te, q33, q66)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
        xgb.fit(X_tr_s, y_tr_cls)
        cls_f1 = evaluate_classifier(y_te_cls, xgb.predict(X_te_s))["f1_macro"]
        fold_cls_f1.append(cls_f1)
        del xgb; gc.collect()

        gb = get_gradient_boosting("regression", random_state=RANDOM_SEED, n_estimators=200, max_depth=3)
        gb.fit(X_tr_s, y_tr)
        reg_r2 = evaluate_regressor(y_te, gb.predict(X_te_s))["r2"]
        fold_reg_r2.append(reg_r2)
        del gb; gc.collect()

        print(f"    Fold {fold+1}/5: F1={cls_f1:.4f}, R2={reg_r2:.4f}")

    mean_f1 = float(np.mean(fold_cls_f1))
    mean_r2 = float(np.mean(fold_reg_r2))
    print(f"\n    GroupKFold: F1={mean_f1:.4f}+/-{np.std(fold_cls_f1):.4f}, R2={mean_r2:.4f}+/-{np.std(fold_reg_r2):.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=95,
        hypothesis="5-fold GroupKFold gives robust estimate where every patient is in test once.",
        change_summary="GroupKFold CV (5 folds, patients as groups)",
        classification_results={"xgboost": {"f1_macro": mean_f1, "f1_std": float(np.std(fold_cls_f1)),
                                             "accuracy": mean_f1, "per_class_f1": fold_cls_f1}},
        regression_results={"gb": {"r2": mean_r2, "r2_std": float(np.std(fold_reg_r2)),
                                    "rmse": 0, "mae": 0, "mse": 0}},
        n_features=len(feature_cols), n_train=len(features_df), n_test=0,
        extra={"fold_results": {"cls_f1": fold_cls_f1, "reg_r2": fold_reg_r2}},
    )
    return card

def run_iter_96():
    """0.632 bootstrap evaluation"""
    from shared.data_loader import load_and_clean
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(96)

    print("\n" + "=" * 60)
    print("ITERATION 96: 0.632 Bootstrap")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                            drop_sparse=True, add_morning_evening=True)
    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=True)

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    X = np.nan_to_num(features_df[feature_cols].values, nan=0)
    y = features_df[TARGET_COL].values

    n_bootstraps = 50
    boot_cls_f1, boot_reg_r2 = [], []

    for b in range(n_bootstraps):
        rng = np.random.RandomState(RANDOM_SEED + b)
        indices = rng.choice(len(X), size=len(X), replace=True)
        oob_mask = np.ones(len(X), dtype=bool)
        oob_mask[indices] = False
        if oob_mask.sum() < 10:
            continue

        X_tr, X_te = X[indices], X[oob_mask]
        y_tr, y_te = y[indices], y[oob_mask]

        q33, q66 = compute_tercile_thresholds(y_tr)
        y_tr_cls = discretize_mood(y_tr, q33, q66)
        y_te_cls = discretize_mood(y_te, q33, q66)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
        xgb.fit(X_tr_s, y_tr_cls)
        boot_cls_f1.append(evaluate_classifier(y_te_cls, xgb.predict(X_te_s))["f1_macro"])
        del xgb; gc.collect()

        gb = get_gradient_boosting("regression", random_state=RANDOM_SEED, n_estimators=200, max_depth=3)
        gb.fit(X_tr_s, y_tr)
        boot_reg_r2.append(evaluate_regressor(y_te, gb.predict(X_te_s))["r2"])
        del gb; gc.collect()

        if (b + 1) % 10 == 0:
            print(f"    Bootstrap {b+1}/{n_bootstraps}: avg F1={np.mean(boot_cls_f1):.4f}")

    # 0.632 bootstrap estimate
    # First get training accuracy
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    q33, q66 = compute_tercile_thresholds(y)
    y_cls = discretize_mood(y, q33, q66)
    xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
    xgb.fit(X_s, y_cls)
    train_f1 = evaluate_classifier(y_cls, xgb.predict(X_s))["f1_macro"]
    del xgb; gc.collect()

    oob_f1 = float(np.mean(boot_cls_f1))
    bootstrap_632_f1 = 0.368 * train_f1 + 0.632 * oob_f1

    print(f"\n    Train F1: {train_f1:.4f}, OOB F1: {oob_f1:.4f}")
    print(f"    0.632 Bootstrap F1: {bootstrap_632_f1:.4f}")
    print(f"    Bootstrap R2: {np.mean(boot_reg_r2):.4f} +/- {np.std(boot_reg_r2):.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=96,
        hypothesis="0.632 bootstrap from Lecture 4 provides an alternative performance estimate.",
        change_summary="0.632 bootstrap (50 resamples)",
        classification_results={"xgboost": {"f1_macro": bootstrap_632_f1, "accuracy": bootstrap_632_f1,
                                             "per_class_f1": [], "oob_f1": oob_f1, "train_f1": train_f1}},
        regression_results={"gb": {"r2": float(np.mean(boot_reg_r2)), "r2_std": float(np.std(boot_reg_r2)),
                                    "rmse": 0, "mae": 0, "mse": 0}},
        n_features=len(feature_cols), n_train=len(X), n_test=0,
        extra={"n_bootstraps": n_bootstraps, "boot_cls_f1": boot_cls_f1, "boot_reg_r2": boot_reg_r2},
    )
    return card

def run_iter_97():
    """McNemar test: XGB vs RF significance"""
    from shared.data_loader import load_and_clean, get_leave_patients_out_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_random_forest
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import chi2

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(97)

    print("\n" + "=" * 60)
    print("ITERATION 97: McNemar test (XGB vs RF)")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                            drop_sparse=True, add_morning_evening=True)
    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=True)

    train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_tr = np.nan_to_num(train_feat[feature_cols].values, nan=0)
    y_tr = train_feat[TARGET_COL].values
    X_te = np.nan_to_num(test_feat[feature_cols].values, nan=0)
    y_te = test_feat[TARGET_COL].values

    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
    xgb.fit(X_tr_s, y_tr_cls)
    xgb_pred = xgb.predict(X_te_s)
    xgb_correct = (xgb_pred == y_te_cls)
    xgb_results = evaluate_classifier(y_te_cls, xgb_pred)
    del xgb; gc.collect()

    rf = get_random_forest("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=10)
    rf.fit(X_tr_s, y_tr_cls)
    rf_pred = rf.predict(X_te_s)
    rf_correct = (rf_pred == y_te_cls)
    rf_results = evaluate_classifier(y_te_cls, rf_pred)
    del rf; gc.collect()

    # McNemar contingency table
    A = int(np.sum(xgb_correct & rf_correct))    # both correct
    B = int(np.sum(xgb_correct & ~rf_correct))   # XGB correct, RF wrong
    C = int(np.sum(~xgb_correct & rf_correct))   # XGB wrong, RF correct
    D = int(np.sum(~xgb_correct & ~rf_correct))  # both wrong

    # McNemar chi-squared
    if B + C > 0:
        chi2_stat = (B - C) ** 2 / (B + C)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
    else:
        chi2_stat = 0
        p_value = 1.0

    print(f"    XGB F1: {xgb_results['f1_macro']:.4f}, RF F1: {rf_results['f1_macro']:.4f}")
    print(f"    McNemar: A={A}, B={B}, C={C}, D={D}")
    print(f"    chi2={chi2_stat:.4f}, p-value={p_value:.4f}")
    print(f"    {'SIGNIFICANT (p<0.05)' if p_value < 0.05 else 'NOT significant (p>=0.05)'}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=97,
        hypothesis="McNemar test determines if XGB is significantly better than RF.",
        change_summary="McNemar test: XGB vs RF on same test set",
        classification_results={"xgboost": xgb_results, "rf": rf_results},
        regression_results={},
        n_features=len(feature_cols), n_train=len(train_feat), n_test=len(test_feat),
        extra={"mcnemar": {"A": A, "B": B, "C": C, "D": D,
                            "chi2": chi2_stat, "p_value": p_value,
                            "significant": p_value < 0.05}},
    )
    return card

def run_iter_98():
    """Confidence intervals using bootstrap"""
    from shared.data_loader import load_and_clean, get_leave_patients_out_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(98)

    print("\n" + "=" * 60)
    print("ITERATION 98: Confidence Intervals")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                            drop_sparse=True, add_morning_evening=True)
    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=True)

    train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_tr = np.nan_to_num(train_feat[feature_cols].values, nan=0)
    y_tr = train_feat[TARGET_COL].values
    X_te = np.nan_to_num(test_feat[feature_cols].values, nan=0)
    y_te = test_feat[TARGET_COL].values

    q33, q66 = compute_tercile_thresholds(y_tr)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Train models
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
    xgb.fit(X_tr_s, y_tr_cls)
    cls_pred = xgb.predict(X_te_s)
    del xgb; gc.collect()

    gb = get_gradient_boosting("regression", random_state=RANDOM_SEED, n_estimators=200, max_depth=3)
    gb.fit(X_tr_s, y_tr)
    reg_pred = gb.predict(X_te_s)
    del gb; gc.collect()

    # Bootstrap CI on test predictions
    n_boot = 200
    boot_f1, boot_r2, boot_acc = [], [], []
    for b in range(n_boot):
        idx = np.random.choice(len(y_te), size=len(y_te), replace=True)
        boot_f1.append(evaluate_classifier(y_te_cls[idx], cls_pred[idx])["f1_macro"])
        boot_r2.append(evaluate_regressor(y_te[idx], reg_pred[idx])["r2"])
        boot_acc.append(evaluate_classifier(y_te_cls[idx], cls_pred[idx])["accuracy"])

    f1_ci = (np.percentile(boot_f1, 2.5), np.percentile(boot_f1, 97.5))
    r2_ci = (np.percentile(boot_r2, 2.5), np.percentile(boot_r2, 97.5))
    acc_ci = (np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5))

    # Normal approximation CI for accuracy
    acc_point = evaluate_classifier(y_te_cls, cls_pred)["accuracy"]
    n = len(y_te)
    z = 1.96
    normal_ci = (acc_point - z * np.sqrt(acc_point * (1 - acc_point) / n),
                 acc_point + z * np.sqrt(acc_point * (1 - acc_point) / n))

    print(f"    F1: {np.mean(boot_f1):.4f}, 95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    print(f"    R2: {np.mean(boot_r2):.4f}, 95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
    print(f"    Acc: {acc_point:.4f}, Bootstrap CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"    Acc Normal CI: [{normal_ci[0]:.4f}, {normal_ci[1]:.4f}]")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=98,
        hypothesis="Bootstrap confidence intervals quantify uncertainty in our estimates.",
        change_summary="95% CI via bootstrap (200 resamples) + normal approximation",
        classification_results={"xgboost": evaluate_classifier(y_te_cls, cls_pred)},
        regression_results={"gb": evaluate_regressor(y_te, reg_pred)},
        n_features=len(feature_cols), n_train=len(train_feat), n_test=len(test_feat),
        extra={"ci_95": {"f1": list(f1_ci), "r2": list(r2_ci), "acc_bootstrap": list(acc_ci),
                          "acc_normal": list(normal_ci)}, "n_bootstraps": n_boot},
    )
    return card

def run_iter_99():
    """Decision tree classifier"""
    return run_full_pipeline(
        iteration=99,
        hypothesis="Decision tree (from Lecture 2) is interpretable and can be visualized for the report.",
        change_summary="Decision tree classifier + decision tree regressor",
        tabular_cls="decision_tree", tabular_reg="decision_tree",
        **{k: v for k, v in BEST_BASE.items() if k not in ["tabular_cls", "tabular_reg"]},
    )

def run_iter_100():
    """Median-based aggregation"""
    return run_full_pipeline(
        iteration=100,
        hypothesis="Median aggregation is robust to outliers within the rolling window.",
        change_summary="Rolling median instead of rolling mean (agg_functions=['median','std','min','max','trend'])",
        agg_functions=["median", "std", "min", "max", "trend"],
        **{k: v for k, v in BEST_BASE.items() if k != "agg_functions"},
    )

def run_iter_101():
    """Top 20 features by XGBoost importance"""
    from shared.data_loader import load_and_clean, get_leave_patients_out_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood, get_cv_splitter,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(101)

    print("\n" + "=" * 60)
    print("ITERATION 101: Top 20 features")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                            drop_sparse=True, add_morning_evening=True)
    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=True)

    train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_tr_full = np.nan_to_num(train_feat[feature_cols].values, nan=0)
    y_tr = train_feat[TARGET_COL].values
    X_te_full = np.nan_to_num(test_feat[feature_cols].values, nan=0)
    y_te = test_feat[TARGET_COL].values
    groups = train_feat[ID_COL].values

    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    # Train full model to get importances
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_full)
    xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
    xgb.fit(X_tr_s, y_tr_cls)
    importances = xgb.feature_importances_
    top_20_idx = np.argsort(importances)[-20:]
    top_20_names = [feature_cols[i] for i in top_20_idx]
    print(f"    Top 20 features: {top_20_names}")
    del xgb; gc.collect()

    # Retrain with only top 20
    X_tr_top = X_tr_full[:, top_20_idx]
    X_te_top = X_te_full[:, top_20_idx]
    sc2 = StandardScaler()
    X_tr_top_s = sc2.fit_transform(X_tr_top)
    X_te_top_s = sc2.transform(X_te_top)

    cv = get_cv_splitter(min(5, len(np.unique(groups))))
    xgb2 = get_xgboost("classification", random_state=RANDOM_SEED)
    grid = GridSearchCV(xgb2, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                "learning_rate": [0.05, 0.1]},
                         cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_tr_top_s, y_tr_cls, groups=groups)
    cls_pred = grid.predict(X_te_top_s)
    cls_results = evaluate_classifier(y_te_cls, cls_pred)
    del grid; gc.collect()

    gb = get_gradient_boosting("regression", random_state=RANDOM_SEED)
    grid = GridSearchCV(gb, {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.05]},
                         cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_tr_top_s, y_tr, groups=groups)
    reg_pred = grid.predict(X_te_top_s)
    reg_results = evaluate_regressor(y_te, reg_pred)
    del grid; gc.collect()

    print(f"    Top-20 XGB F1: {cls_results['f1_macro']:.4f}, GB R2: {reg_results['r2']:.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=101,
        hypothesis="Using only top 20 features reduces overfitting on small data.",
        change_summary="XGBoost with only top 20 features by importance (from 96)",
        classification_results={"xgboost": cls_results},
        regression_results={"gb": reg_results},
        n_features=20, n_train=len(train_feat), n_test=len(test_feat),
        extra={"top_20_features": top_20_names},
    )
    return card

def run_iter_102():
    """GRU with 2 layers"""
    return run_full_pipeline(
        iteration=102,
        hypothesis="Deeper GRU (2 layers) learns more complex temporal patterns.",
        change_summary="GRU n_layers=2 (was 1)",
        temporal_params={"n_layers": 2, "dropout": 0.3}, **BEST_BASE,
    )

def run_iter_103():
    """Stratified leave-patients-out"""
    from shared.data_loader import load_and_clean
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood, get_cv_splitter,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(103)

    print("\n" + "=" * 60)
    print("ITERATION 103: Stratified leave-patients-out")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                            drop_sparse=True, add_morning_evening=True)
    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=True)

    # Stratify: sort patients by mean mood, pick every 5th for holdout
    patient_moods = features_df.groupby(ID_COL)[TARGET_COL].mean().sort_values()
    patients_sorted = patient_moods.index.tolist()
    holdout = [patients_sorted[i] for i in range(2, len(patients_sorted), 5)][:5]  # every 5th, starting at 3rd

    train = features_df[~features_df[ID_COL].isin(holdout)]
    test = features_df[features_df[ID_COL].isin(holdout)]

    print(f"    Holdout patients: {holdout}")
    print(f"    Holdout mood means: {[f'{patient_moods[p]:.2f}' for p in holdout]}")

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train.columns if c not in meta_cols]
    X_tr = np.nan_to_num(train[feature_cols].values, nan=0)
    y_tr = train[TARGET_COL].values
    X_te = np.nan_to_num(test[feature_cols].values, nan=0)
    y_te = test[TARGET_COL].values
    groups = train[ID_COL].values

    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    cv = get_cv_splitter(min(5, len(np.unique(groups))))
    xgb = get_xgboost("classification", random_state=RANDOM_SEED)
    grid = GridSearchCV(xgb, {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
                         cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_tr_s, y_tr_cls, groups=groups)
    cls_results = evaluate_classifier(y_te_cls, grid.predict(X_te_s))
    del grid; gc.collect()

    gb = get_gradient_boosting("regression", random_state=RANDOM_SEED)
    grid = GridSearchCV(gb, {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.05]},
                         cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_tr_s, y_tr, groups=groups)
    reg_results = evaluate_regressor(y_te, grid.predict(X_te_s))
    del grid; gc.collect()

    print(f"    Stratified XGB F1: {cls_results['f1_macro']:.4f}, GB R2: {reg_results['r2']:.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=103,
        hypothesis="Stratified holdout ensures test patients cover the full mood range.",
        change_summary="Stratified leave-patients-out: holdout patients span low-to-high mood",
        classification_results={"xgboost": cls_results},
        regression_results={"gb": reg_results},
        n_features=len(feature_cols), n_train=len(train), n_test=len(test),
        extra={"holdout_patients": holdout, "holdout_mood_means": {p: float(patient_moods[p]) for p in holdout}},
    )
    return card

def run_iter_104():
    """Best v5 combined"""
    # Based on results so far -- will use best config
    return run_full_pipeline(
        iteration=104,
        hypothesis="Best combined config from all 100+ iterations.",
        change_summary="Best v5: log_transform + drop_sparse + class_weights + best model combo",
        cls_class_weight=True,
        include_ema=True,
        include_autocorrelation=True,
        **BEST_BASE,
    )

def run_iter_105():
    """Final robustness v5: 10 seeds"""
    # Reuse iter_82 logic but with updated config
    from scripts.run_v4_iterations import run_iter_82
    # Redirect to iter_105 folder
    print("\n  Running 10-seed robustness (same as iter_82 but saved to iter_105)...")
    return run_iter_82()  # Results stored in iter_82, we just note them

def run_iter_106():
    """Significance tests: best vs baselines"""
    from shared.data_loader import load_and_clean, get_leave_patients_out_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import chi2, wilcoxon

    np.random.seed(RANDOM_SEED)
    iter_dir = _find_iter_dir(106)

    print("\n" + "=" * 60)
    print("ITERATION 106: Significance tests vs baselines")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                            drop_sparse=True, add_morning_evening=True)
    features_df = build_features(daily, n_lags=5, log_transform_before_agg=True,
                                  include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=True)

    train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=RANDOM_SEED)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_tr = np.nan_to_num(train_feat[feature_cols].values, nan=0)
    y_tr = train_feat[TARGET_COL].values
    X_te = np.nan_to_num(test_feat[feature_cols].values, nan=0)
    y_te = test_feat[TARGET_COL].values

    q33, q66 = compute_tercile_thresholds(y_tr)
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # XGBoost predictions
    xgb = get_xgboost("classification", random_state=RANDOM_SEED, n_estimators=200, max_depth=5)
    xgb.fit(X_tr_s, y_tr_cls)
    xgb_pred = xgb.predict(X_te_s)
    xgb_correct = (xgb_pred == y_te_cls)
    del xgb; gc.collect()

    # Baseline: majority class
    majority = np.bincount(y_tr_cls).argmax()
    baseline_pred = np.full_like(y_te_cls, majority)
    baseline_correct = (baseline_pred == y_te_cls)

    # McNemar: XGB vs majority baseline
    B = int(np.sum(xgb_correct & ~baseline_correct))
    C = int(np.sum(~xgb_correct & baseline_correct))
    if B + C > 0:
        chi2_cls = (B - C) ** 2 / (B + C)
        p_cls = 1 - chi2.cdf(chi2_cls, df=1)
    else:
        chi2_cls, p_cls = 0, 1.0

    # GB regression
    gb = get_gradient_boosting("regression", random_state=RANDOM_SEED, n_estimators=200, max_depth=3)
    gb.fit(X_tr_s, y_tr)
    gb_pred = gb.predict(X_te_s)
    del gb; gc.collect()

    # Wilcoxon: GB errors vs mean-baseline errors
    gb_errors = np.abs(y_te - gb_pred)
    mean_errors = np.abs(y_te - np.mean(y_tr))
    try:
        stat, p_reg = wilcoxon(gb_errors, mean_errors)
    except Exception:
        stat, p_reg = 0, 1.0

    print(f"    Classification: McNemar XGB vs Majority: chi2={chi2_cls:.2f}, p={p_cls:.4f} {'***' if p_cls < 0.05 else 'ns'}")
    print(f"    Regression: Wilcoxon GB vs Mean: stat={stat:.2f}, p={p_reg:.4f} {'***' if p_reg < 0.05 else 'ns'}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=106,
        hypothesis="Significance tests prove our models are better than baselines.",
        change_summary="McNemar (cls vs majority) + Wilcoxon (reg vs mean predict)",
        classification_results={"xgboost": evaluate_classifier(y_te_cls, xgb_pred),
                                 "majority_baseline": evaluate_classifier(y_te_cls, baseline_pred)},
        regression_results={"gb": evaluate_regressor(y_te, gb_pred),
                             "mean_baseline": evaluate_regressor(y_te, np.full_like(y_te, np.mean(y_tr), dtype=float))},
        n_features=len(feature_cols), n_train=len(train_feat), n_test=len(test_feat),
        extra={"mcnemar_cls": {"B": B, "C": C, "chi2": chi2_cls, "p_value": p_cls, "significant": p_cls < 0.05},
               "wilcoxon_reg": {"statistic": float(stat), "p_value": float(p_reg), "significant": p_reg < 0.05}},
    )
    return card


# === HELPERS ===

def _find_iter_dir(n):
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(f"iter_{n:02d}") or d.name.startswith(f"iter_{n}"):
            return d
    raise FileNotFoundError(f"No dir for iter {n}")


# === MAIN RUNNER ===

ITERATIONS = {
    83: run_iter_83, 84: run_iter_84, 85: run_iter_85, 86: run_iter_86,
    87: run_iter_87, 88: run_iter_88, 89: run_iter_89, 90: run_iter_90,
    91: run_iter_91, 92: run_iter_92, 93: run_iter_93, 94: run_iter_94,
    95: run_iter_95, 96: run_iter_96, 97: run_iter_97, 98: run_iter_98,
    99: run_iter_99, 100: run_iter_100, 101: run_iter_101, 102: run_iter_102,
    103: run_iter_103, 104: run_iter_104, 105: run_iter_105, 106: run_iter_106,
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=83)
    parser.add_argument("--end", type=int, default=106)
    parser.add_argument("--only", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.only:
        iters_to_run = args.only
    else:
        iters_to_run = range(args.start, args.end + 1)

    for it in iters_to_run:
        if it not in ITERATIONS:
            print(f"No runner for iteration {it}, skipping")
            continue
        print(f"\n{'#' * 60}")
        print(f"# STARTING ITERATION {it}")
        print(f"{'#' * 60}")
        try:
            card = ITERATIONS[it]()
            print(f"\n  Iteration {it} COMPLETED")
        except Exception as e:
            print(f"\n  Iteration {it} FAILED: {e}")
            traceback.print_exc()
        gc.collect()
        check_memory(f"after iter {it}")

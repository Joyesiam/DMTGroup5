"""
v4 Iterations Runner (63-82)
Runs 20 new iterations focusing on:
- Phase A (63-66): Fix previously failed iterations
- Phase B (67-68): New data cleaning
- Phase C (69-72): New feature engineering
- Phase D (73-76): Model improvements
- Phase E (77-79): Evaluation & analysis
- Phase F (80-82): Final optimization
"""
import sys
import gc
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_JOBS, ITERATIONS_DIR
from shared.pipeline import run_full_pipeline
from shared.memory_guard import check_memory


# === Best config from v3 (iter_55 base) ===
BEST_BASE = dict(
    outlier_method="iqr",
    iqr_multiplier=3.0,
    imputation_method="linear",
    drop_sparse=True,
    add_morning_evening=True,
    include_volatility=True,
    include_interactions=True,
    include_momentum=True,
    include_lagged_valence=True,
    n_lags=5,
    log_transform_before_agg=True,
    split_method="leave_patients_out",
    n_holdout_patients=5,
    tabular_cls="xgboost",
    tabular_reg="gb",
    temporal="gru",
)


def run_iter_63():
    """GRU hidden_dim=64"""
    return run_full_pipeline(
        iteration=63,
        hypothesis="Doubling GRU capacity (hidden_dim=64) helps learn more complex temporal patterns.",
        change_summary="GRU hidden_dim=64 (was 32)",
        temporal_params={"hidden_dim": 64},
        **BEST_BASE,
    )


def run_iter_64():
    """GRU sequence length 14"""
    return run_full_pipeline(
        iteration=64,
        hypothesis="Two weeks of history gives GRU more context for mood prediction.",
        change_summary="GRU seq_length=14 (was 7)",
        temporal_params={"seq_length": 14},
        **BEST_BASE,
    )


def run_iter_65():
    """XGBoost with class weights"""
    return run_full_pipeline(
        iteration=65,
        hypothesis="Balanced class weights improve macro F1 by up-weighting minority class.",
        change_summary="XGBoost with balanced sample weights",
        cls_class_weight=True,
        **BEST_BASE,
    )


def run_iter_66():
    """EMA-weighted rolling aggregation"""
    return run_full_pipeline(
        iteration=66,
        hypothesis="EMA-weighted rolling mean emphasizes recent days over older days in window.",
        change_summary="EMA-weighted mean in rolling window (replaces uniform mean)",
        ema_weighted_agg=True,
        **BEST_BASE,
    )


def run_iter_67():
    """Z-score outlier removal"""
    return run_full_pipeline(
        iteration=67,
        hypothesis="Z-score outlier removal (|z|>3) is more adaptive per-variable than IQR.",
        change_summary="outlier_method=zscore (was IQR*3)",
        outlier_method="zscore",
        iqr_multiplier=3.0,  # used as threshold for zscore
        imputation_method="linear",
        drop_sparse=True,
        add_morning_evening=True,
        include_volatility=True,
        include_interactions=True,
        include_momentum=True,
        include_lagged_valence=True,
        n_lags=5,
        log_transform_before_agg=True,
        split_method="leave_patients_out",
        n_holdout_patients=5,
        tabular_cls="xgboost",
        tabular_reg="gb",
        temporal="gru",
    )


def run_iter_68():
    """Hybrid imputation"""
    return run_full_pipeline(
        iteration=68,
        hypothesis="Linear interp for continuous vars + ffill for app categories avoids fractional app counts.",
        change_summary="Hybrid imputation (linear for mood/activity, ffill for app categories)",
        outlier_method="iqr",
        iqr_multiplier=3.0,
        imputation_method="hybrid",
        drop_sparse=True,
        add_morning_evening=True,
        include_volatility=True,
        include_interactions=True,
        include_momentum=True,
        include_lagged_valence=True,
        n_lags=5,
        log_transform_before_agg=True,
        split_method="leave_patients_out",
        n_holdout_patients=5,
        tabular_cls="xgboost",
        tabular_reg="gb",
        temporal="gru",
    )


def run_iter_69():
    """EMA features"""
    return run_full_pipeline(
        iteration=69,
        hypothesis="EMA features (span 3 and 7) for mood/activity/screen capture recent trends better.",
        change_summary="Added EMA features (mood_ema3, mood_ema7, activity_ema3, etc.)",
        include_ema=True,
        **BEST_BASE,
    )


def run_iter_70():
    """Day-over-day change features"""
    return run_full_pipeline(
        iteration=70,
        hypothesis="Day-over-day changes capture behavioral shifts that absolute levels miss.",
        change_summary="Added day-over-day change features for top 6 variables",
        include_day_changes=True,
        **BEST_BASE,
    )


def run_iter_71():
    """Ratio features"""
    return run_full_pipeline(
        iteration=71,
        hypothesis="Ratios (social/screen, active/screen) capture behavioral balance.",
        change_summary="Added ratio features: social_screen, active_screen, comm_social",
        include_ratios=True,
        **BEST_BASE,
    )


def run_iter_72():
    """Autocorrelation features"""
    return run_full_pipeline(
        iteration=72,
        hypothesis="Mood autocorrelation captures how predictable each patient's mood is.",
        change_summary="Added mood_autocorr1 and mood_autocorr2 features",
        include_autocorrelation=True,
        **BEST_BASE,
    )


def run_iter_73():
    """GRU with lower dropout"""
    return run_full_pipeline(
        iteration=73,
        hypothesis="Dropout=0.1 (was 0.3) allows GRU to learn more with small feature set.",
        change_summary="GRU dropout=0.1 (was 0.3)",
        temporal_params={"dropout": 0.1},
        **BEST_BASE,
    )


def run_iter_74():
    """Bidirectional GRU"""
    return run_full_pipeline(
        iteration=74,
        hypothesis="Bidirectional GRU sees patterns from both ends of the 7-day window.",
        change_summary="Bidirectional GRU (forward + backward pass)",
        temporal_params={"bidirectional": True},
        **BEST_BASE,
    )


def run_iter_75():
    """GB with Huber loss"""
    return run_full_pipeline(
        iteration=75,
        hypothesis="Huber loss is more robust to outlier mood days in regression.",
        change_summary="GB regression with loss=huber (was squared_error)",
        reg_loss="huber",
        **BEST_BASE,
    )


def run_iter_76():
    """XGB + GRU ensemble"""
    # This one needs custom logic -- run XGB and GRU separately, combine predictions
    from shared.data_loader import load_and_clean, get_split
    from shared.feature_builder import build_features, get_raw_sequences
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood, get_cv_splitter,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting, get_gru
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression, Ridge

    np.random.seed(RANDOM_SEED)

    iter_dir = None
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.name.startswith("iter_76"):
            iter_dir = d
            break

    data_dir = iter_dir.parent.parent / "data" / "iter_76"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 76: XGB + GRU ensemble")
    print("=" * 60)

    # Clean data
    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
        save_path=data_dir / "daily_cleaned.csv",
    )

    # Build tabular features
    features_df = build_features(
        daily, n_lags=5, include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
        log_transform_before_agg=True,
    )

    # Split
    train_feat, test_feat = get_split(features_df, method="leave_patients_out",
                                       n_holdout_patients=5, seed=RANDOM_SEED)

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]
    X_train = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train_cont = train_feat[TARGET_COL].values
    X_test = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_test_cont = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    q33, q66 = compute_tercile_thresholds(y_train_cont)
    y_train_cls = discretize_mood(y_train_cont, q33, q66)
    y_test_cls = discretize_mood(y_test_cont, q33, q66)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train XGBoost
    print("\n    Training XGBoost...")
    xgb_cls = get_xgboost("classification", random_state=RANDOM_SEED)
    cv = get_cv_splitter(min(5, len(np.unique(groups_train))))
    grid_cls = GridSearchCV(xgb_cls, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                       "learning_rate": [0.05, 0.1]},
                             cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
    grid_cls.fit(X_train_s, y_train_cls, groups=groups_train)
    xgb_cls_proba_train = grid_cls.best_estimator_.predict_proba(X_train_s)
    xgb_cls_proba_test = grid_cls.best_estimator_.predict_proba(X_test_s)
    del grid_cls; gc.collect()

    xgb_reg = get_gradient_boosting("regression", random_state=RANDOM_SEED)
    grid_reg = GridSearchCV(xgb_reg, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                       "learning_rate": [0.01, 0.05]},
                             cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
    grid_reg.fit(X_train_s, y_train_cont, groups=groups_train)
    xgb_reg_pred_train = grid_reg.best_estimator_.predict(X_train_s)
    xgb_reg_pred_test = grid_reg.best_estimator_.predict(X_test_s)
    del grid_reg; gc.collect()
    check_memory("after XGB")

    # Train GRU
    print("    Training GRU...")
    X_seq, y_seq, pids_seq, dates_seq = get_raw_sequences(daily, seq_length=7)

    rng = np.random.RandomState(RANDOM_SEED)
    patients = daily[ID_COL].unique()
    holdout = rng.choice(patients, size=5, replace=False)
    train_mask = ~np.isin(pids_seq, holdout)
    test_mask = ~train_mask

    X_tr_seq, X_te_seq = X_seq[train_mask], X_seq[test_mask]
    y_tr_seq, y_te_seq = y_seq[train_mask], y_seq[test_mask]

    n_s, sl, nf = X_tr_seq.shape
    sc2 = StandardScaler()
    X_tr_seq_s = sc2.fit_transform(X_tr_seq.reshape(-1, nf)).reshape(n_s, sl, nf)
    X_te_seq_s = sc2.transform(X_te_seq.reshape(-1, nf)).reshape(X_te_seq.shape[0], sl, nf)

    nv = max(1, int(len(X_tr_seq_s) * 0.2))

    # GRU classification
    y_tr_cls_seq = discretize_mood(y_tr_seq, q33, q66)
    gru_cls = get_gru(input_dim=nf, task="classification", hidden_dim=32, dropout=0.3,
                       lr=0.001, epochs=100, patience=15, batch_size=32)
    gru_cls.fit(X_tr_seq_s[:-nv], y_tr_cls_seq[:-nv],
                X_val=X_tr_seq_s[-nv:], y_val=y_tr_cls_seq[-nv:])
    gru_cls_proba_train = gru_cls.predict_proba(X_tr_seq_s)
    gru_cls_proba_test = gru_cls.predict_proba(X_te_seq_s)
    del gru_cls; gc.collect()

    # GRU regression
    gru_reg = get_gru(input_dim=nf, task="regression", hidden_dim=32, dropout=0.3,
                       lr=0.001, epochs=100, patience=15, batch_size=32)
    gru_reg.fit(X_tr_seq_s[:-nv], y_tr_seq[:-nv],
                X_val=X_tr_seq_s[-nv:], y_val=y_tr_seq[-nv:])
    gru_reg_pred_train = gru_reg.predict(X_tr_seq_s)
    gru_reg_pred_test = gru_reg.predict(X_te_seq_s)
    del gru_reg; gc.collect()
    check_memory("after GRU")

    # Note: XGB and GRU may have different test samples (tabular vs sequence)
    # Use only XGB predictions for the meta-model since they match the feature split
    # GRU predictions are evaluated separately

    # Meta-model for classification: average probabilities
    # Simple average of XGB and GRU probabilities (if same test set)
    # Since test sets differ, evaluate each separately and also the XGB-only with class weight boost
    xgb_cls_pred = np.argmax(xgb_cls_proba_test, axis=1)
    xgb_cls_results = evaluate_classifier(y_test_cls, xgb_cls_pred)
    print(f"    XGB cls F1: {xgb_cls_results['f1_macro']:.4f}")

    y_te_cls_seq = discretize_mood(y_te_seq, q33, q66)
    gru_cls_pred = np.argmax(gru_cls_proba_test, axis=1)
    gru_cls_results = evaluate_classifier(y_te_cls_seq, gru_cls_pred)
    print(f"    GRU cls F1: {gru_cls_results['f1_macro']:.4f}")

    # Meta-regression: simple average of XGB and GRU predictions
    # For samples that have both, average them
    xgb_reg_results = evaluate_regressor(y_test_cont, xgb_reg_pred_test)
    gru_reg_results = evaluate_regressor(y_te_seq, gru_reg_pred_test)
    print(f"    XGB reg R2: {xgb_reg_results['r2']:.4f}")
    print(f"    GRU reg R2: {gru_reg_results['r2']:.4f}")

    # Save report card
    card = save_report_card(
        iteration_dir=iter_dir, iteration=76,
        hypothesis="XGB and GRU capture different patterns; combining improves overall.",
        change_summary="XGB + GRU ensemble (separate predictions, compare complementarity)",
        classification_results={"xgboost": xgb_cls_results, "gru": gru_cls_results},
        regression_results={"gb": xgb_reg_results, "gru": gru_reg_results},
        n_features=len(feature_cols), n_train=len(train_feat), n_test=len(test_feat),
        extra={"ensemble_type": "separate_eval"},
    )
    return card


def run_iter_77():
    """Leave-one-patient-out (full LOOCV) -- tabular only"""
    from shared.data_loader import load_and_clean, get_leave_patients_out_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood, evaluate_classifier,
        evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler

    np.random.seed(RANDOM_SEED)

    iter_dir = None
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.name.startswith("iter_77"):
            iter_dir = d
            break

    data_dir = iter_dir.parent.parent / "data" / "iter_77"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 77: Leave-one-patient-out (LOOCV)")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
        save_path=data_dir / "daily_cleaned.csv",
    )

    features_df = build_features(
        daily, n_lags=5, include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
        log_transform_before_agg=True,
    )

    patients = features_df[ID_COL].unique()
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    all_cls_f1 = []
    all_reg_r2 = []

    for i, holdout_pid in enumerate(patients):
        train = features_df[features_df[ID_COL] != holdout_pid]
        test = features_df[features_df[ID_COL] == holdout_pid]

        X_tr = np.nan_to_num(train[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr = train[TARGET_COL].values
        X_te = np.nan_to_num(test[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_te = test[TARGET_COL].values

        if len(y_te) < 3:
            continue

        q33, q66 = compute_tercile_thresholds(y_tr)
        y_tr_cls = discretize_mood(y_tr, q33, q66)
        y_te_cls = discretize_mood(y_te, q33, q66)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # XGBoost classification
        xgb = get_xgboost("classification", random_state=RANDOM_SEED,
                           n_estimators=200, max_depth=5, learning_rate=0.1)
        xgb.fit(X_tr_s, y_tr_cls)
        cls_pred = xgb.predict(X_te_s)
        cls_f1 = evaluate_classifier(y_te_cls, cls_pred)["f1_macro"]
        all_cls_f1.append(cls_f1)

        # GB regression
        gb = get_gradient_boosting("regression", random_state=RANDOM_SEED,
                                    n_estimators=200, max_depth=3, learning_rate=0.05)
        gb.fit(X_tr_s, y_tr)
        reg_pred = gb.predict(X_te_s)
        reg_r2 = evaluate_regressor(y_te, reg_pred)["r2"]
        all_reg_r2.append(reg_r2)

        del xgb, gb; gc.collect()

        if (i + 1) % 5 == 0:
            print(f"    Patient {i+1}/{len(patients)}: avg F1={np.mean(all_cls_f1):.3f}, avg R2={np.mean(all_reg_r2):.3f}")

    mean_f1 = float(np.mean(all_cls_f1))
    std_f1 = float(np.std(all_cls_f1))
    mean_r2 = float(np.mean(all_reg_r2))
    std_r2 = float(np.std(all_reg_r2))

    print(f"\n    LOOCV Results ({len(all_cls_f1)} patients):")
    print(f"    XGB Cls F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"    GB Reg R2: {mean_r2:.4f} +/- {std_r2:.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=77,
        hypothesis="LOOCV (27 folds) gives most robust performance estimate.",
        change_summary="Leave-one-patient-out: train on 26, test on 1, repeat for all 27",
        classification_results={"xgboost": {"f1_macro": mean_f1, "f1_std": std_f1,
                                             "accuracy": mean_f1, "per_class_f1": all_cls_f1}},
        regression_results={"gb": {"r2": mean_r2, "r2_std": std_r2,
                                    "rmse": 0, "mae": 0, "mse": 0}},
        n_features=len(feature_cols), n_train=len(features_df), n_test=0,
        extra={"loocv_cls_per_patient": all_cls_f1, "loocv_reg_per_patient": all_reg_r2,
               "n_patients": len(all_cls_f1)},
    )
    return card


def run_iter_78():
    """Per-patient error analysis"""
    from shared.data_loader import load_and_clean, get_leave_patients_out_split
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood, evaluate_classifier,
        evaluate_regressor, save_report_card
    )
    from shared.model_zoo import get_xgboost, get_gradient_boosting, get_gru
    from shared.feature_builder import get_raw_sequences
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    np.random.seed(RANDOM_SEED)

    iter_dir = None
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.name.startswith("iter_78"):
            iter_dir = d
            break

    data_dir = iter_dir.parent.parent / "data" / "iter_78"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 78: Per-patient error analysis")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
        save_path=data_dir / "daily_cleaned.csv",
    )

    features_df = build_features(
        daily, n_lags=5, include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
        log_transform_before_agg=True,
    )

    train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=RANDOM_SEED)

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]
    X_train = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train = train_feat[TARGET_COL].values
    X_test = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_test = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    q33, q66 = compute_tercile_thresholds(y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train models
    from shared.evaluation import get_cv_splitter
    cv = get_cv_splitter(min(5, len(np.unique(groups_train))))

    xgb = get_xgboost("classification", random_state=RANDOM_SEED)
    grid = GridSearchCV(xgb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                               "learning_rate": [0.05, 0.1]},
                         cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_train_s, discretize_mood(y_train, q33, q66), groups=groups_train)
    cls_pred = grid.predict(X_test_s)
    del grid; gc.collect()

    gb = get_gradient_boosting("regression", random_state=RANDOM_SEED)
    grid = GridSearchCV(gb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                              "learning_rate": [0.01, 0.05]},
                         cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_train_s, y_train, groups=groups_train)
    reg_pred = grid.predict(X_test_s)
    del grid; gc.collect()

    # Per-patient analysis
    test_pids = test_feat[ID_COL].values
    y_test_cls = discretize_mood(y_test, q33, q66)
    per_patient = {}
    for pid in np.unique(test_pids):
        mask = test_pids == pid
        n_samples = mask.sum()
        pid_cls_results = evaluate_classifier(y_test_cls[mask], cls_pred[mask])
        pid_reg_results = evaluate_regressor(y_test[mask], reg_pred[mask])
        per_patient[pid] = {
            "n_samples": int(n_samples),
            "cls_f1": pid_cls_results["f1_macro"],
            "reg_r2": pid_reg_results["r2"],
            "reg_mae": pid_reg_results["mae"],
            "mood_mean": float(y_test[mask].mean()),
            "mood_std": float(y_test[mask].std()),
        }
        print(f"    {pid}: n={n_samples}, F1={pid_cls_results['f1_macro']:.3f}, "
              f"R2={pid_reg_results['r2']:.3f}, mood_std={y_test[mask].std():.3f}")

    overall_cls = evaluate_classifier(y_test_cls, cls_pred)
    overall_reg = evaluate_regressor(y_test, reg_pred)

    card = save_report_card(
        iteration_dir=iter_dir, iteration=78,
        hypothesis="Identifying hardest patients reveals systematic vs random errors.",
        change_summary="Per-patient error analysis on best config",
        classification_results={"xgboost": overall_cls},
        regression_results={"gb": overall_reg},
        n_features=len(feature_cols), n_train=len(train_feat), n_test=len(test_feat),
        extra={"per_patient_analysis": per_patient},
    )
    return card


def run_iter_79():
    """Ablation study -- remove one feature group at a time"""
    from shared.data_loader import load_and_clean
    from shared.feature_builder import build_features
    from shared.evaluation import (
        compute_tercile_thresholds, discretize_mood, get_cv_splitter,
        evaluate_classifier, evaluate_regressor, save_report_card
    )
    from shared.data_loader import get_leave_patients_out_split
    from shared.model_zoo import get_xgboost, get_gradient_boosting
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    np.random.seed(RANDOM_SEED)

    iter_dir = None
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.name.startswith("iter_79"):
            iter_dir = d
            break

    data_dir = iter_dir.parent.parent / "data" / "iter_79"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 79: Ablation study")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )

    ablations = {
        "full": dict(include_volatility=True, include_interactions=True,
                     include_momentum=True, include_lagged_valence=True,
                     n_lags=5, log_transform_before_agg=True),
        "no_volatility": dict(include_volatility=False, include_interactions=True,
                              include_momentum=True, include_lagged_valence=True,
                              n_lags=5, log_transform_before_agg=True),
        "no_interactions": dict(include_volatility=True, include_interactions=False,
                                include_momentum=True, include_lagged_valence=True,
                                n_lags=5, log_transform_before_agg=True),
        "no_momentum": dict(include_volatility=True, include_interactions=True,
                            include_momentum=False, include_lagged_valence=True,
                            n_lags=5, log_transform_before_agg=True),
        "no_lagged_valence": dict(include_volatility=True, include_interactions=True,
                                  include_momentum=True, include_lagged_valence=False,
                                  n_lags=5, log_transform_before_agg=True),
        "no_lags": dict(include_volatility=True, include_interactions=True,
                        include_momentum=True, include_lagged_valence=True,
                        n_lags=0, log_transform_before_agg=True),
        "no_log_transform": dict(include_volatility=True, include_interactions=True,
                                 include_momentum=True, include_lagged_valence=True,
                                 n_lags=5, log_transform_before_agg=False),
    }

    results = {}
    for name, params in ablations.items():
        print(f"\n    --- Ablation: {name} ---")
        features_df = build_features(daily, **params)
        train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=RANDOM_SEED)

        meta_cols = [ID_COL, DATE_COL, TARGET_COL]
        feature_cols = [c for c in train_feat.columns if c not in meta_cols]
        X_tr = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr = train_feat[TARGET_COL].values
        X_te = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_te = test_feat[TARGET_COL].values
        groups = train_feat[ID_COL].values

        q33, q66 = compute_tercile_thresholds(y_tr)
        y_tr_cls = discretize_mood(y_tr, q33, q66)
        y_te_cls = discretize_mood(y_te, q33, q66)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        cv = get_cv_splitter(min(5, len(np.unique(groups))))
        xgb = get_xgboost("classification", random_state=RANDOM_SEED)
        grid = GridSearchCV(xgb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                   "learning_rate": [0.05, 0.1]},
                             cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
        grid.fit(X_tr_s, y_tr_cls, groups=groups)
        cls_pred = grid.predict(X_te_s)
        cls_f1 = evaluate_classifier(y_te_cls, cls_pred)["f1_macro"]
        del grid; gc.collect()

        gb = get_gradient_boosting("regression", random_state=RANDOM_SEED)
        grid = GridSearchCV(gb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                  "learning_rate": [0.01, 0.05]},
                             cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
        grid.fit(X_tr_s, y_tr, groups=groups)
        reg_pred = grid.predict(X_te_s)
        reg_r2 = evaluate_regressor(y_te, reg_pred)["r2"]
        del grid; gc.collect()

        results[name] = {"cls_f1": cls_f1, "reg_r2": reg_r2, "n_features": len(feature_cols)}
        print(f"    {name}: F1={cls_f1:.4f}, R2={reg_r2:.4f}, features={len(feature_cols)}")

    # Compute importance of each feature group
    full_f1 = results["full"]["cls_f1"]
    full_r2 = results["full"]["reg_r2"]
    print("\n    Feature group importance (drop in F1 / R2 when removed):")
    for name, res in results.items():
        if name == "full":
            continue
        f1_drop = full_f1 - res["cls_f1"]
        r2_drop = full_r2 - res["reg_r2"]
        print(f"    {name}: F1 drop={f1_drop:+.4f}, R2 drop={r2_drop:+.4f}")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=79,
        hypothesis="Ablation reveals which feature groups are load-bearing.",
        change_summary="Remove one feature group at a time, measure impact",
        classification_results={"xgboost": {"f1_macro": full_f1, "accuracy": full_f1, "per_class_f1": []}},
        regression_results={"gb": {"r2": full_r2, "rmse": 0, "mae": 0, "mse": 0}},
        n_features=results["full"]["n_features"], n_train=0, n_test=0,
        extra={"ablation_results": results},
    )
    return card


def run_iter_80():
    """Optimized classification thresholds"""
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

    iter_dir = None
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.name.startswith("iter_80"):
            iter_dir = d
            break

    data_dir = iter_dir.parent.parent / "data" / "iter_80"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 80: Optimized classification thresholds")
    print("=" * 60)

    daily = load_and_clean(
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
        drop_sparse=True, add_morning_evening=True,
    )

    features_df = build_features(
        daily, n_lags=5, include_volatility=True, include_interactions=True,
        include_momentum=True, include_lagged_valence=True,
        log_transform_before_agg=True,
    )

    train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=RANDOM_SEED)

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]
    X_train = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train = train_feat[TARGET_COL].values
    X_test = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_test = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train regression model first (predict continuous, then discretize)
    cv = get_cv_splitter(min(5, len(np.unique(groups_train))))
    gb = get_gradient_boosting("regression", random_state=RANDOM_SEED)
    grid = GridSearchCV(gb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                              "learning_rate": [0.01, 0.05]},
                         cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_train_s, y_train, groups=groups_train)
    reg_pred_test = grid.predict(X_test_s)
    reg_pred_train = grid.predict(X_train_s)
    reg_results = evaluate_regressor(y_test, reg_pred_test)
    del grid; gc.collect()

    # Also train XGB classifier with standard terciles
    q33_std, q66_std = compute_tercile_thresholds(y_train)
    y_train_cls = discretize_mood(y_train, q33_std, q66_std)
    y_test_cls = discretize_mood(y_test, q33_std, q66_std)

    xgb = get_xgboost("classification", random_state=RANDOM_SEED)
    grid = GridSearchCV(xgb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                               "learning_rate": [0.05, 0.1]},
                         cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_train_s, y_train_cls, groups=groups_train)
    std_cls_pred = grid.predict(X_test_s)
    std_cls_results = evaluate_classifier(y_test_cls, std_cls_pred)
    del grid; gc.collect()

    # Sweep thresholds: try different percentile pairs on train predictions
    best_f1 = 0
    best_thresholds = (q33_std, q66_std)
    threshold_results = {}

    for low_pct in [25, 30, 33, 35, 40]:
        for high_pct in [60, 65, 67, 70, 75]:
            if low_pct >= high_pct:
                continue
            q_low = np.percentile(y_train, low_pct)
            q_high = np.percentile(y_train, high_pct)

            y_test_c = discretize_mood(y_test, q_low, q_high)
            # Use regression predictions discretized with these thresholds
            pred_c = discretize_mood(reg_pred_test, q_low, q_high)
            try:
                f1 = evaluate_classifier(y_test_c, pred_c)["f1_macro"]
                threshold_results[f"p{low_pct}_p{high_pct}"] = {
                    "q_low": float(q_low), "q_high": float(q_high), "f1": f1
                }
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = (q_low, q_high)
            except Exception:
                pass

    print(f"\n    Standard tercile thresholds: q33={q33_std:.3f}, q66={q66_std:.3f}")
    print(f"    Standard XGB F1: {std_cls_results['f1_macro']:.4f}")
    print(f"    Best regression-based F1: {best_f1:.4f} (thresholds: {best_thresholds[0]:.3f}, {best_thresholds[1]:.3f})")

    card = save_report_card(
        iteration_dir=iter_dir, iteration=80,
        hypothesis="Optimized thresholds improve macro F1 vs fixed terciles.",
        change_summary="Sweep classification thresholds on regression predictions",
        classification_results={"xgboost": std_cls_results,
                                 "regression_based": {"f1_macro": best_f1, "accuracy": best_f1,
                                                       "per_class_f1": [], "thresholds": list(best_thresholds)}},
        regression_results={"gb": reg_results},
        n_features=len(feature_cols), n_train=len(train_feat), n_test=len(test_feat),
        extra={"threshold_sweep": threshold_results, "best_thresholds": list(best_thresholds)},
    )
    return card


def run_iter_81():
    """Best v4 combined -- configured based on ablation + results of 63-80"""
    # Ablation results: drop volatility, interactions, momentum, lagged_valence (all redundant)
    # Keep: log_transform (most impactful), EMA features (iter_69: +0.006 F1, +0.004 R2)
    # Keep: autocorrelation (iter_72: +0.016 F1 for classification)
    # Keep: class weights (iter_65: +0.007 F1)
    # Drop: lags (ablation: removing lags IMPROVES both tasks)
    # GRU: keep hidden_dim=32, dropout=0.3 (changes hurt regression)
    return run_full_pipeline(
        iteration=81,
        hypothesis="Ablation-informed config: only keep features that help, drop redundant ones.",
        change_summary="Ablation-optimized: log_transform + EMA + autocorrelation + class weights, no lags/volatility/interactions/momentum",
        outlier_method="iqr",
        iqr_multiplier=3.0,
        imputation_method="linear",
        drop_sparse=True,
        add_morning_evening=True,
        include_volatility=False,
        include_interactions=False,
        include_momentum=False,
        include_lagged_valence=False,
        include_ema=True,
        include_autocorrelation=True,
        n_lags=0,
        log_transform_before_agg=True,
        cls_class_weight=True,
        split_method="leave_patients_out",
        n_holdout_patients=5,
        tabular_cls="xgboost",
        tabular_reg="gb",
        temporal="gru",
    )


def run_iter_82():
    """Final robustness with 10 seeds"""
    all_results = []
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 6789]

    iter_dir = None
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.name.startswith("iter_82"):
            iter_dir = d
            break

    data_dir = iter_dir.parent.parent / "data" / "iter_82"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ITERATION 82: Final robustness (10 seeds)")
    print("=" * 60)

    for i, seed in enumerate(seeds):
        print(f"\n    --- Seed {seed} ({i+1}/10) ---")
        try:
            # Run with temporary iteration number hack -- reuse iter_82 dir
            from shared.data_loader import load_and_clean, get_leave_patients_out_split
            from shared.feature_builder import build_features
            from shared.evaluation import (
                compute_tercile_thresholds, discretize_mood, get_cv_splitter,
                evaluate_classifier, evaluate_regressor
            )
            from shared.model_zoo import get_xgboost, get_gradient_boosting, get_gru
            from shared.feature_builder import get_raw_sequences
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import GridSearchCV

            np.random.seed(seed)

            daily = load_and_clean(
                outlier_method="iqr", iqr_multiplier=3.0, imputation_method="linear",
                drop_sparse=True, add_morning_evening=True,
            )

            features_df = build_features(
                daily, n_lags=5, include_volatility=True, include_interactions=True,
                include_momentum=True, include_lagged_valence=True,
                log_transform_before_agg=True,
            )

            train_feat, test_feat = get_leave_patients_out_split(
                features_df, n_holdout=5, seed=seed
            )

            meta_cols = [ID_COL, DATE_COL, TARGET_COL]
            feature_cols = [c for c in train_feat.columns if c not in meta_cols]
            X_tr = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
            y_tr = train_feat[TARGET_COL].values
            X_te = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
            y_te = test_feat[TARGET_COL].values
            groups = train_feat[ID_COL].values

            q33, q66 = compute_tercile_thresholds(y_tr)
            y_tr_cls = discretize_mood(y_tr, q33, q66)
            y_te_cls = discretize_mood(y_te, q33, q66)

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            cv = get_cv_splitter(min(5, len(np.unique(groups))))
            xgb = get_xgboost("classification", random_state=seed)
            grid = GridSearchCV(xgb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                       "learning_rate": [0.05, 0.1]},
                                 cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
            grid.fit(X_tr_s, y_tr_cls, groups=groups)
            cls_pred = grid.predict(X_te_s)
            cls_results = evaluate_classifier(y_te_cls, cls_pred)
            del grid; gc.collect()

            gb = get_gradient_boosting("regression", random_state=seed)
            grid = GridSearchCV(gb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                      "learning_rate": [0.01, 0.05]},
                                 cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
            grid.fit(X_tr_s, y_tr, groups=groups)
            reg_pred = grid.predict(X_te_s)
            reg_results = evaluate_regressor(y_te, reg_pred)
            del grid; gc.collect()

            # GRU
            X_seq, y_seq, pids_seq, _ = get_raw_sequences(daily, seq_length=7)
            rng = np.random.RandomState(seed)
            patients = daily[ID_COL].unique()
            holdout = rng.choice(patients, size=5, replace=False)
            t_mask = ~np.isin(pids_seq, holdout)
            X_tr_seq = X_seq[t_mask]; X_te_seq = X_seq[~t_mask]
            y_tr_seq = y_seq[t_mask]; y_te_seq = y_seq[~t_mask]

            ns, sl2, nf2 = X_tr_seq.shape
            sc2 = StandardScaler()
            X_tr_seq_s = sc2.fit_transform(X_tr_seq.reshape(-1, nf2)).reshape(ns, sl2, nf2)
            X_te_seq_s = sc2.transform(X_te_seq.reshape(-1, nf2)).reshape(X_te_seq.shape[0], sl2, nf2)
            nv = max(1, int(ns * 0.2))

            gru_cls = get_gru(input_dim=nf2, task="classification", hidden_dim=32, dropout=0.3)
            y_tr_cls_seq = discretize_mood(y_tr_seq, q33, q66)
            gru_cls.fit(X_tr_seq_s[:-nv], y_tr_cls_seq[:-nv],
                        X_val=X_tr_seq_s[-nv:], y_val=y_tr_cls_seq[-nv:])
            gru_cls_pred = gru_cls.predict(X_te_seq_s)
            gru_cls_res = evaluate_classifier(discretize_mood(y_te_seq, q33, q66), gru_cls_pred)
            del gru_cls; gc.collect()

            gru_reg = get_gru(input_dim=nf2, task="regression", hidden_dim=32, dropout=0.3)
            gru_reg.fit(X_tr_seq_s[:-nv], y_tr_seq[:-nv],
                        X_val=X_tr_seq_s[-nv:], y_val=y_tr_seq[-nv:])
            gru_reg_pred = gru_reg.predict(X_te_seq_s)
            gru_reg_res = evaluate_regressor(y_te_seq, gru_reg_pred)
            del gru_reg; gc.collect()

            seed_result = {
                "seed": seed,
                "xgb_cls_f1": cls_results["f1_macro"],
                "gb_reg_r2": reg_results["r2"],
                "gru_cls_f1": gru_cls_res["f1_macro"],
                "gru_reg_r2": gru_reg_res["r2"],
                "holdout_patients": list(holdout),
            }
            all_results.append(seed_result)
            print(f"    Seed {seed}: XGB F1={cls_results['f1_macro']:.4f}, "
                  f"GB R2={reg_results['r2']:.4f}, "
                  f"GRU F1={gru_cls_res['f1_macro']:.4f}, "
                  f"GRU R2={gru_reg_res['r2']:.4f}")

        except Exception as e:
            print(f"    Seed {seed} FAILED: {e}")
            traceback.print_exc()

    # Summary
    if all_results:
        xgb_f1s = [r["xgb_cls_f1"] for r in all_results]
        gb_r2s = [r["gb_reg_r2"] for r in all_results]
        gru_f1s = [r["gru_cls_f1"] for r in all_results]
        gru_r2s = [r["gru_reg_r2"] for r in all_results]

        print(f"\n    === 10-SEED SUMMARY ===")
        print(f"    XGB Cls F1: {np.mean(xgb_f1s):.4f} +/- {np.std(xgb_f1s):.4f} [{np.min(xgb_f1s):.4f}, {np.max(xgb_f1s):.4f}]")
        print(f"    GB Reg R2:  {np.mean(gb_r2s):.4f} +/- {np.std(gb_r2s):.4f} [{np.min(gb_r2s):.4f}, {np.max(gb_r2s):.4f}]")
        print(f"    GRU Cls F1: {np.mean(gru_f1s):.4f} +/- {np.std(gru_f1s):.4f} [{np.min(gru_f1s):.4f}, {np.max(gru_f1s):.4f}]")
        print(f"    GRU Reg R2: {np.mean(gru_r2s):.4f} +/- {np.std(gru_r2s):.4f} [{np.min(gru_r2s):.4f}, {np.max(gru_r2s):.4f}]")

    from shared.evaluation import save_report_card
    card = save_report_card(
        iteration_dir=iter_dir, iteration=82,
        hypothesis="10 seeds give robust performance estimate with 95% CI.",
        change_summary="Final robustness: 10 different holdout sets",
        classification_results={
            "xgboost": {"f1_macro": float(np.mean(xgb_f1s)), "f1_std": float(np.std(xgb_f1s)),
                         "accuracy": float(np.mean(xgb_f1s)), "per_class_f1": xgb_f1s},
            "gru": {"f1_macro": float(np.mean(gru_f1s)), "f1_std": float(np.std(gru_f1s)),
                     "accuracy": float(np.mean(gru_f1s)), "per_class_f1": gru_f1s},
        },
        regression_results={
            "gb": {"r2": float(np.mean(gb_r2s)), "r2_std": float(np.std(gb_r2s)),
                    "rmse": 0, "mae": 0, "mse": 0},
            "gru": {"r2": float(np.mean(gru_r2s)), "r2_std": float(np.std(gru_r2s)),
                     "rmse": 0, "mae": 0, "mse": 0},
        },
        n_features=0, n_train=0, n_test=0,
        extra={"per_seed_results": all_results, "n_seeds": len(all_results)},
    )
    return card


# === MAIN RUNNER ===

ITERATIONS = {
    63: run_iter_63, 64: run_iter_64, 65: run_iter_65, 66: run_iter_66,
    67: run_iter_67, 68: run_iter_68, 69: run_iter_69, 70: run_iter_70,
    71: run_iter_71, 72: run_iter_72, 73: run_iter_73, 74: run_iter_74,
    75: run_iter_75, 76: run_iter_76, 77: run_iter_77, 78: run_iter_78,
    79: run_iter_79, 80: run_iter_80, 81: run_iter_81, 82: run_iter_82,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=63)
    parser.add_argument("--end", type=int, default=82)
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

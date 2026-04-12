"""Iteration 21: Robustness check with 5 seeds."""
import sys, gc, warnings
import numpy as np
import json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ID_COL, DATE_COL, TARGET_COL, N_CV_FOLDS, N_JOBS
from shared.data_loader import load_and_clean, get_leave_patients_out_split
from shared.feature_builder import build_features, get_raw_sequences
from shared.evaluation import (
    compute_tercile_thresholds, discretize_mood, get_cv_splitter,
    evaluate_classifier, evaluate_regressor, save_report_card
)
from shared.model_zoo import get_xgboost, get_gradient_boosting, get_gru
from shared.memory_guard import check_memory
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 123, 456, 789, 1024]

if __name__ == "__main__":
    print("=" * 60)
    print("ITERATION 21: ROBUSTNESS CHECK (5 SEEDS)")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill")
    features_df = build_features(daily, window_sizes=[7], n_lags=3,
                                  include_volatility=True, include_interactions=True)

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    all_cls_f1, all_reg_r2, all_gru_cls_f1, all_gru_reg_r2 = [], [], [], []

    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n  --- Seed {seed} ---")

        train_feat, test_feat = get_leave_patients_out_split(features_df, n_holdout=5, seed=seed)

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

        # XGBoost Classification
        xgb = get_xgboost("classification", random_state=seed)
        cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups))))
        grid = GridSearchCV(xgb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                   "learning_rate": [0.05, 0.1]},
                             cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
        grid.fit(X_tr_s, y_tr_cls, groups=groups)
        cls_f1 = evaluate_classifier(y_te_cls, grid.best_estimator_.predict(X_te_s))["f1_macro"]
        all_cls_f1.append(cls_f1)
        del grid; gc.collect()

        # GB Regression
        gb = get_gradient_boosting("regression", random_state=seed)
        grid = GridSearchCV(gb, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                  "learning_rate": [0.01, 0.05]},
                             cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
        grid.fit(X_tr_s, y_tr, groups=groups)
        reg_r2 = evaluate_regressor(y_te, grid.best_estimator_.predict(X_te_s))["r2"]
        all_reg_r2.append(reg_r2)
        del grid; gc.collect()

        # GRU (temporal)
        X_seq, y_seq, pids, dates = get_raw_sequences(daily, seq_length=7)
        holdout_patients = test_feat[ID_COL].unique()
        t_mask = ~np.isin(pids, holdout_patients)
        X_st, X_se = X_seq[t_mask], X_seq[~t_mask]
        y_st, y_se = y_seq[t_mask], y_seq[~t_mask]
        ns, sl, nf = X_st.shape
        sc2 = StandardScaler()
        X_st_s = sc2.fit_transform(X_st.reshape(-1, nf)).reshape(ns, sl, nf)
        X_se_s = sc2.transform(X_se.reshape(-1, nf)).reshape(X_se.shape[0], sl, nf)
        del X_seq, X_st, X_se; gc.collect()

        nv = max(1, int(len(X_st_s) * 0.2))
        gru_cls = get_gru(input_dim=nf, task="classification", hidden_dim=32, dropout=0.3,
                          lr=0.001, epochs=100, patience=15, batch_size=32)
        gru_cls.fit(X_st_s[:-nv], discretize_mood(y_st[:-nv], q33, q66),
                    X_val=X_st_s[-nv:], y_val=discretize_mood(y_st[-nv:], q33, q66))
        gru_cls_f1 = evaluate_classifier(discretize_mood(y_se, q33, q66),
                                          gru_cls.predict(X_se_s))["f1_macro"]
        all_gru_cls_f1.append(gru_cls_f1)
        del gru_cls; gc.collect()

        gru_reg = get_gru(input_dim=nf, task="regression", hidden_dim=32, dropout=0.3,
                          lr=0.001, epochs=100, patience=15, batch_size=32)
        gru_reg.fit(X_st_s[:-nv], y_st[:-nv], X_val=X_st_s[-nv:], y_val=y_st[-nv:])
        gru_reg_r2 = evaluate_regressor(y_se, gru_reg.predict(X_se_s))["r2"]
        all_gru_reg_r2.append(gru_reg_r2)
        del gru_reg, X_st_s, X_se_s; gc.collect()

        print(f"    XGB F1={cls_f1:.4f}, GB R2={reg_r2:.4f}, GRU F1={gru_cls_f1:.4f}, GRU R2={gru_reg_r2:.4f}")
        check_memory(f"seed {seed}")

    print(f"\n  === FINAL RESULTS (5 seeds) ===")
    print(f"  XGB Cls F1:  {np.mean(all_cls_f1):.4f} +/- {np.std(all_cls_f1):.4f}")
    print(f"  GB Reg R2:   {np.mean(all_reg_r2):.4f} +/- {np.std(all_reg_r2):.4f}")
    print(f"  GRU Cls F1:  {np.mean(all_gru_cls_f1):.4f} +/- {np.std(all_gru_cls_f1):.4f}")
    print(f"  GRU Reg R2:  {np.mean(all_gru_reg_r2):.4f} +/- {np.std(all_gru_reg_r2):.4f}")

    # Baseline
    majority_acc = 1/3  # random
    print(f"  Baseline (random): ~{majority_acc:.3f}")

    iter_dir = Path(__file__).parent
    save_report_card(
        iteration_dir=iter_dir, iteration=21,
        hypothesis="Robustness check with 5 random seeds.",
        change_summary="5 seeds [42,123,456,789,1024], leave-patients-out, best config.",
        classification_results={
            "xgboost": {"f1_macro": float(np.mean(all_cls_f1)),
                        "f1_std": float(np.std(all_cls_f1)),
                        "f1_per_seed": [float(f) for f in all_cls_f1]},
            "gru": {"f1_macro": float(np.mean(all_gru_cls_f1)),
                    "f1_std": float(np.std(all_gru_cls_f1)),
                    "f1_per_seed": [float(f) for f in all_gru_cls_f1]},
        },
        regression_results={
            "gb": {"r2": float(np.mean(all_reg_r2)),
                   "r2_std": float(np.std(all_reg_r2)),
                   "r2_per_seed": [float(r) for r in all_reg_r2]},
            "gru": {"r2": float(np.mean(all_gru_reg_r2)),
                    "r2_std": float(np.std(all_gru_reg_r2)),
                    "r2_per_seed": [float(r) for r in all_gru_reg_r2]},
        },
        n_features=len(feature_cols), n_train=0, n_test=0,
    )
    print(f"\n  Report saved: {iter_dir / 'report_card.json'}")

"""
Iteration 3 -- Multi-Scale Windows Classification.
Features at 3-day, 7-day, and 14-day scales + volatility + interactions.
"""
import sys, gc, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_CV_FOLDS, N_JOBS
from shared.data_loader import load_and_clean, get_temporal_split
from shared.feature_builder import build_features, get_raw_sequences
from shared.evaluation import (
    compute_tercile_thresholds, discretize_mood, get_cv_splitter, evaluate_classifier
)
from shared.model_zoo import get_random_forest, get_lstm
from shared.memory_guard import check_memory
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)


def run():
    print("=" * 60)
    print("ITERATION 3 -- MULTI-SCALE CLASSIFICATION")
    print("=" * 60)

    daily = load_and_clean()

    features_df = build_features(
        daily, window_sizes=[3, 7, 14], n_lags=3,
        include_volatility=True, include_interactions=True
    )
    print(f"  Features: {features_df.shape}")

    train_feat, test_feat = get_temporal_split(features_df)
    print(f"  Train: {len(train_feat)}, Test: {len(test_feat)}")

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]
    print(f"  Feature count: {len(feature_cols)}")

    X_train = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train_cont = train_feat[TARGET_COL].values
    X_test = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_test_cont = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    q33, q66 = compute_tercile_thresholds(y_train_cont)
    y_train = discretize_mood(y_train_cont, q33, q66)
    y_test = discretize_mood(y_test_cont, q33, q66)
    print(f"  Class dist (test): {np.bincount(y_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n  Training Random Forest...")
    rf = get_random_forest("classification")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_leaf": [2, 5],
    }
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    rf_best = grid.best_estimator_
    rf_pred = rf_best.predict(X_test_scaled)
    rf_results = evaluate_classifier(y_test, rf_pred)
    rf_cv = float(grid.best_score_)
    check_memory("after RF")
    print(f"  CV F1: {rf_cv:.4f}, Test F1: {rf_results['f1_macro']:.4f}")
    print(f"  Per-class F1: {rf_results['per_class_f1']}")

    importances = rf_best.feature_importances_
    top_idx = np.argsort(importances)[-5:]
    print("  Top 5:")
    for idx in reversed(top_idx):
        print(f"    {feature_cols[idx]}: {importances[idx]:.4f}")

    del grid; gc.collect()

    # LSTM (unchanged)
    print("\n  Training LSTM (unchanged)...")
    X_seq, y_seq, _, dates_seq = get_raw_sequences(daily, seq_length=7)
    all_dates = sorted(daily[DATE_COL].unique())
    cutoff_date = all_dates[int(len(all_dates) * 0.8)]
    seq_dates_ts = np.array([np.datetime64(d) for d in dates_seq])
    train_mask = seq_dates_ts < np.datetime64(cutoff_date)
    test_mask = ~train_mask

    X_st, X_se = X_seq[train_mask], X_seq[test_mask]
    n_s, sl, nf = X_st.shape
    sc2 = StandardScaler()
    X_st_s = sc2.fit_transform(X_st.reshape(-1, nf)).reshape(n_s, sl, nf)
    X_se_s = sc2.transform(X_se.reshape(-1, nf)).reshape(X_se.shape[0], sl, nf)
    del X_seq, X_st, X_se; gc.collect()

    y_st_cls = discretize_mood(y_seq[train_mask], q33, q66)
    y_se_cls = discretize_mood(y_seq[test_mask], q33, q66)
    nv = max(1, int(len(X_st_s) * 0.2))

    lstm = get_lstm(input_dim=nf, task="classification", hidden_dim=32, dropout=0.3,
                    lr=0.001, epochs=100, patience=15, batch_size=32)
    lstm.fit(X_st_s[:-nv], y_st_cls[:-nv], X_val=X_st_s[-nv:], y_val=y_st_cls[-nv:])
    lstm_pred = lstm.predict(X_se_s)
    lstm_results = evaluate_classifier(y_se_cls, lstm_pred)
    print(f"  LSTM F1: {lstm_results['f1_macro']:.4f}")

    del lstm, X_st_s, X_se_s; gc.collect()

    return {
        "rf": rf_results, "lstm": lstm_results,
        "rf_train_f1": rf_cv,
        "feature_cols": feature_cols, "importances": importances.tolist(),
        "q33": q33, "q66": q66,
        "n_features": len(feature_cols),
        "n_train": len(train_feat), "n_test": len(test_feat),
    }


if __name__ == "__main__":
    run()

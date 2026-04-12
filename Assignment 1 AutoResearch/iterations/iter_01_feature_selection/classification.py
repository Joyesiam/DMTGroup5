"""
Iteration 1 -- Feature Selection Classification Pipeline.
Same as iter_00 but with top-30 features by mutual information.
"""
import sys
import gc
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_CV_FOLDS, N_JOBS
from shared.data_loader import load_and_clean, get_temporal_split
from shared.feature_builder import build_features, get_raw_sequences, select_features
from shared.evaluation import (
    compute_tercile_thresholds, discretize_mood, get_cv_splitter,
    evaluate_classifier
)
from shared.model_zoo import get_random_forest, get_lstm
from shared.memory_guard import check_memory
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)


def run():
    print("=" * 60)
    print("ITERATION 1 -- FEATURE SELECTION CLASSIFICATION")
    print("=" * 60)

    print("\n[1/7] Loading and cleaning data...")
    daily = load_and_clean()

    print("\n[2/7] Building features (7-day window)...")
    features_df = build_features(daily, window_sizes=[7], n_lags=3)
    print(f"  Full feature dataset: {features_df.shape}")

    print("\n[3/7] Train/test split...")
    train_feat, test_feat = get_temporal_split(features_df)
    print(f"  Train: {len(train_feat)}, Test: {len(test_feat)}")

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train_full = train_feat[feature_cols]
    y_train_cont = train_feat[TARGET_COL].values
    X_test_full = test_feat[feature_cols]
    y_test_cont = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    q33, q66 = compute_tercile_thresholds(y_train_cont)
    y_train = discretize_mood(y_train_cont, q33, q66)
    y_test = discretize_mood(y_test_cont, q33, q66)

    # --- Feature Selection (top 30 by mutual info on train) ---
    print("\n[4/7] Selecting top 30 features by mutual information...")
    X_train_sel, selected_names, selector = select_features(
        X_train_full, y_train_cont, method="mutual_info", k=30
    )
    # Apply same selection to test
    mask = selector.get_support()
    X_test_sel = X_test_full.iloc[:, mask]

    print(f"  Selected {len(selected_names)} features:")
    for name in selected_names[:10]:
        print(f"    {name}")
    if len(selected_names) > 10:
        print(f"    ... and {len(selected_names) - 10} more")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel.values)
    X_test_scaled = scaler.transform(X_test_sel.values)

    # --- Random Forest ---
    print("\n[5/7] Training Random Forest (30 features)...")
    rf = get_random_forest("classification")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_leaf": [2, 5],
    }
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring="f1_macro",
        n_jobs=N_JOBS, verbose=0,
    )
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    rf_best = grid.best_estimator_
    rf_pred = rf_best.predict(X_test_scaled)
    rf_results = evaluate_classifier(y_test, rf_pred)
    rf_cv_score = float(grid.best_score_)
    check_memory("after RF")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1: {rf_cv_score:.4f}")
    print(f"  Test Acc: {rf_results['accuracy']:.4f}, F1: {rf_results['f1_macro']:.4f}")
    print(f"  Per-class F1: {rf_results['per_class_f1']}")

    importances = rf_best.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    print("  Top 10 features:")
    for idx in reversed(top_idx):
        print(f"    {selected_names[idx]}: {importances[idx]:.4f}")

    del grid
    gc.collect()

    # --- LSTM (unchanged from iter_00) ---
    print("\n[6/7] Training LSTM on raw daily sequences (unchanged)...")
    X_seq, y_seq, pids_seq, dates_seq = get_raw_sequences(daily, seq_length=7)

    all_dates = sorted(daily[DATE_COL].unique())
    cutoff_idx = int(len(all_dates) * 0.8)
    cutoff_date = all_dates[cutoff_idx]

    seq_dates_ts = np.array([np.datetime64(d) for d in dates_seq])
    cutoff_ts = np.datetime64(cutoff_date)
    train_mask = seq_dates_ts < cutoff_ts
    test_mask = seq_dates_ts >= cutoff_ts

    X_seq_train, X_seq_test = X_seq[train_mask], X_seq[test_mask]
    y_seq_train_cont, y_seq_test_cont = y_seq[train_mask], y_seq[test_mask]

    n_samples, seq_len, n_feats = X_seq_train.shape
    seq_scaler = StandardScaler()
    X_seq_train_scaled = seq_scaler.fit_transform(
        X_seq_train.reshape(-1, n_feats)
    ).reshape(n_samples, seq_len, n_feats)
    X_seq_test_scaled = seq_scaler.transform(
        X_seq_test.reshape(-1, n_feats)
    ).reshape(X_seq_test.shape[0], seq_len, n_feats)

    del X_seq, X_seq_train, X_seq_test
    gc.collect()

    y_seq_train_cls = discretize_mood(y_seq_train_cont, q33, q66)
    y_seq_test_cls = discretize_mood(y_seq_test_cont, q33, q66)

    n_val = max(1, int(len(X_seq_train_scaled) * 0.2))
    lstm_cls = get_lstm(
        input_dim=n_feats, task="classification",
        hidden_dim=32, n_layers=1, dropout=0.3,
        lr=0.001, epochs=100, patience=15, batch_size=32,
    )
    lstm_cls.fit(
        X_seq_train_scaled[:-n_val], y_seq_train_cls[:-n_val],
        X_val=X_seq_train_scaled[-n_val:], y_val=y_seq_train_cls[-n_val:]
    )
    lstm_pred = lstm_cls.predict(X_seq_test_scaled)
    lstm_results = evaluate_classifier(y_seq_test_cls, lstm_pred)
    check_memory("after LSTM")
    print(f"  LSTM epochs: {len(lstm_cls.train_losses_)}")
    print(f"  Test Acc: {lstm_results['accuracy']:.4f}, F1: {lstm_results['f1_macro']:.4f}")

    print("\n[7/7] Summary")
    print(f"  RF   -- Acc: {rf_results['accuracy']:.4f}, F1: {rf_results['f1_macro']:.4f}")
    print(f"  LSTM -- Acc: {lstm_results['accuracy']:.4f}, F1: {lstm_results['f1_macro']:.4f}")

    del lstm_cls, X_seq_train_scaled, X_seq_test_scaled
    gc.collect()

    return {
        "rf": rf_results,
        "lstm": lstm_results,
        "rf_train_f1": rf_cv_score,
        "feature_cols": selected_names,
        "importances": importances.tolist(),
        "q33": q33, "q66": q66,
        "n_features": len(selected_names),
        "n_train": len(train_feat),
        "n_test": len(test_feat),
    }


if __name__ == "__main__":
    run()

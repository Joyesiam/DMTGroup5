"""
Iteration 0 -- Baseline Classification Pipeline.
Random Forest (tabular) + LSTM (temporal) for 3-class mood prediction.
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
from shared.feature_builder import build_features, get_raw_sequences
from shared.evaluation import (
    compute_tercile_thresholds, discretize_mood, get_cv_splitter,
    evaluate_classifier
)
from shared.model_zoo import get_random_forest, get_lstm
from shared.memory_guard import check_memory, cleanup
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)


def run():
    print("=" * 60)
    print("ITERATION 0 -- BASELINE CLASSIFICATION")
    print("=" * 60)

    # --- Data Pipeline ---
    print("\n[1/6] Loading and cleaning data...")
    daily = load_and_clean()
    check_memory("after data load")
    print(f"  Daily data: {daily.shape[0]} rows, {daily.shape[1]} columns")

    print("\n[2/6] Building features (7-day window)...")
    features_df = build_features(daily, window_sizes=[7], n_lags=3)
    check_memory("after feature build")
    print(f"  Feature dataset: {features_df.shape[0]} instances, {features_df.shape[1]} columns")

    # --- Train/Test Split ---
    print("\n[3/6] Splitting train/test (chronological, 80/20)...")
    train_feat, test_feat = get_temporal_split(features_df)
    print(f"  Train: {len(train_feat)}, Test: {len(test_feat)}")

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train = train_feat[feature_cols].values
    y_train_cont = train_feat[TARGET_COL].values
    X_test = test_feat[feature_cols].values
    y_test_cont = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    # Compute tercile thresholds on TRAIN ONLY
    q33, q66 = compute_tercile_thresholds(y_train_cont)
    print(f"  Tercile thresholds (train): q33={q33:.3f}, q66={q66:.3f}")

    y_train = discretize_mood(y_train_cont, q33, q66)
    y_test = discretize_mood(y_test_cont, q33, q66)
    print(f"  Class distribution (train): {np.bincount(y_train)}")
    print(f"  Class distribution (test):  {np.bincount(y_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Random Forest (small grid, n_jobs=1) ---
    print("\n[4/6] Training Random Forest with GroupKFold CV...")
    rf = get_random_forest("classification")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_leaf": [2, 5],
    }
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring="f1_macro",
        n_jobs=N_JOBS, verbose=0, error_score="raise",
    )
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    rf_best = grid.best_estimator_
    rf_pred = rf_best.predict(X_test_scaled)
    rf_results = evaluate_classifier(y_test, rf_pred)
    check_memory("after RF GridSearchCV")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1 (macro): {grid.best_score_:.4f}")
    print(f"  Test accuracy: {rf_results['accuracy']:.4f}")
    print(f"  Test F1 (macro): {rf_results['f1_macro']:.4f}")
    print(f"  Per-class F1: {rf_results['per_class_f1']}")

    # Feature importance
    importances = rf_best.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    print("  Top 10 features:")
    for idx in reversed(top_idx):
        print(f"    {feature_cols[idx]}: {importances[idx]:.4f}")

    # Save score before freeing grid
    rf_cv_score = float(grid.best_score_)
    del grid
    gc.collect()

    # --- LSTM (on raw daily sequences) ---
    print("\n[5/6] Training LSTM on raw daily sequences...")
    X_seq, y_seq, pids_seq, dates_seq = get_raw_sequences(daily, seq_length=7)
    print(f"  Sequences: {X_seq.shape[0]} instances, shape {X_seq.shape}")

    all_dates = sorted(daily[DATE_COL].unique())
    cutoff_idx = int(len(all_dates) * 0.8)
    cutoff_date = all_dates[cutoff_idx]

    seq_dates_ts = np.array([np.datetime64(d) for d in dates_seq])
    cutoff_ts = np.datetime64(cutoff_date)
    train_mask = seq_dates_ts < cutoff_ts
    test_mask = seq_dates_ts >= cutoff_ts

    X_seq_train, X_seq_test = X_seq[train_mask], X_seq[test_mask]
    y_seq_train_cont, y_seq_test_cont = y_seq[train_mask], y_seq[test_mask]

    # Normalize sequences
    n_samples, seq_len, n_feats = X_seq_train.shape
    seq_scaler = StandardScaler()
    flat_scaled = seq_scaler.fit_transform(X_seq_train.reshape(-1, n_feats))
    X_seq_train_scaled = flat_scaled.reshape(n_samples, seq_len, n_feats)
    X_seq_test_scaled = seq_scaler.transform(
        X_seq_test.reshape(-1, n_feats)
    ).reshape(X_seq_test.shape[0], seq_len, n_feats)

    # Free unscaled sequences
    del X_seq, X_seq_train, X_seq_test, flat_scaled
    gc.collect()
    check_memory("before LSTM training")

    # Discretize
    y_seq_train_cls = discretize_mood(y_seq_train_cont, q33, q66)
    y_seq_test_cls = discretize_mood(y_seq_test_cont, q33, q66)

    # Validation split for early stopping
    n_val = max(1, int(len(X_seq_train_scaled) * 0.2))
    X_val = X_seq_train_scaled[-n_val:]
    y_val = y_seq_train_cls[-n_val:]
    X_tr = X_seq_train_scaled[:-n_val]
    y_tr = y_seq_train_cls[:-n_val]

    lstm_cls = get_lstm(
        input_dim=n_feats, task="classification",
        hidden_dim=32, n_layers=1, dropout=0.3,
        lr=0.001, epochs=100, patience=15, batch_size=32,
    )
    lstm_cls.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    lstm_pred = lstm_cls.predict(X_seq_test_scaled)
    lstm_results = evaluate_classifier(y_seq_test_cls, lstm_pred)
    check_memory("after LSTM training")
    print(f"  LSTM epochs trained: {len(lstm_cls.train_losses_)}")
    print(f"  Test accuracy: {lstm_results['accuracy']:.4f}")
    print(f"  Test F1 (macro): {lstm_results['f1_macro']:.4f}")
    print(f"  Per-class F1: {lstm_results['per_class_f1']}")

    # --- Summary ---
    print("\n[6/6] Summary")
    print(f"  Random Forest -- Acc: {rf_results['accuracy']:.4f}, F1: {rf_results['f1_macro']:.4f}")
    print(f"  LSTM          -- Acc: {lstm_results['accuracy']:.4f}, F1: {lstm_results['f1_macro']:.4f}")

    # Cleanup before returning
    del lstm_cls, X_seq_train_scaled, X_seq_test_scaled
    gc.collect()

    return {
        "rf": rf_results,
        "lstm": lstm_results,
        "rf_train_f1": rf_cv_score,
        "feature_cols": feature_cols,
        "importances": importances.tolist(),
        "q33": q33,
        "q66": q66,
        "n_features": len(feature_cols),
        "n_train": len(train_feat),
        "n_test": len(test_feat),
    }


if __name__ == "__main__":
    results = run()

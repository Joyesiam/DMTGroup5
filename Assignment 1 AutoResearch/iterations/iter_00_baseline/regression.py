"""
Iteration 0 -- Baseline Regression Pipeline.
Gradient Boosting (tabular) + LSTM (temporal) for continuous mood prediction.
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
from shared.evaluation import evaluate_regressor, get_cv_splitter
from shared.model_zoo import get_gradient_boosting, get_lstm
from shared.memory_guard import check_memory, cleanup
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)


def run():
    print("=" * 60)
    print("ITERATION 0 -- BASELINE REGRESSION")
    print("=" * 60)

    # --- Data Pipeline ---
    print("\n[1/6] Loading and cleaning data...")
    daily = load_and_clean()
    check_memory("after data load")
    print(f"  Daily data: {daily.shape[0]} rows, {daily.shape[1]} columns")

    print("\n[2/6] Building features (7-day window)...")
    features_df = build_features(daily, window_sizes=[7], n_lags=3)
    print(f"  Feature dataset: {features_df.shape[0]} instances, {features_df.shape[1]} columns")

    # --- Train/Test Split ---
    print("\n[3/6] Splitting train/test (chronological, 80/20)...")
    train_feat, test_feat = get_temporal_split(features_df)
    print(f"  Train: {len(train_feat)}, Test: {len(test_feat)}")

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train = train_feat[feature_cols].values
    y_train = train_feat[TARGET_COL].values
    X_test = test_feat[feature_cols].values
    y_test = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Gradient Boosting (small grid, n_jobs=1) ---
    print("\n[4/6] Training Gradient Boosting with GroupKFold CV...")
    gb = get_gradient_boosting("regression")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05],
    }
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(
        gb, param_grid, cv=cv, scoring="neg_mean_squared_error",
        n_jobs=N_JOBS, verbose=0, error_score="raise",
    )
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    gb_best = grid.best_estimator_
    gb_pred = gb_best.predict(X_test_scaled)
    gb_results = evaluate_regressor(y_test, gb_pred)
    check_memory("after GB GridSearchCV")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV MSE: {-grid.best_score_:.4f}")
    print(f"  Test MSE: {gb_results['mse']:.4f}, RMSE: {gb_results['rmse']:.4f}")
    print(f"  Test MAE: {gb_results['mae']:.4f}, R2: {gb_results['r2']:.4f}")

    # Train set performance (overfitting check)
    gb_train_pred = gb_best.predict(X_train_scaled)
    gb_train_results = evaluate_regressor(y_train, gb_train_pred)
    print(f"  Train R2: {gb_train_results['r2']:.4f} (overfitting check)")

    # Feature importance
    importances = gb_best.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    print("  Top 10 features:")
    for idx in reversed(top_idx):
        print(f"    {feature_cols[idx]}: {importances[idx]:.4f}")

    # Free grid before LSTM
    del grid
    gc.collect()

    # --- LSTM Regression ---
    print("\n[5/6] Training LSTM Regressor on raw daily sequences...")
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
    y_seq_train, y_seq_test = y_seq[train_mask], y_seq[test_mask]

    # Normalize sequences
    n_samples, seq_len, n_feats = X_seq_train.shape
    seq_scaler = StandardScaler()
    X_seq_train_scaled = seq_scaler.fit_transform(
        X_seq_train.reshape(-1, n_feats)
    ).reshape(n_samples, seq_len, n_feats)
    X_seq_test_scaled = seq_scaler.transform(
        X_seq_test.reshape(-1, n_feats)
    ).reshape(X_seq_test.shape[0], seq_len, n_feats)

    # Free raw sequences
    del X_seq, X_seq_train, X_seq_test
    gc.collect()
    check_memory("before LSTM regression")

    # Validation split for early stopping
    n_val = max(1, int(len(X_seq_train_scaled) * 0.2))
    X_val = X_seq_train_scaled[-n_val:]
    y_val = y_seq_train[-n_val:]
    X_tr = X_seq_train_scaled[:-n_val]
    y_tr = y_seq_train[:-n_val]

    lstm_reg = get_lstm(
        input_dim=n_feats, task="regression",
        hidden_dim=32, n_layers=1, dropout=0.3,
        lr=0.001, epochs=100, patience=15, batch_size=32,
    )
    lstm_reg.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    lstm_pred = lstm_reg.predict(X_seq_test_scaled)
    lstm_results = evaluate_regressor(y_seq_test, lstm_pred)
    check_memory("after LSTM regression")
    print(f"  LSTM epochs trained: {len(lstm_reg.train_losses_)}")
    print(f"  Test MSE: {lstm_results['mse']:.4f}, RMSE: {lstm_results['rmse']:.4f}")
    print(f"  Test MAE: {lstm_results['mae']:.4f}, R2: {lstm_results['r2']:.4f}")

    # Train set performance
    lstm_train_pred = lstm_reg.predict(X_tr)
    lstm_train_results = evaluate_regressor(y_tr, lstm_train_pred)
    print(f"  Train R2: {lstm_train_results['r2']:.4f} (overfitting check)")

    # --- Summary ---
    print("\n[6/6] Summary")
    print(f"  Gradient Boosting -- RMSE: {gb_results['rmse']:.4f}, R2: {gb_results['r2']:.4f}")
    print(f"  LSTM              -- RMSE: {lstm_results['rmse']:.4f}, R2: {lstm_results['r2']:.4f}")

    # Cleanup
    del lstm_reg, X_seq_train_scaled, X_seq_test_scaled
    gc.collect()

    return {
        "gb": gb_results,
        "lstm": lstm_results,
        "gb_train_r2": gb_train_results["r2"],
        "lstm_train_r2": lstm_train_results["r2"],
        "feature_cols": feature_cols,
        "importances": importances.tolist(),
        "n_features": len(feature_cols),
        "n_train": len(train_feat),
        "n_test": len(test_feat),
    }


if __name__ == "__main__":
    results = run()

"""
Iteration 1 -- Feature Selection Regression Pipeline.
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
from shared.evaluation import evaluate_regressor, get_cv_splitter
from shared.model_zoo import get_gradient_boosting, get_lstm
from shared.memory_guard import check_memory
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)


def run():
    print("=" * 60)
    print("ITERATION 1 -- FEATURE SELECTION REGRESSION")
    print("=" * 60)

    print("\n[1/7] Loading and cleaning data...")
    daily = load_and_clean()

    print("\n[2/7] Building features (7-day window)...")
    features_df = build_features(daily, window_sizes=[7], n_lags=3)

    print("\n[3/7] Train/test split...")
    train_feat, test_feat = get_temporal_split(features_df)
    print(f"  Train: {len(train_feat)}, Test: {len(test_feat)}")

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train_full = train_feat[feature_cols]
    y_train = train_feat[TARGET_COL].values
    X_test_full = test_feat[feature_cols]
    y_test = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    # --- Feature Selection ---
    print("\n[4/7] Selecting top 30 features...")
    X_train_sel, selected_names, selector = select_features(
        X_train_full, y_train, method="mutual_info", k=30
    )
    mask = selector.get_support()
    X_test_sel = X_test_full.iloc[:, mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel.values)
    X_test_scaled = scaler.transform(X_test_sel.values)

    # --- Gradient Boosting ---
    print("\n[5/7] Training GB (30 features)...")
    gb = get_gradient_boosting("regression")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05],
    }
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(
        gb, param_grid, cv=cv, scoring="neg_mean_squared_error",
        n_jobs=N_JOBS, verbose=0,
    )
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    gb_best = grid.best_estimator_
    gb_pred = gb_best.predict(X_test_scaled)
    gb_results = evaluate_regressor(y_test, gb_pred)
    check_memory("after GB")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Test RMSE: {gb_results['rmse']:.4f}, R2: {gb_results['r2']:.4f}")

    gb_train_pred = gb_best.predict(X_train_scaled)
    gb_train_results = evaluate_regressor(y_train, gb_train_pred)
    print(f"  Train R2: {gb_train_results['r2']:.4f}")

    importances = gb_best.feature_importances_
    del grid
    gc.collect()

    # --- LSTM (unchanged) ---
    print("\n[6/7] Training LSTM Regressor (unchanged)...")
    X_seq, y_seq, pids_seq, dates_seq = get_raw_sequences(daily, seq_length=7)

    all_dates = sorted(daily[DATE_COL].unique())
    cutoff_idx = int(len(all_dates) * 0.8)
    cutoff_date = all_dates[cutoff_idx]

    seq_dates_ts = np.array([np.datetime64(d) for d in dates_seq])
    cutoff_ts = np.datetime64(cutoff_date)
    train_mask = seq_dates_ts < cutoff_ts
    test_mask = seq_dates_ts >= cutoff_ts

    X_seq_train, X_seq_test = X_seq[train_mask], X_seq[test_mask]
    y_seq_train, y_seq_test = y_seq[train_mask], y_seq[test_mask]

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

    n_val = max(1, int(len(X_seq_train_scaled) * 0.2))
    lstm_reg = get_lstm(
        input_dim=n_feats, task="regression",
        hidden_dim=32, n_layers=1, dropout=0.3,
        lr=0.001, epochs=100, patience=15, batch_size=32,
    )
    lstm_reg.fit(
        X_seq_train_scaled[:-n_val], y_seq_train[:-n_val],
        X_val=X_seq_train_scaled[-n_val:], y_val=y_seq_train[-n_val:]
    )
    lstm_pred = lstm_reg.predict(X_seq_test_scaled)
    lstm_results = evaluate_regressor(y_seq_test, lstm_pred)
    check_memory("after LSTM")
    print(f"  LSTM RMSE: {lstm_results['rmse']:.4f}, R2: {lstm_results['r2']:.4f}")

    lstm_train_pred = lstm_reg.predict(X_seq_train_scaled[:-n_val])
    lstm_train_results = evaluate_regressor(y_seq_train[:-n_val], lstm_train_pred)
    print(f"  Train R2: {lstm_train_results['r2']:.4f}")

    print("\n[7/7] Summary")
    print(f"  GB   -- RMSE: {gb_results['rmse']:.4f}, R2: {gb_results['r2']:.4f}")
    print(f"  LSTM -- RMSE: {lstm_results['rmse']:.4f}, R2: {lstm_results['r2']:.4f}")

    del lstm_reg, X_seq_train_scaled, X_seq_test_scaled
    gc.collect()

    return {
        "gb": gb_results,
        "lstm": lstm_results,
        "gb_train_r2": gb_train_results["r2"],
        "lstm_train_r2": lstm_train_results["r2"],
        "feature_cols": selected_names,
        "importances": importances.tolist(),
        "n_features": len(selected_names),
        "n_train": len(train_feat),
        "n_test": len(test_feat),
    }


if __name__ == "__main__":
    run()

"""
Iteration 4 -- XGBoost + GRU Regression.
"""
import sys, gc, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_CV_FOLDS, N_JOBS
from shared.data_loader import load_and_clean, get_temporal_split
from shared.feature_builder import build_features, get_raw_sequences
from shared.evaluation import evaluate_regressor, get_cv_splitter
from shared.model_zoo import get_xgboost, get_gru
from shared.memory_guard import check_memory
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)


def run():
    print("=" * 60)
    print("ITERATION 4 -- XGBOOST + GRU REGRESSION")
    print("=" * 60)

    daily = load_and_clean()
    features_df = build_features(
        daily, window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True
    )

    train_feat, test_feat = get_temporal_split(features_df)
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train = train_feat[TARGET_COL].values
    X_test = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_test = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- XGBoost Regressor ---
    print("  Training XGBoost Regressor...")
    xgb = get_xgboost("regression")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05],
        "reg_alpha": [0, 0.1],
    }
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(xgb, param_grid, cv=cv, scoring="neg_mean_squared_error",
                        n_jobs=N_JOBS, verbose=0)
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    xgb_best = grid.best_estimator_
    xgb_pred = xgb_best.predict(X_test_scaled)
    xgb_results = evaluate_regressor(y_test, xgb_pred)
    check_memory("after XGB")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Test RMSE: {xgb_results['rmse']:.4f}, R2: {xgb_results['r2']:.4f}")

    xgb_train_results = evaluate_regressor(y_train, xgb_best.predict(X_train_scaled))
    print(f"  Train R2: {xgb_train_results['r2']:.4f}")

    importances = xgb_best.feature_importances_
    del grid; gc.collect()

    # --- GRU Regressor ---
    print("  Training GRU Regressor...")
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

    y_st, y_se = y_seq[train_mask], y_seq[test_mask]
    nv = max(1, int(len(X_st_s) * 0.2))

    gru = get_gru(input_dim=nf, task="regression", hidden_dim=32, dropout=0.3,
                  lr=0.001, epochs=100, patience=15, batch_size=32)
    gru.fit(X_st_s[:-nv], y_st[:-nv], X_val=X_st_s[-nv:], y_val=y_st[-nv:])
    gru_pred = gru.predict(X_se_s)
    gru_results = evaluate_regressor(y_se, gru_pred)
    print(f"  GRU RMSE: {gru_results['rmse']:.4f}, R2: {gru_results['r2']:.4f}")

    gru_train_results = evaluate_regressor(y_st[:-nv], gru.predict(X_st_s[:-nv]))
    print(f"  Train R2: {gru_train_results['r2']:.4f}")
    del gru, X_st_s, X_se_s; gc.collect()

    return {
        "xgb": xgb_results, "gru": gru_results,
        "xgb_train_r2": xgb_train_results["r2"],
        "gru_train_r2": gru_train_results["r2"],
        "feature_cols": feature_cols, "importances": importances.tolist(),
        "n_features": len(feature_cols),
        "n_train": len(train_feat), "n_test": len(test_feat),
    }


if __name__ == "__main__":
    run()

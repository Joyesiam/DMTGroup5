"""
Iteration 5 -- Stacking Ensemble Regression.
"""
import sys, gc, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_JOBS
from shared.data_loader import load_and_clean, get_temporal_split
from shared.feature_builder import build_features, get_raw_sequences
from shared.evaluation import evaluate_regressor
from shared.model_zoo import get_stacking_regressor, get_gru
from shared.memory_guard import check_memory
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)


def run():
    print("=" * 60)
    print("ITERATION 5 -- ENSEMBLE REGRESSION")
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Stacking Regressor ---
    print("  Training Stacking Regressor (XGB + RF -> Ridge)...")
    ensemble = get_stacking_regressor()
    ensemble.fit(X_train_scaled, y_train)
    ens_pred = ensemble.predict(X_test_scaled)
    ens_results = evaluate_regressor(y_test, ens_pred)
    check_memory("after ensemble")
    print(f"  Test RMSE: {ens_results['rmse']:.4f}, R2: {ens_results['r2']:.4f}")

    ens_train_results = evaluate_regressor(y_train, ensemble.predict(X_train_scaled))
    print(f"  Train R2: {ens_train_results['r2']:.4f}")
    del ensemble; gc.collect()

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
    del gru, X_st_s, X_se_s; gc.collect()

    return {
        "ensemble": ens_results, "gru": gru_results,
        "ensemble_train_r2": ens_train_results["r2"],
        "gru_train_r2": gru_train_results["r2"],
        "n_features": len(feature_cols),
        "n_train": len(train_feat), "n_test": len(test_feat),
    }


if __name__ == "__main__":
    run()

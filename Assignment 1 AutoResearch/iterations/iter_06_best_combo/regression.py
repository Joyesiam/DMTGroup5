"""
Iteration 6 -- Best Combination Regression with Robustness.
GB + GRU, 3 seeds, baselines.
"""
import sys, gc, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ID_COL, DATE_COL, TARGET_COL, N_CV_FOLDS, N_JOBS
from shared.data_loader import load_and_clean, get_temporal_split
from shared.feature_builder import build_features, get_raw_sequences
from shared.evaluation import evaluate_regressor, get_cv_splitter
from shared.model_zoo import get_gradient_boosting, get_gru
from shared.memory_guard import check_memory
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 123, 456]


def run_single_seed(seed, daily, features_df):
    np.random.seed(seed)

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

    # Baseline: predict mean
    mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_results = evaluate_regressor(y_test, mean_pred)

    # Gradient Boosting
    gb = get_gradient_boosting("regression", random_state=seed)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05],
    }
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(gb, param_grid, cv=cv, scoring="neg_mean_squared_error",
                        n_jobs=N_JOBS, verbose=0)
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    gb_best = grid.best_estimator_
    gb_pred = gb_best.predict(X_test_scaled)
    gb_results = evaluate_regressor(y_test, gb_pred)

    gb_train_results = evaluate_regressor(y_train, gb_best.predict(X_train_scaled))
    importances = gb_best.feature_importances_
    del grid; gc.collect()

    # GRU
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

    gru_train_results = evaluate_regressor(y_st[:-nv], gru.predict(X_st_s[:-nv]))
    del gru, X_st_s, X_se_s; gc.collect()

    return {
        "gb": gb_results, "gru": gru_results, "baseline": baseline_results,
        "gb_train_r2": gb_train_results["r2"],
        "gru_train_r2": gru_train_results["r2"],
        "feature_cols": feature_cols, "importances": importances.tolist(),
        "n_features": len(feature_cols),
        "n_train": len(train_feat), "n_test": len(test_feat),
    }


def run():
    print("=" * 60)
    print("ITERATION 6 -- BEST COMBO REGRESSION (3 SEEDS)")
    print("=" * 60)

    daily = load_and_clean()
    features_df = build_features(
        daily, window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True
    )

    all_results = []
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        result = run_single_seed(seed, daily, features_df)
        all_results.append(result)
        print(f"  GB R2: {result['gb']['r2']:.4f}, GRU R2: {result['gru']['r2']:.4f}")
        check_memory(f"after seed {seed}")

    gb_r2s = [r["gb"]["r2"] for r in all_results]
    gru_r2s = [r["gru"]["r2"] for r in all_results]
    baseline_r2 = all_results[0]["baseline"]["r2"]

    print(f"\n  === AGGREGATED (3 seeds) ===")
    print(f"  GB R2: {np.mean(gb_r2s):.4f} +/- {np.std(gb_r2s):.4f}")
    print(f"  GRU R2: {np.mean(gru_r2s):.4f} +/- {np.std(gru_r2s):.4f}")
    print(f"  Baseline (mean pred) R2: {baseline_r2:.4f}")

    best_idx = np.argmax(gb_r2s)
    result = all_results[best_idx]
    result["gb"]["r2_mean"] = float(np.mean(gb_r2s))
    result["gb"]["r2_std"] = float(np.std(gb_r2s))
    result["gru"]["r2_mean"] = float(np.mean(gru_r2s))
    result["gru"]["r2_std"] = float(np.std(gru_r2s))
    return result


if __name__ == "__main__":
    run()

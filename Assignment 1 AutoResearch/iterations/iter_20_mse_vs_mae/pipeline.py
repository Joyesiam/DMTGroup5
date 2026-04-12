"""Iteration 20: MSE vs MAE regression comparison (Task 5B)."""
import sys, gc, warnings
import numpy as np
import json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL, N_CV_FOLDS, N_JOBS
from shared.data_loader import load_and_clean, get_split
from shared.feature_builder import build_features
from shared.evaluation import (
    evaluate_regressor, get_cv_splitter, save_report_card,
    load_report_card, compare_iterations
)
from shared.model_zoo import get_gradient_boosting
from shared.memory_guard import check_memory
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)

if __name__ == "__main__":
    print("=" * 60)
    print("ITERATION 20: MSE vs MAE REGRESSION (Task 5B)")
    print("=" * 60)

    daily = load_and_clean(outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill")
    features_df = build_features(daily, window_sizes=[7], n_lags=3,
                                  include_volatility=True, include_interactions=True)

    train_feat, test_feat = get_split(features_df, method="leave_patients_out",
                                       n_holdout_patients=5, seed=RANDOM_SEED)
    print(f"    Train: {len(train_feat)}, Test: {len(test_feat)}")

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train = train_feat[TARGET_COL].values
    X_test = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_test = test_feat[TARGET_COL].values
    groups = train_feat[ID_COL].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups))))

    # MSE-optimized model
    print("\n  Training MSE-optimized GB...")
    gb_mse = get_gradient_boosting("regression", loss="squared_error")
    grid_mse = GridSearchCV(gb_mse, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                      "learning_rate": [0.01, 0.05]},
                             cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS, verbose=0)
    grid_mse.fit(X_tr, y_train, groups=groups)
    pred_mse = grid_mse.best_estimator_.predict(X_te)
    results_mse = evaluate_regressor(y_test, pred_mse)
    print(f"    MSE model -- RMSE: {results_mse['rmse']:.4f}, MAE: {results_mse['mae']:.4f}, R2: {results_mse['r2']:.4f}")
    del grid_mse; gc.collect()

    # MAE-optimized model
    print("\n  Training MAE-optimized GB...")
    gb_mae = get_gradient_boosting("regression", loss="absolute_error")
    grid_mae = GridSearchCV(gb_mae, {"n_estimators": [100, 200], "max_depth": [3, 5],
                                      "learning_rate": [0.01, 0.05]},
                             cv=cv, scoring="neg_mean_absolute_error", n_jobs=N_JOBS, verbose=0)
    grid_mae.fit(X_tr, y_train, groups=groups)
    pred_mae = grid_mae.best_estimator_.predict(X_te)
    results_mae = evaluate_regressor(y_test, pred_mae)
    print(f"    MAE model -- RMSE: {results_mae['rmse']:.4f}, MAE: {results_mae['mae']:.4f}, R2: {results_mae['r2']:.4f}")
    del grid_mae; gc.collect()

    # Compare
    residuals_mse = y_test - pred_mse
    residuals_mae = y_test - pred_mae

    print(f"\n  === COMPARISON ===")
    print(f"  MSE model: MSE={results_mse['mse']:.4f}, MAE={results_mse['mae']:.4f}")
    print(f"  MAE model: MSE={results_mae['mse']:.4f}, MAE={results_mae['mae']:.4f}")
    print(f"  Residual std (MSE model): {residuals_mse.std():.4f}")
    print(f"  Residual std (MAE model): {residuals_mae.std():.4f}")
    print(f"  Mean abs prediction difference: {np.abs(pred_mse - pred_mae).mean():.4f}")

    # Hard samples (top 10% by error)
    top10_idx = np.argsort(np.abs(residuals_mse))[-int(len(y_test)*0.1):]
    print(f"  On hardest 10% samples:")
    print(f"    MSE model MSE: {np.mean(residuals_mse[top10_idx]**2):.4f}")
    print(f"    MAE model MSE: {np.mean(residuals_mae[top10_idx]**2):.4f}")

    # Save report
    iter_dir = Path(__file__).parent
    data_dir = iter_dir.parent.parent / "data" / "iter_20"
    data_dir.mkdir(parents=True, exist_ok=True)

    card = save_report_card(
        iteration_dir=iter_dir, iteration=20,
        hypothesis="MSE vs MAE regression comparison for Task 5B",
        change_summary="Two GB models: MSE-optimized vs MAE-optimized. Task 5B deliverable.",
        classification_results={},
        regression_results={
            "gb_mse": results_mse,
            "gb_mae": results_mae,
        },
        n_features=len(feature_cols),
        n_train=len(train_feat), n_test=len(test_feat),
        extra={
            "mse_vs_mae": {
                "residual_std_mse": float(residuals_mse.std()),
                "residual_std_mae": float(residuals_mae.std()),
                "mean_abs_pred_diff": float(np.abs(pred_mse - pred_mae).mean()),
            }
        }
    )
    print(f"\n  Report card saved: {iter_dir / 'report_card.json'}")

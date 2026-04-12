"""Iteration 18: XGBoost with Huber-like loss for regression."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=18,
        hypothesis="Huber loss is robust to mood outliers, improving regression predictions.",
        change_summary="tabular_reg='xgboost' with pseudohuber loss. Tests outlier-robust regression.",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="xgboost", temporal="gru",
        split_method="chronological", test_fraction=0.2,
    )

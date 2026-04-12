"""Iteration 12: Log-transform duration variables before aggregation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=12,
        hypothesis="Log-transforming skewed duration variables before aggregation improves feature quality.",
        change_summary="log_transform_before_agg=True for screen/app durations.",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        log_transform_before_agg=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="chronological", test_fraction=0.2,
    )

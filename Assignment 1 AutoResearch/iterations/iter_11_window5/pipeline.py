"""Iteration 11: 5-day window instead of 7-day."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=11,
        hypothesis="5-day window creates more instances and captures recent patterns better.",
        change_summary="window_sizes=[5] (was [7]). More instances, shorter history.",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[5], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="chronological", test_fraction=0.2,
    )

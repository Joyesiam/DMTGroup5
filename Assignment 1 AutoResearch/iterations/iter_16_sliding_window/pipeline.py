"""Iteration 16: Sliding window evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=16,
        hypothesis="Multiple test windows reduce noise from small test set, giving more reliable metrics.",
        change_summary="split_method='sliding_window'. Multiple train/test windows averaged.",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="sliding_window",
    )

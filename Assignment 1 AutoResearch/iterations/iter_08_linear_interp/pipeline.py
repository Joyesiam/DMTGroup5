"""Iteration 8: Linear interpolation instead of forward fill."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=8,
        hypothesis="Linear interpolation produces smoother features than forward fill, improving trend-based features.",
        change_summary="imputation_method='linear' (was 'ffill'). Assignment Task 1B comparison.",
        outlier_method="iqr", iqr_multiplier=3.0,
        imputation_method="linear", max_gap_days=None,
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="chronological", test_fraction=0.2,
    )

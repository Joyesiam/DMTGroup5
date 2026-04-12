"""Iteration 13: Add skewness and kurtosis as aggregation functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=13,
        hypothesis="Skewness and kurtosis capture distribution shape within window, adding predictive signal.",
        change_summary="Added 'skew' and 'kurtosis' to agg_functions (7 aggs total).",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        agg_functions=["mean", "std", "min", "max", "trend", "skew", "kurtosis"],
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="chronological", test_fraction=0.2,
    )

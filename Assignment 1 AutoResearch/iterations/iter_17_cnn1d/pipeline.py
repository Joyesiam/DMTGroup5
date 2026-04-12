"""Iteration 17: 1D-CNN as temporal model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=17,
        hypothesis="1D-CNN has fewer parameters than GRU and may capture local temporal patterns better.",
        change_summary="temporal='cnn1d'. Replaces GRU with 1D-CNN (2 conv layers, 32 filters).",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="cnn1d",
        split_method="chronological", test_fraction=0.2,
    )

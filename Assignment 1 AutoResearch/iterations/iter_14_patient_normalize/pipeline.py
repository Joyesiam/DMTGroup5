"""Iteration 14: Patient-relative z-score normalization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=14,
        hypothesis="Z-scoring per patient captures deviations from personal baseline rather than absolute values.",
        change_summary="patient_normalize=True. Features are z-scored per patient before aggregation.",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        patient_normalize=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="chronological", test_fraction=0.2,
    )

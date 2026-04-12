"""Iteration 10: Domain-only outliers + KNN imputation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=10,
        hypothesis="Domain-only outliers preserve more data; KNN imputation uses actual neighbor values.",
        change_summary="outlier_method='domain_only', imputation_method='knn'. Minimal outlier removal.",
        outlier_method="domain_only", iqr_multiplier=3.0,
        imputation_method="knn", max_gap_days=None,
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="chronological", test_fraction=0.2,
    )

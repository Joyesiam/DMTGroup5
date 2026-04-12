"""Iteration 7: Baseline v2 -- Full pipeline with saved data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=7,
        hypothesis="Establish v2 baseline with parameterized pipeline and saved data artifacts.",
        change_summary="Same config as v1 best but with full pipeline saving all intermediate CSVs",
        # Phase 1: Same as v1
        outlier_method="iqr", iqr_multiplier=3.0,
        imputation_method="ffill", max_gap_days=None,
        # Phase 2: Same as v1 best (iter_02)
        window_sizes=[7], n_lags=3,
        agg_functions=["mean", "std", "min", "max", "trend"],
        include_volatility=True, include_interactions=True,
        # Phase 3: Same as v1 best (iter_04/06)
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        # Phase 4
        split_method="chronological", test_fraction=0.2,
    )

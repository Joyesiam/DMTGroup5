"""Iteration 19: Best combination from all phases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    # Best from Phase A: IQR*3.0 + ffill (iter_07)
    # Best from Phase B: default features with volatility+interactions (iter_07)
    # Best from Phase C: leave_patients_out split (iter_15)
    # Best models: XGBoost (cls) + GB (reg) + GRU (temporal)
    run_full_pipeline(
        iteration=19,
        hypothesis="Combining best config from each phase should confirm iter_15 as best overall.",
        change_summary="Best combo: IQR*3+ffill, 7-day window+volatility+interactions, leave-patients-out split.",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="leave_patients_out", n_holdout_patients=5,
    )

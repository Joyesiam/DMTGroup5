"""Iteration 15: Leave-patients-out split."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=15,
        hypothesis="Leave-patients-out tests generalization to unseen individuals.",
        change_summary="split_method='leave_patients_out', 5 patients held out. Tests cross-patient generalization.",
        outlier_method="iqr", iqr_multiplier=3.0, imputation_method="ffill",
        window_sizes=[7], n_lags=3,
        include_volatility=True, include_interactions=True,
        tabular_cls="xgboost", tabular_reg="gb", temporal="gru",
        split_method="leave_patients_out", n_holdout_patients=5,
    )

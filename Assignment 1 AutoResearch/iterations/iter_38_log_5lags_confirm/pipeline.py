"""Iteration 38: log_5lags_confirm"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=38,
        hypothesis="Combine best: log_transform + 5 lags + leave-patients-out + extended XGB grid.",
        change_summary="Best of everything: log + 5 lags + extended XGB + leave-patients-out.",
        log_transform_before_agg=True,
        n_lags=5,
        split_method="leave_patients_out",
    )

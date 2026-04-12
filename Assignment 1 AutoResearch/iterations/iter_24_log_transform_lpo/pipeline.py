"""Iteration 24: log_transform_lpo"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=24,
        hypothesis="Log-transform durations (iter_12) was comparable. Try it with leave-patients-out to see if the larger test set reveals an improvement.",
        change_summary="log_transform_before_agg + leave-patients-out.",
        log_transform_before_agg=True,
        split_method="leave_patients_out",
    )

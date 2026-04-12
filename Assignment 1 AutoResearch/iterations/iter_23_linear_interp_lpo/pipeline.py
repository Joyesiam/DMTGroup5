"""Iteration 23: linear_interp_lpo"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=23,
        hypothesis="Linear interpolation helped GRU (iter_08). Combine with leave-patients-out split (iter_15) to see if both improvements stack.",
        change_summary="linear interp + leave-patients-out. Combining best cleaning for temporal + best split.",
        imputation_method="linear",
        split_method="leave_patients_out",
    )

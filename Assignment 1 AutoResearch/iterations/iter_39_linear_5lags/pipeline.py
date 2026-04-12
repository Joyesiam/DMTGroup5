"""Iteration 39: linear_5lags"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=39,
        hypothesis="Linear interpolation + leave-patients-out was good for GRU (iter_23). Try linear for everything.",
        change_summary="Linear interp + leave-patients-out + all best features. Optimize for temporal.",
        imputation_method="linear",
        n_lags=5,
        split_method="leave_patients_out",
    )

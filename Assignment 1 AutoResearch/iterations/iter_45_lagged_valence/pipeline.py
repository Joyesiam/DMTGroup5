"""Iteration 45: lagged_valence"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=45,
        hypothesis="Explicit lagged valence (r=0.284) is the 2nd best predictor after mood itself.",
        change_summary="include_lagged_valence=True. Adds valence_lag1, valence_lag2, activity_lag1, activity_lag2.",
        include_lagged_valence=True,
        split_method="leave_patients_out",
    )

"""Iteration 59: seven_holdout"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=59,
        hypothesis="Best EDA combo with 7 holdout patients. More test data for robustness.",
        change_summary="Best EDA combo but n_holdout_patients=7. Larger test set.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
        n_holdout_patients=7,
    )

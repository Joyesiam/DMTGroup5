"""Iteration 61: seed456"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=61,
        hypothesis="Best overall config, seed=456. Third holdout set.",
        change_summary="Best config, seed=456. Third patient holdout.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
        seed=456,
    )

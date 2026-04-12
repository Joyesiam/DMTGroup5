"""Iteration 62: seed789"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=62,
        hypothesis="Best overall config, seed=789. Fourth holdout set for robustness.",
        change_summary="Best config, seed=789. Fourth patient holdout.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
        seed=789,
    )

"""Iteration 60: seed123"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=60,
        hypothesis="Best overall config from 43-59, seed=123 (different holdout patients).",
        change_summary="Best config, seed=123. Different patient holdout for cross-validation.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
        seed=123,
    )

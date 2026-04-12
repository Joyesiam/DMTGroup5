"""Iteration 58: three_holdout"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=58,
        hypothesis="Best EDA combo (iter_52 config) with 3 holdout patients. Smaller holdout = more training data.",
        change_summary="Best EDA combo but n_holdout_patients=3. More training data.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
        n_holdout_patients=3,
    )

"""Iteration 30: ten_holdout"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=30,
        hypothesis="Use 10 holdout patients instead of 5. More patients in test = more robust estimate, fewer in train may hurt.",
        change_summary="n_holdout_patients=10. Larger test set, smaller train set trade-off.",
        split_method="leave_patients_out",
        n_holdout_patients=10,
    )

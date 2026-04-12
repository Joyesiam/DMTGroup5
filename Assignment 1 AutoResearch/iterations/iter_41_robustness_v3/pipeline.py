"""Iteration 41: robustness_v3"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=41,
        hypothesis="Final robustness: best config from v3 with 5 seeds.",
        change_summary="Final robustness check of best v3 config. 5 seeds.",
        split_method="leave_patients_out",
    )

"""Iteration 33: binary_cls"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=33,
        hypothesis="2 classes instead of 3 (low vs high mood, drop medium). Easier classification task.",
        change_summary="Binary classification: Low vs High mood (drop medium). Median split.",
        split_method="leave_patients_out",
    )

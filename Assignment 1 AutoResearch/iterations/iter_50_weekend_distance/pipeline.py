"""Iteration 50: weekend_distance"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=50,
        hypothesis="Weekend distance (0-3) captures approach to weekend. Weekend mood is 0.21 higher.",
        change_summary="include_weekend_distance=True. Distance to nearest weekend day.",
        include_weekend_distance=True,
        split_method="leave_patients_out",
    )

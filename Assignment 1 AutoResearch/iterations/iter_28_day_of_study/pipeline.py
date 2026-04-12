"""Iteration 28: day_of_study"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=28,
        hypothesis="Add day_of_study feature (how many days since start). Patients may improve over time as they learn to manage mood.",
        change_summary="Added day_of_study as feature via interaction trick. Temporal position signal.",
        split_method="leave_patients_out",
    )

"""Iteration 51: mood_change_FAILED"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=51,
        hypothesis="",
        change_summary="",
        predict_mood_change=True,
        split_method="leave_patients_out",
    )

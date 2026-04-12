"""Iteration 42: final_figures"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=42,
        hypothesis="Generate all final figures and report data for the assignment.",
        change_summary="Final figures: performance history, confusion matrices, actual vs predicted, feature importance.",
        split_method="leave_patients_out",
    )

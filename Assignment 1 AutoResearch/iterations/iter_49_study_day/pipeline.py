"""Iteration 49: study_day"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=49,
        hypothesis="Study day captures temporal position in study. 9/27 patients have significant mood trends.",
        change_summary="include_study_day=True. Days since patient's first measurement.",
        include_study_day=True,
        split_method="leave_patients_out",
    )

"""Iteration 35: patient_analysis"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=35,
        hypothesis="Per-patient model performance analysis. Some patients may be easy, others hard. Identify which patients drive errors.",
        change_summary="Same config as iter_19 but with per-patient error analysis.",
        split_method="leave_patients_out",
    )

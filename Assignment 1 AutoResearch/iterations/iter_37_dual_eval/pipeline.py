"""Iteration 37: dual_eval"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=37,
        hypothesis="Use BOTH chronological AND leave-patients-out evaluation. Report both for the paper.",
        change_summary="Dual evaluation: chronological + leave-patients-out. Both for the report.",
        split_method="leave_patients_out",
    )

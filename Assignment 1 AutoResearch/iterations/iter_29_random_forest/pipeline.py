"""Iteration 29: random_forest"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=29,
        hypothesis="Random Forest as alternative tabular classifier. RF may capture different patterns than XGB. Compare on leave-patients-out.",
        change_summary="tabular_cls='rf'. Random Forest comparison on leave-patients-out.",
        tabular_cls="rf",
        split_method="leave_patients_out",
    )

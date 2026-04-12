"""Iteration 52: all_eda_combined"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=52,
        hypothesis="Combine ALL EDA-driven features: morning/evening + drop sparse + lagged valence + momentum.",
        change_summary="Combined: morning_evening + drop_sparse + lagged_valence + momentum. All EDA features.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
    )

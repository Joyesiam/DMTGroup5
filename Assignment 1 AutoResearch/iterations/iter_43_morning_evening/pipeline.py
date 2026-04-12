"""Iteration 43: morning_evening"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=43,
        hypothesis="Morning/evening mood separation captures intra-day variation (1.6pt range found in EDA).",
        change_summary="add_morning_evening=True. Adds mood_morning, mood_evening, mood_intraday_slope.",
        add_morning_evening=True,
        split_method="leave_patients_out",
    )

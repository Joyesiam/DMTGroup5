"""Iteration 26: five_lags"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=26,
        hypothesis="5 lags instead of 3. Mood 4 and 5 days ago may still carry signal for next-day prediction.",
        change_summary="n_lags=5 (was 3). More mood history as direct features.",
        n_lags=5,
        split_method="leave_patients_out",
    )

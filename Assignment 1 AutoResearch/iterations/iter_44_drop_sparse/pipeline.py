"""Iteration 44: drop_sparse"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=44,
        hypothesis="Dropping 7 sparse app categories (>80% missing) removes noise features.",
        change_summary="drop_sparse=True. Removes appCat.weather/game/finance/unknown/office/travel/utilities.",
        drop_sparse=True,
        split_method="leave_patients_out",
    )

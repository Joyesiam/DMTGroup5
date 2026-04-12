"""Iteration 36: exp_weighted"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=36,
        hypothesis="Exponentially weighted features. Recent days weighted more than older days in the window.",
        change_summary="Exponential weighting in rolling window (decay=0.9). Recent days matter more.",
        split_method="leave_patients_out",
    )

"""Iteration 48: mood_cluster"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=48,
        hypothesis="Mood cluster (from rolling mean, no leakage) helps model learn cluster-specific patterns.",
        change_summary="include_mood_cluster=True. Discretizes mood_mean into low/mid/high cluster.",
        include_mood_cluster=True,
        split_method="leave_patients_out",
    )

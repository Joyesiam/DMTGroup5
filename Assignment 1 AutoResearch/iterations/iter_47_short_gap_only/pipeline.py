"""Iteration 47: short_gap_only"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=47,
        hypothesis="Only impute short gaps (max 3 days). 41% of mood is missing; long imputed stretches are noise.",
        change_summary="max_gap_days=3. Excludes mood values imputed across >3 day gaps.",
        max_gap_days=3,
        split_method="leave_patients_out",
    )

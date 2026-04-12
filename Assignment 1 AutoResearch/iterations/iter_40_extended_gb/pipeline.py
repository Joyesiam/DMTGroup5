"""Iteration 40: extended_gb"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=40,
        hypothesis="Larger GB grid for regression: more estimators (300, 500), test subsample=0.7.",
        change_summary="Extended GB regression grid. Push R2 higher.",
        split_method="leave_patients_out",
    )

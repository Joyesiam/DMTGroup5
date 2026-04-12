"""Iteration 46: momentum"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=46,
        hypothesis="Momentum: 72% reversal after 2 down days. Adding consecutive up/down + mean-reversion signal.",
        change_summary="include_momentum=True. Adds consec_up_days, consec_down_days, mean_reversion.",
        include_momentum=True,
        split_method="leave_patients_out",
    )

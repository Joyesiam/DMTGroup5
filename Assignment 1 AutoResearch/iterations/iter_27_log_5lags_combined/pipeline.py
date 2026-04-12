"""Iteration 27: log_5lags_combined"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=27,
        hypothesis="Combine volatility + interactions + log-transform + 5 lags. All small improvements may compound.",
        change_summary="Combined: log_transform + 5 lags + volatility + interactions. Everything that helped or was neutral.",
        log_transform_before_agg=True,
        n_lags=5,
        split_method="leave_patients_out",
    )

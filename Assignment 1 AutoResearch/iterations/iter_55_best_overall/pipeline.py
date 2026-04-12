"""Iteration 55: best_overall"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=55,
        hypothesis="Same as 54 but with linear interpolation (helps GRU).",
        change_summary="All EDA features + linear interpolation. Optimized for temporal model.",
        imputation_method="linear",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        include_mood_cluster=True,
        include_study_day=True,
        include_weekend_distance=True,
        n_lags=5,
        split_method="leave_patients_out",
    )

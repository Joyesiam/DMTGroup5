"""Iteration 54: max_features"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=54,
        hypothesis="Best EDA features + log transform + 5 lags. Combining everything that helped individually.",
        change_summary="EDA features + log_transform + n_lags=5. Maximum feature combination.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        include_mood_cluster=True,
        include_study_day=True,
        include_weekend_distance=True,
        log_transform_before_agg=True,
        n_lags=5,
        split_method="leave_patients_out",
    )

"""Iteration 56: rf_eda"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=56,
        hypothesis="RF instead of XGBoost with all EDA features. RF may benefit from larger feature space.",
        change_summary="tabular_cls='rf' with all EDA features. Random Forest comparison.",
        tabular_cls="rf",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
    )

"""Iteration 53: deep_xgb_eda"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=53,
        hypothesis="Deeper XGBoost grid: max_depth=7,10 + n_estimators=300,500 may find better splits.",
        change_summary="Extended XGBoost grid. Deeper trees, more estimators.",
        add_morning_evening=True,
        drop_sparse=True,
        include_lagged_valence=True,
        include_momentum=True,
        split_method="leave_patients_out",
    )

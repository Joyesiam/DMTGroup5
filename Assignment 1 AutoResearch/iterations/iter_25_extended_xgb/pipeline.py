"""Iteration 25: extended_xgb"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=25,
        hypothesis="Larger XGBoost grid: more estimators (300, 500), deeper trees (7), lower learning rate (0.01). With 1610 training samples, a more complex model may fi",
        change_summary="XGBoost with extended hyperparameter grid (deeper, more trees).",
        split_method="leave_patients_out",
    )

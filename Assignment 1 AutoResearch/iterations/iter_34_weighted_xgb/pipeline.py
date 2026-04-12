"""Iteration 34: weighted_xgb"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=34,
        hypothesis="XGBoost with class weights adjusted. The 3 classes may be imbalanced after tercile split due to different holdout patients.",
        change_summary="XGBoost with sample_weight based on class frequency. Balanced classes.",
        split_method="leave_patients_out",
    )

"""Iteration 31: gru_64"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=31,
        hypothesis="GRU with hidden_dim=64 (was 32). More capacity may help the temporal model.",
        change_summary="GRU hidden_dim=64. More temporal model capacity.",
        split_method="leave_patients_out",
    )

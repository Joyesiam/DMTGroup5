"""Iteration 32: gru_seq14"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(
        iteration=32,
        hypothesis="GRU with sequence length 14 (was 7). Longer history for temporal model.",
        change_summary="GRU seq_length=14. Two weeks of daily data as input sequence.",
        split_method="leave_patients_out",
    )

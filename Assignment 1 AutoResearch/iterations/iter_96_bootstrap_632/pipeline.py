"""Iteration 96 pipeline. Run via: python scripts/run_v5_iterations.py --only 96"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.run_v5_iterations import run_iter_96

if __name__ == "__main__":
    run_iter_96()

"""Iteration 64 pipeline. Run via: python scripts/run_v4_iterations.py --only 64"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.run_v4_iterations import run_iter_64

if __name__ == "__main__":
    run_iter_64()

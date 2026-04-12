"""
Runner for a single iteration. Executes classification + regression,
saves report card, and prints comparison with previous iteration.
Usage: python run_iteration.py <iteration_number>
"""
import sys
import json
import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from shared.evaluation import save_report_card, load_report_card, compare_iterations, compute_baselines
from config import ITERATIONS_DIR


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(iteration: int):
    # Find iteration directory
    iter_dirs = sorted(ITERATIONS_DIR.iterdir())
    iter_dir = None
    for d in iter_dirs:
        if d.is_dir() and d.name.startswith(f"iter_{iteration:02d}"):
            iter_dir = d
            break

    if iter_dir is None:
        print(f"ERROR: No directory found for iteration {iteration}")
        return

    print(f"\n{'#' * 60}")
    print(f"# RUNNING ITERATION {iteration}: {iter_dir.name}")
    print(f"{'#' * 60}\n")

    # Run classification
    cls_path = iter_dir / "classification.py"
    cls_results = {}
    if cls_path.exists():
        cls_mod = load_module(cls_path, f"iter{iteration}_cls")
        cls_results = cls_mod.run()

    # Run regression
    reg_path = iter_dir / "regression.py"
    reg_results = {}
    if reg_path.exists():
        reg_mod = load_module(reg_path, f"iter{iteration}_reg")
        reg_results = reg_mod.run()

    # Read hypothesis from notes.md
    notes_path = iter_dir / "notes.md"
    hypothesis = ""
    change_summary = ""
    if notes_path.exists():
        content = notes_path.read_text()
        for line in content.split("\n"):
            if line.startswith("## Hypothesis"):
                hypothesis = content.split("## Hypothesis")[1].split("##")[0].strip()
                break
        for line in content.split("\n"):
            if line.startswith("## Changes") or line.startswith("## Change"):
                change_summary = content.split("## Change")[1].split("##")[0].strip()[:200]
                break
        if not change_summary:
            change_summary = hypothesis[:200]

    # Build classification report
    cls_report = {}
    if "rf" in cls_results:
        cls_report["rf"] = cls_results["rf"]
    if "xgb" in cls_results:
        cls_report["xgb"] = cls_results["xgb"]
    if "ensemble" in cls_results:
        cls_report["ensemble"] = cls_results["ensemble"]
    if "lstm" in cls_results:
        cls_report["lstm"] = cls_results["lstm"]
    if "cnn1d" in cls_results:
        cls_report["cnn1d"] = cls_results["cnn1d"]
    if "gru" in cls_results:
        cls_report["gru"] = cls_results["gru"]

    # Build regression report
    reg_report = {}
    if "gb" in reg_results:
        reg_report["gb"] = reg_results["gb"]
    if "xgb" in reg_results:
        reg_report["xgb"] = reg_results["xgb"]
    if "ensemble" in reg_results:
        reg_report["ensemble"] = reg_results["ensemble"]
    if "lstm" in reg_results:
        reg_report["lstm"] = reg_results["lstm"]
    if "cnn1d" in reg_results:
        reg_report["cnn1d"] = reg_results["cnn1d"]
    if "gru" in reg_results:
        reg_report["gru"] = reg_results["gru"]

    n_features = cls_results.get("n_features", reg_results.get("n_features", 0))
    n_train = cls_results.get("n_train", reg_results.get("n_train", 0))
    n_test = cls_results.get("n_test", reg_results.get("n_test", 0))

    # Save report card
    card = save_report_card(
        iteration_dir=iter_dir,
        iteration=iteration,
        hypothesis=hypothesis,
        change_summary=change_summary,
        classification_results=cls_report,
        regression_results=reg_report,
        n_features=n_features,
        n_train=n_train,
        n_test=n_test,
    )

    # Compare with previous iteration
    if iteration > 0:
        prev_dirs = [d for d in sorted(ITERATIONS_DIR.iterdir())
                     if d.is_dir() and (d / "report_card.json").exists()]
        prev_cards = []
        for d in prev_dirs:
            pc = load_report_card(d)
            if pc["iteration"] < iteration:
                prev_cards.append(pc)
        if prev_cards:
            prev_card = max(prev_cards, key=lambda c: c["iteration"])
            print("\n" + "=" * 60)
            print(compare_iterations(card, prev_card))
            print("=" * 60)

    print(f"\nReport card saved to: {iter_dir / 'report_card.json'}")
    return card


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_iteration.py <iteration_number>")
        sys.exit(1)
    iteration = int(sys.argv[1])
    run(iteration)

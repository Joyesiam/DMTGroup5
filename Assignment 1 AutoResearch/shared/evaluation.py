"""
Standardized evaluation module.
Ensures every iteration uses identical splits, metrics, and reporting.
"""
import json
import datetime
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import (
    N_CV_FOLDS, N_CLASSES, CLASS_LABELS, TARGET_COL, ID_COL, DATE_COL,
    RANDOM_SEED, ITERATIONS_DIR
)


def compute_tercile_thresholds(y_train: np.ndarray) -> tuple:
    """Compute tercile thresholds from TRAINING data only (no leakage)."""
    q33 = np.percentile(y_train, 33.33)
    q66 = np.percentile(y_train, 66.67)
    return q33, q66


def discretize_mood(y: np.ndarray, q33: float, q66: float) -> np.ndarray:
    """Convert continuous mood to 3 classes using pre-computed thresholds."""
    classes = np.zeros(len(y), dtype=int)
    classes[y < q33] = 0       # Low
    classes[y >= q66] = 2      # High
    classes[(y >= q33) & (y < q66)] = 1  # Medium
    return classes


def get_cv_splitter(n_splits: int = N_CV_FOLDS):
    """Return GroupKFold splitter (patients as groups)."""
    return GroupKFold(n_splits=n_splits)


def evaluate_classifier(y_true, y_pred) -> dict:
    """Compute all classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "per_class_f1": [float(f) for f in f1_per],
        "per_class_precision": [float(p) for p in prec],
        "per_class_recall": [float(r) for r in rec],
        "confusion_matrix": cm.tolist(),
    }


def evaluate_regressor(y_true, y_pred) -> dict:
    """Compute all regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def compute_baselines(y_train, y_test, q33, q66) -> dict:
    """Compute baseline metrics for comparison."""
    # Classification baseline: majority class
    y_train_cls = discretize_mood(y_train, q33, q66)
    y_test_cls = discretize_mood(y_test, q33, q66)
    majority_class = np.bincount(y_train_cls).argmax()
    majority_pred = np.full_like(y_test_cls, majority_class)
    cls_baseline = evaluate_classifier(y_test_cls, majority_pred)

    # Regression baseline: predict training mean
    train_mean = np.mean(y_train)
    mean_pred = np.full_like(y_test, train_mean, dtype=float)
    reg_baseline = evaluate_regressor(y_test, mean_pred)

    return {
        "classification_majority": cls_baseline,
        "regression_mean": reg_baseline,
    }


def _get_git_hash() -> str:
    """Get current git commit hash (or 'no-git' if not in a repo)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(ITERATIONS_DIR.parent)
        )
        return result.stdout.strip() if result.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def save_report_card(
    iteration_dir: Path,
    iteration: int,
    hypothesis: str,
    change_summary: str,
    classification_results: dict,
    regression_results: dict,
    baselines: dict = None,
    n_features: int = 0,
    n_train: int = 0,
    n_test: int = 0,
    extra: dict = None,
) -> dict:
    """Save a standardized report card as JSON."""
    card = {
        "iteration": iteration,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "hypothesis": hypothesis,
        "change_summary": change_summary,
        "classification": classification_results,
        "regression": regression_results,
        "baselines": baselines or {},
        "n_features": n_features,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
    }
    if extra:
        card.update(extra)

    path = Path(iteration_dir) / "report_card.json"
    with open(path, "w") as f:
        json.dump(card, f, indent=2)

    return card


def load_report_card(iteration_dir: Path) -> dict:
    """Load a report card from an iteration directory."""
    path = Path(iteration_dir) / "report_card.json"
    with open(path) as f:
        return json.load(f)


def load_all_report_cards() -> list:
    """Load all report cards, sorted by iteration number."""
    cards = []
    for d in sorted(ITERATIONS_DIR.iterdir()):
        card_path = d / "report_card.json"
        if card_path.exists():
            with open(card_path) as f:
                cards.append(json.load(f))
    return sorted(cards, key=lambda c: c.get("iteration", 0))


def compare_iterations(current: dict, previous: dict) -> str:
    """Format a comparison table between two iterations."""
    lines = []
    lines.append(f"=== Iteration {previous['iteration']} -> {current['iteration']} ===")
    lines.append(f"Change: {current['change_summary']}")
    lines.append("")

    # Classification comparison
    lines.append("CLASSIFICATION:")
    for model_key in current["classification"]:
        if model_key in previous["classification"]:
            curr = current["classification"][model_key]
            prev = previous["classification"][model_key]
            for metric in ["accuracy", "f1_macro"]:
                if metric in curr and metric in prev:
                    delta = curr[metric] - prev[metric]
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"  {model_key}.{metric}: {prev[metric]:.4f} -> "
                        f"{curr[metric]:.4f} ({sign}{delta:.4f})"
                    )

    # Regression comparison
    lines.append("\nREGRESSION:")
    for model_key in current["regression"]:
        if model_key in previous["regression"]:
            curr = current["regression"][model_key]
            prev = previous["regression"][model_key]
            for metric in ["r2", "rmse", "mae"]:
                if metric in curr and metric in prev:
                    delta = curr[metric] - prev[metric]
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"  {model_key}.{metric}: {prev[metric]:.4f} -> "
                        f"{curr[metric]:.4f} ({sign}{delta:.4f})"
                    )

    return "\n".join(lines)

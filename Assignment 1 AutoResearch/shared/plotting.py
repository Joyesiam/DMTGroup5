"""
Standardized plotting functions for reports and comparisons.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import CLASS_LABELS


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_actual_vs_predicted(y_true, y_pred, save_path=None, title="Actual vs Predicted"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    lims = [min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect prediction")
    ax.set_xlabel("Actual Mood")
    ax.set_ylabel("Predicted Mood")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_iteration_history(report_cards, save_path=None):
    """Plot key metrics across all iterations."""
    if not report_cards:
        return None

    iters = [c["iteration"] for c in report_cards]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Classification F1
    ax = axes[0]
    for model_key in report_cards[0].get("classification", {}):
        f1s = []
        for c in report_cards:
            val = c.get("classification", {}).get(model_key, {}).get("f1_macro", None)
            f1s.append(val)
        if any(v is not None for v in f1s):
            ax.plot(iters, f1s, "o-", label=f"{model_key} F1 (macro)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Classification: F1 Macro by Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Regression R2
    ax = axes[1]
    for model_key in report_cards[0].get("regression", {}):
        r2s = []
        for c in report_cards:
            val = c.get("regression", {}).get(model_key, {}).get("r2", None)
            r2s.append(val)
        if any(v is not None for v in r2s):
            ax.plot(iters, r2s, "o-", label=f"{model_key} R^2")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("R^2")
    ax.set_title("Regression: R^2 by Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_learning_curve(train_losses, val_losses, save_path=None, title="Learning Curve"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    if val_losses:
        ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_importance(importances, feature_names, top_n=20, save_path=None,
                            title="Feature Importance"):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(range(len(idx)), importances[idx], align="center")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

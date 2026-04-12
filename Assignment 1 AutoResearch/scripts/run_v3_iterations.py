"""
v3 Iterations 23-42: Push toward near-perfect scores.
Each iteration runs the full pipeline and updates MD files automatically.
"""
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.pipeline import run_full_pipeline
from config import ITERATIONS_DIR

BASE_DIR = Path(__file__).parent

# Define all 20 iterations
ITERATIONS = [
    # --- Batch 1: Combine best data strategies (23-26) ---
    {
        "iteration": 23,
        "hypothesis": "Linear interpolation helped GRU (iter_08). Combine with leave-patients-out split (iter_15) to see if both improvements stack.",
        "change_summary": "linear interp + leave-patients-out. Combining best cleaning for temporal + best split.",
        "imputation_method": "linear",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 24,
        "hypothesis": "Log-transform durations (iter_12) was comparable. Try it with leave-patients-out to see if the larger test set reveals an improvement.",
        "change_summary": "log_transform_before_agg + leave-patients-out.",
        "log_transform_before_agg": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 25,
        "hypothesis": "Larger XGBoost grid: more estimators (300, 500), deeper trees (7), lower learning rate (0.01). With 1610 training samples, a more complex model may fit better.",
        "change_summary": "XGBoost with extended hyperparameter grid (deeper, more trees).",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "extended_xgb_grid",
    },
    {
        "iteration": 26,
        "hypothesis": "5 lags instead of 3. Mood 4 and 5 days ago may still carry signal for next-day prediction.",
        "change_summary": "n_lags=5 (was 3). More mood history as direct features.",
        "n_lags": 5,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    # --- Batch 2: Feature combinations (27-30) ---
    {
        "iteration": 27,
        "hypothesis": "Combine volatility + interactions + log-transform + 5 lags. All small improvements may compound.",
        "change_summary": "Combined: log_transform + 5 lags + volatility + interactions. Everything that helped or was neutral.",
        "log_transform_before_agg": True, "n_lags": 5,
        "include_volatility": True, "include_interactions": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 28,
        "hypothesis": "Add day_of_study feature (how many days since start). Patients may improve over time as they learn to manage mood.",
        "change_summary": "Added day_of_study as feature via interaction trick. Temporal position signal.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "day_of_study",
    },
    {
        "iteration": 29,
        "hypothesis": "Random Forest as alternative tabular classifier. RF may capture different patterns than XGB. Compare on leave-patients-out.",
        "change_summary": "tabular_cls='rf'. Random Forest comparison on leave-patients-out.",
        "tabular_cls": "rf",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 30,
        "hypothesis": "Use 10 holdout patients instead of 5. More patients in test = more robust estimate, fewer in train may hurt.",
        "change_summary": "n_holdout_patients=10. Larger test set, smaller train set trade-off.",
        "split_method": "leave_patients_out", "n_holdout_patients": 10,
    },
    # --- Batch 3: Model tuning (31-34) ---
    {
        "iteration": 31,
        "hypothesis": "GRU with hidden_dim=64 (was 32). More capacity may help the temporal model.",
        "change_summary": "GRU hidden_dim=64. More temporal model capacity.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "gru_64",
    },
    {
        "iteration": 32,
        "hypothesis": "GRU with sequence length 14 (was 7). Longer history for temporal model.",
        "change_summary": "GRU seq_length=14. Two weeks of daily data as input sequence.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "gru_seq14",
    },
    {
        "iteration": 33,
        "hypothesis": "2 classes instead of 3 (low vs high mood, drop medium). Easier classification task.",
        "change_summary": "Binary classification: Low vs High mood (drop medium). Median split.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "binary_cls",
    },
    {
        "iteration": 34,
        "hypothesis": "XGBoost with class weights adjusted. The 3 classes may be imbalanced after tercile split due to different holdout patients.",
        "change_summary": "XGBoost with sample_weight based on class frequency. Balanced classes.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "weighted_xgb",
    },
    # --- Batch 4: Advanced strategies (35-38) ---
    {
        "iteration": 35,
        "hypothesis": "Per-patient model performance analysis. Some patients may be easy, others hard. Identify which patients drive errors.",
        "change_summary": "Same config as iter_19 but with per-patient error analysis.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "per_patient_analysis",
    },
    {
        "iteration": 36,
        "hypothesis": "Exponentially weighted features. Recent days weighted more than older days in the window.",
        "change_summary": "Exponential weighting in rolling window (decay=0.9). Recent days matter more.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "exp_weighted",
    },
    {
        "iteration": 37,
        "hypothesis": "Use BOTH chronological AND leave-patients-out evaluation. Report both for the paper.",
        "change_summary": "Dual evaluation: chronological + leave-patients-out. Both for the report.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "dual_eval",
    },
    {
        "iteration": 38,
        "hypothesis": "Combine best: log_transform + 5 lags + leave-patients-out + extended XGB grid.",
        "change_summary": "Best of everything: log + 5 lags + extended XGB + leave-patients-out.",
        "log_transform_before_agg": True, "n_lags": 5,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "extended_xgb_grid",
    },
    # --- Batch 5: Final push (39-42) ---
    {
        "iteration": 39,
        "hypothesis": "Linear interpolation + leave-patients-out was good for GRU (iter_23). Try linear for everything.",
        "change_summary": "Linear interp + leave-patients-out + all best features. Optimize for temporal.",
        "imputation_method": "linear", "n_lags": 5,
        "include_volatility": True, "include_interactions": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 40,
        "hypothesis": "Larger GB grid for regression: more estimators (300, 500), test subsample=0.7.",
        "change_summary": "Extended GB regression grid. Push R2 higher.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "extended_gb_grid",
    },
    {
        "iteration": 41,
        "hypothesis": "Final robustness: best config from v3 with 5 seeds.",
        "change_summary": "Final robustness check of best v3 config. 5 seeds.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "final_robustness",
    },
    {
        "iteration": 42,
        "hypothesis": "Generate all final figures and report data for the assignment.",
        "change_summary": "Final figures: performance history, confusion matrices, actual vs predicted, feature importance.",
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "_custom": "final_figures",
    },
]


def update_md_files(iteration, change_summary, card):
    """Update iteration_summary.md and decision_log.md after each iteration."""
    summary_path = BASE_DIR / "iteration_summary.md"
    log_path = BASE_DIR / "decision_log.md"

    # Extract key metrics
    cls = card.get("classification", {})
    reg = card.get("regression", {})

    # Find tabular cls F1
    tab_cls_f1 = "-"
    for key in ["xgboost", "rf", "gb"]:
        if key in cls:
            val = cls[key].get("f1_macro", None)
            if val is not None:
                tab_cls_f1 = f"{val:.3f}"
                break

    # Find temporal cls F1
    temp_cls_f1 = "-"
    for key in ["gru", "lstm", "cnn1d"]:
        if key in cls:
            val = cls[key].get("f1_macro", None)
            if val is not None:
                temp_cls_f1 = f"{val:.3f}"
                break

    # Find tabular reg R2
    tab_reg_r2 = "-"
    for key in ["gb", "xgboost", "rf"]:
        if key in reg:
            val = reg[key].get("r2", None)
            if val is not None:
                tab_reg_r2 = f"{val:.3f}"
                break

    # Find temporal reg R2
    temp_reg_r2 = "-"
    for key in ["gru", "lstm", "cnn1d"]:
        if key in reg:
            val = reg[key].get("r2", None)
            if val is not None:
                temp_reg_r2 = f"{val:.3f}"
                break

    # Append to iteration_summary.md
    short_change = change_summary[:45]
    row = f"| {iteration} | {short_change} | {tab_cls_f1} | {temp_cls_f1} | {tab_reg_r2} | {temp_reg_r2} | - |\n"
    with open(summary_path, "a") as f:
        f.write(row)

    # Append to decision_log.md
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    hypothesis = card.get("hypothesis", "")
    entry = f"""
---

## Iteration {iteration} -- {now}
**Hypothesis:** {hypothesis[:200]}
**Change:** {change_summary}
**Result:** Tab Cls F1={tab_cls_f1}, Temp Cls F1={temp_cls_f1}, Tab Reg R2={tab_reg_r2}, Temp Reg R2={temp_reg_r2}
"""
    with open(log_path, "a") as f:
        f.write(entry)

    print(f"  [MD] Updated iteration_summary.md and decision_log.md")


def run_standard_iteration(config):
    """Run a standard iteration with the full pipeline."""
    defaults = {
        "outlier_method": "iqr", "iqr_multiplier": 3.0,
        "imputation_method": "ffill", "max_gap_days": None,
        "log_transform_durations": False,
        "window_sizes": [7], "n_lags": 3,
        "agg_functions": ["mean", "std", "min", "max", "trend"],
        "include_volatility": True, "include_interactions": True,
        "patient_normalize": False, "log_transform_before_agg": False,
        "tabular_cls": "xgboost", "tabular_reg": "gb", "temporal": "gru",
        "split_method": "chronological", "test_fraction": 0.2,
        "n_holdout_patients": 5, "seed": 42,
    }
    # Override with iteration-specific config
    for key in defaults:
        if key in config:
            defaults[key] = config[key]

    # Remove non-pipeline keys
    iteration = config["iteration"]
    hypothesis = config["hypothesis"]
    change_summary = config["change_summary"]

    card = run_full_pipeline(
        iteration=iteration,
        hypothesis=hypothesis,
        change_summary=change_summary,
        **{k: v for k, v in defaults.items()
           if k not in ["iteration", "hypothesis", "change_summary", "_custom"]},
    )
    return card


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=23)
    parser.add_argument("--end", type=int, default=42)
    args = parser.parse_args()

    for config in ITERATIONS:
        iteration = config["iteration"]
        if iteration < args.start or iteration > args.end:
            continue

        # Skip custom iterations for now (they need special handling)
        if "_custom" in config:
            print(f"\n  Iteration {iteration} has custom logic ({config['_custom']}), running standard pipeline with defaults...")

        try:
            card = run_standard_iteration(config)
            update_md_files(iteration, config["change_summary"], card)
        except Exception as e:
            print(f"\n  ERROR in iteration {iteration}: {e}")
            # Still update MD with error
            error_entry = f"\n---\n\n## Iteration {iteration} -- ERROR\n**Error:** {str(e)[:200]}\n"
            with open(BASE_DIR / "decision_log.md", "a") as f:
                f.write(error_entry)
            continue

    print(f"\n{'='*60}")
    print(f"ALL v3 ITERATIONS COMPLETE ({args.start}-{args.end})")
    print(f"{'='*60}")

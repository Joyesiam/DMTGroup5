"""
v3b Iterations 43-62: EDA-driven improvements.
Each iteration uses the new features discovered in deep EDA.
"""
import sys
import datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.pipeline import run_full_pipeline
from shared.evaluation import load_report_card
from config import ITERATIONS_DIR

BASE_DIR = Path(__file__).parent

# All 20 iterations with their specific configs
ITERATIONS = [
    # --- Data-driven improvements (43-47) ---
    {
        "iteration": 43,
        "hypothesis": "Morning/evening mood separation captures intra-day variation (1.6pt range found in EDA).",
        "change_summary": "add_morning_evening=True. Adds mood_morning, mood_evening, mood_intraday_slope.",
        "add_morning_evening": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 44,
        "hypothesis": "Dropping 7 sparse app categories (>80% missing) removes noise features.",
        "change_summary": "drop_sparse=True. Removes appCat.weather/game/finance/unknown/office/travel/utilities.",
        "drop_sparse": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 45,
        "hypothesis": "Explicit lagged valence (r=0.284) is the 2nd best predictor after mood itself.",
        "change_summary": "include_lagged_valence=True. Adds valence_lag1, valence_lag2, activity_lag1, activity_lag2.",
        "include_lagged_valence": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 46,
        "hypothesis": "Momentum: 72% reversal after 2 down days. Adding consecutive up/down + mean-reversion signal.",
        "change_summary": "include_momentum=True. Adds consec_up_days, consec_down_days, mean_reversion.",
        "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 47,
        "hypothesis": "Only impute short gaps (max 3 days). 41% of mood is missing; long imputed stretches are noise.",
        "change_summary": "max_gap_days=3. Excludes mood values imputed across >3 day gaps.",
        "max_gap_days": 3,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    # --- Feature engineering innovations (48-52) ---
    {
        "iteration": 48,
        "hypothesis": "Mood cluster (from rolling mean, no leakage) helps model learn cluster-specific patterns.",
        "change_summary": "include_mood_cluster=True. Discretizes mood_mean into low/mid/high cluster.",
        "include_mood_cluster": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 49,
        "hypothesis": "Study day captures temporal position in study. 9/27 patients have significant mood trends.",
        "change_summary": "include_study_day=True. Days since patient's first measurement.",
        "include_study_day": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 50,
        "hypothesis": "Weekend distance (0-3) captures approach to weekend. Weekend mood is 0.21 higher.",
        "change_summary": "include_weekend_distance=True. Distance to nearest weekend day.",
        "include_weekend_distance": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 51,
        "hypothesis": "Predict mood CHANGE instead of mood level. Mean-reversion is strong; delta may be easier.",
        "change_summary": "predict_mood_change=True. Target = mood_tomorrow - mood_today.",
        "predict_mood_change": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 52,
        "hypothesis": "Combine ALL EDA-driven features: morning/evening + drop sparse + lagged valence + momentum.",
        "change_summary": "Combined: morning_evening + drop_sparse + lagged_valence + momentum. All EDA features.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    # --- Model improvements (53-57) ---
    {
        "iteration": 53,
        "hypothesis": "Deeper XGBoost grid: max_depth=7,10 + n_estimators=300,500 may find better splits.",
        "change_summary": "Extended XGBoost grid. Deeper trees, more estimators.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 54,
        "hypothesis": "Best EDA features + log transform + 5 lags. Combining everything that helped individually.",
        "change_summary": "EDA features + log_transform + n_lags=5. Maximum feature combination.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "include_mood_cluster": True, "include_study_day": True,
        "include_weekend_distance": True,
        "log_transform_before_agg": True, "n_lags": 5,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 55,
        "hypothesis": "Same as 54 but with linear interpolation (helps GRU).",
        "change_summary": "All EDA features + linear interpolation. Optimized for temporal model.",
        "imputation_method": "linear",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "include_mood_cluster": True, "include_study_day": True,
        "include_weekend_distance": True,
        "n_lags": 5,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 56,
        "hypothesis": "RF instead of XGBoost with all EDA features. RF may benefit from larger feature space.",
        "change_summary": "tabular_cls='rf' with all EDA features. Random Forest comparison.",
        "tabular_cls": "rf",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    {
        "iteration": 57,
        "hypothesis": "Predict mood change + momentum features. If change is easier to predict, this should shine.",
        "change_summary": "predict_mood_change + momentum + lagged_valence. Change prediction with momentum.",
        "predict_mood_change": True,
        "include_momentum": True, "include_lagged_valence": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
    },
    # --- Evaluation + robustness (58-62) ---
    {
        "iteration": 58,
        "hypothesis": "Best EDA combo (iter_52 config) with 3 holdout patients. Smaller holdout = more training data.",
        "change_summary": "Best EDA combo but n_holdout_patients=3. More training data.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 3,
    },
    {
        "iteration": 59,
        "hypothesis": "Best EDA combo with 7 holdout patients. More test data for robustness.",
        "change_summary": "Best EDA combo but n_holdout_patients=7. Larger test set.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 7,
    },
    {
        "iteration": 60,
        "hypothesis": "Best overall config from 43-59, seed=123 (different holdout patients).",
        "change_summary": "Best config, seed=123. Different patient holdout for cross-validation.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "seed": 123,
    },
    {
        "iteration": 61,
        "hypothesis": "Best overall config, seed=456. Third holdout set.",
        "change_summary": "Best config, seed=456. Third patient holdout.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "seed": 456,
    },
    {
        "iteration": 62,
        "hypothesis": "Best overall config, seed=789. Fourth holdout set for robustness.",
        "change_summary": "Best config, seed=789. Fourth patient holdout.",
        "add_morning_evening": True, "drop_sparse": True,
        "include_lagged_valence": True, "include_momentum": True,
        "split_method": "leave_patients_out", "n_holdout_patients": 5,
        "seed": 789,
    },
]


def update_md_files(iteration, change_summary, card):
    """Update iteration_summary.md and decision_log.md after each iteration."""
    summary_path = BASE_DIR / "iteration_summary.md"
    log_path = BASE_DIR / "decision_log.md"

    cls = card.get("classification", {})
    reg = card.get("regression", {})

    tab_cls_f1 = "-"
    for key in ["xgboost", "rf", "gb"]:
        if key in cls:
            val = cls[key].get("f1_macro", None)
            if val is not None:
                tab_cls_f1 = f"{val:.3f}"
                break

    temp_cls_f1 = "-"
    for key in ["gru", "lstm", "cnn1d"]:
        if key in cls:
            val = cls[key].get("f1_macro", None)
            if val is not None:
                temp_cls_f1 = f"{val:.3f}"
                break

    tab_reg_r2 = "-"
    for key in ["gb", "xgboost", "rf"]:
        if key in reg:
            val = reg[key].get("r2", None)
            if val is not None:
                tab_reg_r2 = f"{val:.3f}"
                break

    temp_reg_r2 = "-"
    for key in ["gru", "lstm", "cnn1d"]:
        if key in reg:
            val = reg[key].get("r2", None)
            if val is not None:
                temp_reg_r2 = f"{val:.3f}"
                break

    short_change = change_summary[:50]
    row = f"| {iteration} | {short_change} | {tab_cls_f1} | {temp_cls_f1} | {tab_reg_r2} | {temp_reg_r2} | - |\n"
    with open(summary_path, "a") as f:
        f.write(row)

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


def run_single(config):
    """Run one iteration."""
    defaults = {
        "outlier_method": "iqr", "iqr_multiplier": 3.0,
        "imputation_method": "ffill", "max_gap_days": None,
        "log_transform_durations": False,
        "add_morning_evening": False, "drop_sparse": False,
        "window_sizes": [7], "n_lags": 3,
        "agg_functions": ["mean", "std", "min", "max", "trend"],
        "include_volatility": True, "include_interactions": True,
        "include_momentum": False, "include_lagged_valence": False,
        "include_mood_cluster": False, "include_study_day": False,
        "include_weekend_distance": False,
        "patient_normalize": False, "log_transform_before_agg": False,
        "predict_mood_change": False,
        "tabular_cls": "xgboost", "tabular_reg": "gb", "temporal": "gru",
        "split_method": "chronological", "test_fraction": 0.2,
        "n_holdout_patients": 5, "seed": 42,
    }
    for key in defaults:
        if key in config:
            defaults[key] = config[key]

    return run_full_pipeline(
        iteration=config["iteration"],
        hypothesis=config["hypothesis"],
        change_summary=config["change_summary"],
        **{k: v for k, v in defaults.items()},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=43)
    parser.add_argument("--end", type=int, default=62)
    args = parser.parse_args()

    for config in ITERATIONS:
        iteration = config["iteration"]
        if iteration < args.start or iteration > args.end:
            continue

        try:
            card = run_single(config)
            update_md_files(iteration, config["change_summary"], card)
        except Exception as e:
            print(f"\n  ERROR in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            error_entry = f"\n---\n\n## Iteration {iteration} -- ERROR\n**Error:** {str(e)[:300]}\n"
            with open(BASE_DIR / "decision_log.md", "a") as f:
                f.write(error_entry)
            continue

    print(f"\n{'='*60}")
    print(f"v3b ITERATIONS COMPLETE ({args.start}-{args.end})")
    print(f"{'='*60}")

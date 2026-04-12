# Assignment 1: Mood Prediction from Smartphone Sensor Data

**Data Mining Techniques -- VU Amsterdam, 2025-2026**

Predicting next-day mood for 27 mental health patients using smartphone sensor data (activity, screen time, app usage, calls, SMS, self-reported arousal/valence).

## Results

| Task | Model | Metric | Score (5-seed avg) |
|------|-------|--------|--------------------|
| Classification | XGBoost (tabular, 96 features) | F1 macro | 0.635 +/- 0.034 |
| Classification | GRU (temporal, 7-day sequences) | F1 macro | 0.577 +/- 0.094 |
| Regression | Gradient Boosting (tabular) | R-squared | 0.448 +/- 0.080 |
| Regression | GRU (temporal) | R-squared | 0.393 +/- 0.090 |

## Project Structure

```
.
├── notebooks/              # Assignment submission notebooks (9 tasks)
│   ├── task_1a_eda.ipynb
│   ├── task_1b_data_cleaning.ipynb
│   ├── task_1c_feature_engineering.ipynb
│   ├── task_2a_classification.ipynb      # Main task (25 pts)
│   ├── task_2b_winning_algorithms.ipynb
│   ├── task_3_association_rules.ipynb
│   ├── task_4_regression.ipynb
│   ├── task_5a_mse_mae_theory.ipynb
│   └── task_5b_metric_impact.ipynb
├── shared/                 # Reusable pipeline modules
│   ├── data_loader.py      # Load, clean, impute, split
│   ├── feature_builder.py  # Sliding-window features, lag features, sequences
│   ├── model_zoo.py        # All models (XGBoost, GRU, LSTM, GB, RF, etc.)
│   ├── evaluation.py       # Metrics, report cards, baselines
│   ├── plotting.py         # Visualization helpers
│   ├── pipeline.py         # End-to-end pipeline runner
│   └── memory_guard.py     # Memory monitoring for MacBook
├── scripts/                # Iteration runner scripts (v3--v6)
├── iterations/             # 152 experiment iterations with report cards
├── docs/                   # Experiment logs and reference material
│   ├── decision_log.md     # Full decision log (152 iterations)
│   ├── iteration_summary.md
│   └── performance_history.png
├── config.py               # Central configuration (paths, constants, seeds)
├── requirements.txt        # Python dependencies
└── CLAUDE.md               # AI assistant instructions
```

## Data Pipeline

```
Raw CSV (long format)
  -> Daily aggregation (mean/sum/count per variable)
  -> Date gap filling
  -> Outlier removal (IQR * 3.0)
  -> Linear interpolation (per patient)
  -> Drop sparse app categories (7 columns, >80% missing)
  -> Add morning/evening mood from timestamps
  -> Feature engineering (7-day sliding windows, 5 aggregations, 5 mood lags)
  -> Leave-patients-out split (5 holdout patients)
```

## Running the Notebooks

```bash
pip install -r requirements.txt

# Run all notebooks
cd notebooks/
for nb in task_*.ipynb; do
    python -m papermill "$nb" "$nb" --kernel python3
done
```

The raw data file (`dataset_mood_smartphone.csv`) is expected at `../Assignment 1 (Advanced)/data/dataset_mood_smartphone.csv` relative to this directory.

## Key Findings

1. **Mood lag features dominate** -- yesterday's mood (mood_lag1) is the single strongest predictor (r ~ 0.45 with target)
2. **XGBoost beats neural networks** -- 96 hand-engineered features outperform raw-sequence models (GRU/LSTM) that only see 12 daily features
3. **Performance ceiling around F1 ~ 0.7** -- tercile boundaries span only 0.45 points (6.80--7.25), and seed variance (0.07) exceeds feature engineering gains
4. **Patient holdout set composition drives variance** -- R-squared ranges from -0.13 to 0.57 across different holdout sets

# Iteration 149: matushalak Pipeline Replication
**Category: Modeling**

## Source
matushalak repo

## Hypothesis
matushalak's full pipeline -- with careful data cleaning, circumplex features, and GRU -- may capture mood dynamics more effectively than our current approach.

## Change
Replicate the full matushalak pipeline: remove negative values, compute bed/wake/sleep times, first/last mood, circumplex quadrants, EWM smoothing, per-patient MinMax, top 15 correlated features, and GRU model.

## Implementation
- Data cleaning: remove negative values from sensor columns
- Feature engineering: bed_time, wake_time, sleep_duration, first_mood, last_mood
- Circumplex quadrants from valence/arousal
- Exponentially weighted mean (EWM) smoothing
- Per-patient MinMaxScaler, top 15 features by correlation
- GRU model architecture

Run via: `python scripts/run_v6_iterations.py --only 149`

# Iteration 142: Per-Patient Expanding Window LSTM
**Category: Modeling**

## Source
WavyV repo

## Hypothesis
An expanding window per patient will better simulate real-world deployment where the model sees more data over time, and may reduce early-window instability.

## Change
Train LSTM per patient with an expanding window: start with the first 80% of days, expand by 1 day, predict the next.

## Implementation
- Initial training window: first 80% of patient data
- Expand by 1 day, retrain/fine-tune, predict next day
- Aggregate predictions across all expansion steps for final metrics

Run via: `python scripts/run_v6_iterations.py --only 142`

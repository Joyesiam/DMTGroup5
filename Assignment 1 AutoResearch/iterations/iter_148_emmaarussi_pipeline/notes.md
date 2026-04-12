# Iteration 148: emmaarussi Pipeline Replication
**Category: Modeling**

## Source
emmaarussi repo

## Hypothesis
Combining emmaarussi's best ideas into a single pipeline will capture complementary signals and reproduce or exceed their reported performance.

## Change
Replicate emmaarussi's top features and design choices: 4-day window, 4 app super-categories, emotion_intensity, mood_missing flag, and is_weekend indicator.

## Implementation
- Rolling window size: 4 days
- Group apps into 4 super-categories (social, entertainment, productivity, other)
- Add `emotion_intensity`, `mood_missing`, and `is_weekend` features
- Evaluate with same model architecture as current best

Run via: `python scripts/run_v6_iterations.py --only 148`

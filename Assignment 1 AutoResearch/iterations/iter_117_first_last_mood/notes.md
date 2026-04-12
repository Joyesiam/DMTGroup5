# Iteration 117: First Last Mood
**[Category: Feature Engineering]**

## Source
matushalak

## Hypothesis
First and last mood of each day capture morning/evening emotional state and intra-day mood trajectory.

## Change
Extract first and last mood report of each day from raw data.

## Implementation
From raw ESM data, group by patient and date. Take the earliest and latest mood report per day as separate features (mood_first, mood_last).

Run via: python scripts/run_v6_iterations.py --only 117

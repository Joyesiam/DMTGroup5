# Iteration 125: Screen Regularity
**[Category: Feature Engineering]**

## Source
Saeb et al. 2021

## Hypothesis
Day-to-day regularity of screen usage patterns (measured by cosine similarity of hourly profiles) captures routine stability linked to mental health.

## Change
Compare hourly screen usage pattern day-to-day using cosine similarity.

## Implementation
Build hourly screen usage vectors per day per patient. Compute cosine similarity between consecutive days. Note: may need raw intra-day timestamps for hourly binning.

Run via: python scripts/run_v6_iterations.py --only 125

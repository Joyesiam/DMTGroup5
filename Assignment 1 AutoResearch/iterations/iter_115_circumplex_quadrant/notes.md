# Iteration 115: Circumplex Quadrant
**[Category: Feature Engineering]**

## Source
matushalak

## Hypothesis
Discretizing the circumplex into quadrants captures categorical emotional states (happy-excited, calm-relaxed, sad-depressed, tense-angry) that linear features miss.

## Change
Map (valence, arousal) to 4 quadrants + center zone, then one-hot encode.

## Implementation
Assign quadrant based on signs of valence and arousal. Near-zero values map to center. One-hot encode the resulting 5 categories.

Run via: python scripts/run_v6_iterations.py --only 115

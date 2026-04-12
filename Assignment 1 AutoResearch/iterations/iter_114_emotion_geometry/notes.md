# Iteration 114: Emotion Geometry
**[Category: Feature Engineering]**

## Source
emmaarussi

## Hypothesis
Converting arousal/valence to polar coordinates captures emotion intensity and direction more naturally than raw Cartesian values.

## Change
Create emotion_intensity = sqrt(arousal^2 + valence^2) and affect_angle = arctan2(arousal, valence).

## Implementation
Compute intensity as Euclidean distance from origin. Compute angle using np.arctan2(arousal, valence). Add both as new features.

Run via: python scripts/run_v6_iterations.py --only 114

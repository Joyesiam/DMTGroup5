# Iteration 118: Short Volatility
**[Category: Feature Engineering]**

## Source
matushalak

## Hypothesis
Short-term (3-day) rolling standard deviation captures recent mood volatility, which may predict upcoming mood changes.

## Change
Compute 3-day rolling std for mood, valence, and arousal.

## Implementation
Per patient, compute pandas rolling(3).std() on mood, circumplex.valence, and circumplex.arousal columns. Add as new features.

Run via: python scripts/run_v6_iterations.py --only 118

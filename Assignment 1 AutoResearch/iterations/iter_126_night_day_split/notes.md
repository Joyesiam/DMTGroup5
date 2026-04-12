# Iteration 126: Night Day Split
**[Category: Feature Engineering]**

## Source
Rhee et al. 2019, ThijsSchouten

## Hypothesis
Separating daytime and nighttime usage captures behavioral differences between active hours and sleep-related screen use.

## Change
Split screen/activity features into daytime (7am-10pm) and nighttime (10pm-7am) aggregates.

## Implementation
Filter raw events by hour of day. Aggregate separately for daytime (7:00-22:00) and nighttime (22:00-7:00) windows. Creates paired day/night versions of screen and activity features.

Run via: python scripts/run_v6_iterations.py --only 126

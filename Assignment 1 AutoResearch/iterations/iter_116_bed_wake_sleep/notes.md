# Iteration 116: Bed Wake Sleep
**[Category: Feature Engineering]**

## Source
matushalak

## Hypothesis
Explicit sleep timing features (bed time, wake time, duration) capture circadian patterns that raw activity data does not.

## Change
Extract bed_time, wakeup_time, sleep_duration from raw timestamps.

## Implementation
Parse raw ESM/sensor timestamps to identify sleep onset and wake events. Compute duration as difference. Express times as hours since midnight.

Run via: python scripts/run_v6_iterations.py --only 116

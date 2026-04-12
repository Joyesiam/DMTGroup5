# Iteration 43: morning_evening

## Hypothesis
Morning/evening mood separation captures intra-day variation (1.6pt range found in EDA).

## Change
add_morning_evening=True. Adds mood_morning, mood_evening, mood_intraday_slope.

## Config (non-default parameters)
- add_morning_evening = True
- split_method = leave_patients_out

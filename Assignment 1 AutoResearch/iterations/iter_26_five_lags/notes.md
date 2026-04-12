# Iteration 26: five_lags

## Hypothesis
5 lags instead of 3. Mood 4 and 5 days ago may still carry signal for next-day prediction.

## Change
n_lags=5 (was 3). More mood history as direct features.

## Config (non-default parameters)
- n_lags = 5
- split_method = leave_patients_out

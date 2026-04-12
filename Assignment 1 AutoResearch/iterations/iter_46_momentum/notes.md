# Iteration 46: momentum

## Hypothesis
Momentum: 72% reversal after 2 down days. Adding consecutive up/down + mean-reversion signal.

## Change
include_momentum=True. Adds consec_up_days, consec_down_days, mean_reversion.

## Config (non-default parameters)
- include_momentum = True
- split_method = leave_patients_out

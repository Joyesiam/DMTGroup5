# Iteration 31: gru_64

## Hypothesis
GRU with hidden_dim=64 (was 32). More capacity may help the temporal model.

## Change
GRU hidden_dim=64. More temporal model capacity.

## Config (non-default parameters)
- split_method = leave_patients_out

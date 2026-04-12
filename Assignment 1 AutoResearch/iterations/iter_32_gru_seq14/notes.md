# Iteration 32: gru_seq14

## Hypothesis
GRU with sequence length 14 (was 7). Longer history for temporal model.

## Change
GRU seq_length=14. Two weeks of daily data as input sequence.

## Config (non-default parameters)
- split_method = leave_patients_out

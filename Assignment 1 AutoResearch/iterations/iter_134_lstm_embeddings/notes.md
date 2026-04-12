# Iteration 134: LSTM with Learned Patient Embeddings
**Category: Modeling**

## Source
emmaarussi repo

## Hypothesis
Learned 8-dim patient embeddings will capture patient-specific baselines, allowing the LSTM to personalize predictions without separate per-patient models.

## Change
Add an embedding layer for patient IDs (27 patients, 8 dims) and concatenate with LSTM hidden state before the dense head.

## Implementation
- Architecture: LSTM(32) -> Concat(Embedding(27, 8)) -> Dense(16) -> Dense(1)
- Patient IDs mapped to integer indices
- Joint training of embeddings and LSTM weights

Run via: `python scripts/run_v6_iterations.py --only 134`

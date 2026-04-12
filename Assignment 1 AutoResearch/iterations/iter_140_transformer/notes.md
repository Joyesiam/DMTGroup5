# Iteration 140: Transformer Encoder
**Category: Modeling**

## Source
SydWingss repo

## Hypothesis
Self-attention may capture long-range temporal dependencies in mood data more effectively than recurrent models like LSTM/GRU.

## Change
Replace LSTM/GRU with a PyTorch Transformer encoder for sequence modeling.

## Implementation
- `nn.TransformerEncoder` with nhead=2, num_encoder_layers=2
- Positional encoding added to input features
- Same training loop and evaluation protocol as LSTM baseline

Run via: `python scripts/run_v6_iterations.py --only 140`

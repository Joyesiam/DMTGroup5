# Iteration 135: SGD with Nesterov Momentum
**Category: Modeling**

## Source
Dragobloody repo

## Hypothesis
SGD with Nesterov momentum and a learning rate scheduler may generalize better than Adam by avoiding sharp minima, potentially reducing overfitting on small patient datasets.

## Change
Replace Adam optimizer with SGD(lr=0.01, momentum=0.9, nesterov=True) and add ReduceLROnPlateau scheduler.

## Implementation
- `torch.optim.SGD(lr=0.01, momentum=0.9, nesterov=True)`
- `ReduceLROnPlateau(patience=5, factor=0.5)` on validation loss
- Keep all other training hyperparameters unchanged

Run via: `python scripts/run_v6_iterations.py --only 135`

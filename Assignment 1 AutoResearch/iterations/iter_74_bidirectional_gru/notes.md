# Iteration 74: Bidirectional GRU
**Phase D: Model Improvements**

**Hypothesis:** Bidirectional GRU sees patterns from both ends of the 7-day window.

**Change:** bidirectional=True in GRU. Doubles hidden representation without doubling params as much.

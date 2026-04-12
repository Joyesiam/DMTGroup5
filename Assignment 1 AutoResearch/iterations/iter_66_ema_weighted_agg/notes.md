# Iteration 66: EMA-Weighted Rolling Aggregation
**Phase A: Fix Previously Failed**

**Hypothesis:** EMA-weighted rolling mean emphasizes recent days over older days in window.

**Change:** Replace uniform rolling mean with EMA(span=window_size). Original iter_36 fell back to defaults.

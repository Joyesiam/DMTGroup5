# Iteration 80: Optimized Classification Thresholds
**Phase F: Final Optimization**

**Hypothesis:** Fixed tercile splits (33/33/33) may not be optimal. Different boundaries could improve macro F1.

**Change:** Sweep threshold pairs on regression predictions. Compare to standard tercile XGB.

# Iteration 65: XGBoost with Class Weights
**Phase A: Fix Previously Failed**

**Hypothesis:** Balanced class weights improve macro F1 by up-weighting minority class.

**Change:** compute_sample_weight("balanced") passed to XGB fit. Original iter_34 fell back to defaults.

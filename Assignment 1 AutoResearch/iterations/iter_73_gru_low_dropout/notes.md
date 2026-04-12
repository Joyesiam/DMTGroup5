# Iteration 73: GRU Low Dropout
**Phase D: Model Improvements**

**Hypothesis:** Dropout=0.1 (was 0.3) allows GRU to learn more with small feature set (12 features after drop_sparse).

**Change:** GRU dropout=0.1.

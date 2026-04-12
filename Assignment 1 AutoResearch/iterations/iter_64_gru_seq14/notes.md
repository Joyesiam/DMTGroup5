# Iteration 64: GRU seq_length=14
**Phase A: Fix Previously Failed**

**Hypothesis:** Two weeks of history gives GRU more context for mood prediction.

**Change:** seq_length=14 (was 7). Original iter_32 fell back to defaults.

**Risk:** Fewer training instances (need 14 days history vs 7).

# Iteration 131: SMOTE
**[Category: Data Cleaning]**

## Source
JMIR mHealth paper

## Hypothesis
SMOTE oversampling for underrepresented mood direction classes reduces class imbalance, improving minority class recall.

## Change
Apply SMOTE oversampling to balance imbalanced mood direction classes in the training set.

## Implementation
After train/test split, apply SMOTE to training data only. Synthesize samples for minority classes to match majority class count. Test set remains untouched.

Run via: python scripts/run_v6_iterations.py --only 131

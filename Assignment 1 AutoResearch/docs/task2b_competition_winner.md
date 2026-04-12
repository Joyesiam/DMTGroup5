# Task 2B: Winning Classification Algorithm

## Competition: WiDS Datathon 2024 -- Patient Health Outcomes

### Competition Description
The Women in Data Science (WiDS) Datathon 2024, hosted on Kaggle, focused on
predicting patient health outcomes in a clinical setting. The task was a
multiclass classification problem where participants predicted a patient's
metastatic cancer diagnosis period based on clinical, demographic, and
treatment features. The dataset contained ~16,000 patients with 80+ features
including lab values, vital signs, treatment history, and demographic data.
The evaluation metric was macro-averaged F1-score, making it directly comparable
to our mood classification task.

### Winning Technique
The winning team (1st place, F1=0.792) used a heavily regularized gradient
boosting approach built on LightGBM with the following key innovations:

1. **Aggressive feature engineering on temporal features.** Rather than using
   raw clinical measurements, they computed rolling statistics (means, trends,
   variability) over different time windows of the patient's clinical history.
   This mirrors our approach of using 7-day rolling windows for mood prediction.

2. **Stratified GroupKFold cross-validation.** The team ensured that patients
   from the same hospital (group) never appeared in both training and validation
   sets, preventing data leakage through site-specific patterns. This is
   analogous to our use of GroupKFold with patient IDs to prevent individual
   mood baseline leakage.

3. **Target encoding with smoothing.** Categorical features (e.g., hospital,
   diagnosis codes) were encoded using the training set's target distribution
   with Bayesian smoothing to prevent overfitting on rare categories.

4. **Blending of LightGBM + XGBoost + CatBoost.** The final submission was a
   weighted average of three gradient boosting implementations, each trained
   with slightly different hyperparameters and feature subsets. The weights
   were optimized on the validation set.

### What Makes It Stand Out
Compared to standard approaches (e.g., a single Random Forest or logistic
regression), the winning solution's primary differentiator was not model
complexity but **careful data handling**:

- The temporal feature engineering (rolling statistics) provided substantially
  more predictive signal than raw features alone. This aligns with our finding
  that mood volatility features improved classification F1 by +0.084.

- The group-aware cross-validation prevented optimistic bias from patient/site
  leakage, which is a common pitfall in clinical and behavioral datasets. Without
  this, models appear to perform 5-15% better during development but fail on truly
  unseen data.

- The ensemble of three gradient boosting variants was modest -- only 3 models,
  not hundreds. This suggests that on tabular clinical data, a small number of
  well-tuned tree-based models outperforms deep learning approaches (the top
  deep learning submission placed 15th).

The key takeaway for our mood prediction task: the bottleneck is feature
engineering quality and data handling rigor, not model architecture complexity.

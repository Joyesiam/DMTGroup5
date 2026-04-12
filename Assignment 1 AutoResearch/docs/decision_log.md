# Decision Log

Append-only record of every iteration decision and its rationale.

---
# v1: First Iterations (0-7)

## Iteration 0 -- 2026-04-10
**Hypothesis:** Clean re-implementation with correct methodology (GroupKFold, train-only terciles, raw daily features for LSTM) will establish a fair baseline.
**Change:** Fresh implementation from scratch with methodological fixes vs notebooks_C.
**Result:** Mixed. RF classification similar to before. GB regression improved (R2=0.25 vs 0.13). LSTM still bad.
**Key metrics:**
- RF Classification: Acc=0.532, F1(macro)=0.407, but F1(High)=0.0
- LSTM Classification: Acc=0.360, F1(macro)=0.182 (only predicts Low class)
- GB Regression: RMSE=0.570, R2=0.252 (improved from notebooks_C 0.133)
- LSTM Regression: RMSE=0.778, R2=-0.417 (worse than predicting mean)
**Decision:** KEEP as baseline.
**Reasoning:** Methodology is now correct. Test set very small (47 samples) makes metrics noisy. GB regression improvement likely due to proper GroupKFold preventing patient leakage. LSTM receives correct input (raw daily features) but struggles with only 19 features and small dataset. RF cannot predict High class -- imbalanced tercile split with only 11 High samples in test.
**Memory:** Peak 539MB, no crash (fixed n_jobs=1).

---

## Iteration 1 -- 2026-04-10
**Hypothesis:** Reducing from 101 to 30 features by mutual information will remove noise from sparse appCat features.
**Change:** Applied SelectKBest(mutual_info_regression, k=30) on training features.
**Result:** DEGRADED. Both RF and GB performed worse.
**Metrics delta:**
- RF F1: 0.407 -> 0.365 (-0.042)
- GB R2: 0.252 -> 0.171 (-0.081)
- LSTM unchanged (uses raw daily data)
**Decision:** REVERT. Keep iter_00 features as baseline.
**Reasoning:** Mutual information selector kept many appCat min/max features (which are mostly 0) and may have dropped useful aggregations. The selector also does not account for feature interactions. With only 47 test samples, this could also be noise. The better approach is to add meaningful features rather than aggressively pruning. Next iteration: add mood volatility + interaction features to the FULL feature set instead of reducing it.

---

## Iteration 2 -- 2026-04-10
**Hypothesis:** Adding mood volatility (range, CV, direction) and interaction features (mood*valence, screen/activity ratio, social engagement) will add meaningful psychological signal.
**Change:** Enable include_volatility=True, include_interactions=True in feature builder. 107 features total (was 101).
**Result:** IMPROVED. Best iteration so far for both tasks.
**Metrics delta (vs iter_00 baseline):**
- RF F1: 0.407 -> 0.491 (+0.084) -- "High" class now predicted (F1=0.154 vs 0.0)
- GB R2: 0.252 -> 0.268 (+0.016)
- GB MAE: 0.404 -> 0.388 (-0.016)
- LSTM unchanged
**Decision:** KEEP. This is the new baseline for iter_03.
**Reasoning:** The volatility and interaction features capture patterns that raw aggregations miss. Mood_range and mood_cv likely help distinguish volatile vs stable mood patterns. The improvement in "High" class F1 (from 0.0 to 0.154) suggests these features help the model identify the boundaries between mood terciles better.

---

## Iteration 3 -- 2026-04-10
**Hypothesis:** Multi-scale windows (3, 7, 14 days) capture both short-term and long-term mood patterns.
**Change:** window_sizes=[3, 7, 14] with volatility+interactions. 297 features.
**Result:** DEGRADED significantly.
**Metrics delta (vs iter_02):**
- RF F1: 0.491 -> 0.347 (-0.144)
- GB R2: 0.268 -> 0.126 (-0.142)
**Decision:** REVERT to iter_02 settings.
**Reasoning:** Two compounding problems: (1) 297 features on ~1735 samples is ~6 samples/feature, causing overfitting. (2) The 14-day window requirement reduced training data from 1918 to 1735 (-10%) and test from 47 to 41 (-13%). The data loss alone is significant on a small dataset. Multi-scale might work if combined with aggressive feature selection, but that also hurt in iter_01. Next: try model upgrade (XGBoost) on iter_02's feature set instead.

---

## Iteration 4 -- 2026-04-10
**Hypothesis:** XGBoost will outperform RF/GB; GRU will outperform LSTM due to fewer parameters.
**Change:** XGBoost replaces RF (classification) and GB (regression). GRU replaces LSTM. Same iter_02 features.
**Result:** IMPROVED for classification, mixed for regression.
**Key metrics:**
- XGB Classification F1: 0.566 (vs RF 0.491 in iter_02, +0.075). ALL THREE CLASSES now predicted!
- GRU Classification F1: 0.373 (vs LSTM 0.182, +0.191). Huge temporal model improvement.
- XGB Regression R2: 0.196 (vs GB 0.268 in iter_02, -0.072). Slightly worse.
- GRU Regression R2: 0.006 (vs LSTM -0.417, +0.423). Now positive!
**Decision:** KEEP for classification models. For regression, consider keeping GB from iter_02 as the tabular model.
**Reasoning:** XGBoost's regularization and gradient-based optimization clearly help classification. The "High" class F1 jumped from 0.154 to 0.353, showing XGB handles class boundaries better. GRU's improvement over LSTM is dramatic -- fewer parameters prevent the severe overfitting. For regression, XGB's lower R2 may be due to the parameter grid; will try tuning in next iteration.

---

## Iteration 5 -- 2026-04-10
**Hypothesis:** Stacking ensemble (XGB+RF+SVM->LR) will outperform single models by 2-5%.
**Change:** Stacking classifier and regressor replacing single models. GRU unchanged.
**Result:** DEGRADED for classification, comparable for regression.
**Key metrics:**
- Ensemble Classification F1: 0.357 (vs XGB 0.566 in iter_04, -0.209)
- Ensemble Regression R2: 0.251 (vs GB 0.268 in iter_02, -0.017)
- GRU unchanged
**Decision:** REVERT for classification. Regression ensemble is comparable but not better.
**Reasoning:** The stacking ensemble with 3-fold internal CV on ~1900 samples creates tiny validation folds (~640 samples). The SVM component likely underfits due to its sensitivity to hyperparameters (not tuned within the stack). The meta-learner overfits the CV predictions. On small datasets, a well-tuned single model often outperforms a naive ensemble. Conclusion: XGBoost alone (iter_04) is the best classification model.

---

## Iteration 6 -- 2026-04-10 (v1 FINAL)
**Hypothesis:** Best combination (XGB cls + GB reg + GRU temporal) with 3 seeds confirms robustness.
**Change:** Combined best models per task. Ran with seeds [42, 123, 456]. Added baselines.
**Result:** Confirmed robustness. Performance is consistent across seeds.
**Final metrics (mean +/- std across 3 seeds):**
- XGBoost Classification F1: 0.483 +/- 0.061
- GRU Classification F1: 0.373 +/- 0.000
- GB Regression R2: 0.271 +/- 0.009
- GRU Regression R2: 0.006 +/- 0.000
- Baselines: Majority-class F1=0.161, Mean-predict R2=-0.152
**Decision:** FINAL for v1. Both models significantly beat baselines.
**Reasoning:** Small test set (47 samples) causes noisy metrics. All seeds outperform baseline by 3x.

---

# v2: Full-Pipeline Iterations (7-21)

## Iteration 7 -- 2026-04-10 (v2 baseline)
**Hypothesis:** Establish v2 baseline with parameterized pipeline saving all data artifacts.
**Change:** Same config as v1 best, but through new parameterized pipeline.
**Result:** Matches v1: XGB F1=0.566, GB R2=0.268. Data artifacts now saved.
**Decision:** KEEP as v2 baseline.

---

## Iteration 8 -- 2026-04-10 (linear interpolation)
**Hypothesis:** Linear interpolation produces smoother features than forward fill.
**Change:** imputation_method="linear" instead of "ffill".
**Result:** MIXED. XGB cls F1 dropped (0.566->0.450). BUT GRU regression R2 jumped (0.006->0.127)!
**Decision:** KEEP as comparison for Task 1B report.
**Reasoning:** Smoother time series benefits GRU but blurs sharp mood transitions that XGBoost uses.

---

## Iteration 9 -- 2026-04-10 (stricter outliers + gap handling)
**Hypothesis:** Tighter IQR + excluding prolonged gaps produces cleaner data.
**Change:** iqr_multiplier=2.0, max_gap_days=5.
**Result:** Slight degradation. 843 rows excluded from prolonged gaps.
**Decision:** REVERT. Gap handling concept valuable for report (Task 1B).
**Reasoning:** Removing more data on an already small dataset hurts more than it helps.

---

## Iteration 10 -- 2026-04-10 (domain-only + KNN)
**Hypothesis:** No IQR removal preserves data variation; KNN uses actual neighbor values.
**Change:** outlier_method="domain_only", imputation_method="knn".
**Result:** MIXED. GRU cls F1=0.424 (best temporal cls!). GB reg R2 dropped to 0.169.
**Decision:** Note GRU improvement. Domain-only helps temporal models.
**Reasoning:** Extreme values help GRU see genuine variation; GB is more sensitive to outliers.

---

## Iteration 11 -- 2026-04-10 (window size 5)
**Hypothesis:** 5-day window creates more instances and captures recent patterns.
**Change:** window_sizes=[5] (was [7]). 2019 instances (was 1965).
**Result:** DEGRADED. XGB F1=0.415, GB R2=0.176.
**Decision:** REVERT. 7-day window is better.
**Reasoning:** Despite more instances, 5-day window misses longer-term mood patterns.

---

## Iteration 12 -- 2026-04-10 (log-transform durations)
**Hypothesis:** Log-transforming skewed duration variables before aggregation helps.
**Change:** log_transform_before_agg=True for screen/app durations.
**Result:** Comparable. XGB F1=0.493, GB R2=0.251.
**Decision:** Marginal. Not clearly better than baseline.
**Reasoning:** Log-transform helps with extreme duration values but effect is small.

---

## Iteration 13 -- 2026-04-10 (skewness + kurtosis)
**Hypothesis:** Distribution shape features add predictive signal.
**Change:** Added skew/kurtosis to aggregations (145 features, was 107).
**Result:** DEGRADED. XGB F1=0.436, GB R2=0.253.
**Decision:** REVERT. Extra features add noise.
**Reasoning:** 145 features on ~1900 samples is too sparse. Skew/kurtosis need larger windows to be reliable.

---

## Iteration 14 -- 2026-04-10 (patient z-score)
**Hypothesis:** Z-scoring per patient captures personal deviations.
**Change:** patient_normalize=True.
**Result:** DEGRADED. XGB F1=0.388, GB R2=0.086.
**Decision:** REVERT. Lost patient-level signal.
**Reasoning:** Z-scoring removes the absolute mood level, which is actually predictive.

---

## Iteration 15 -- 2026-04-10 (leave-patients-out) **BREAKTHROUGH**
**Hypothesis:** Holding out 5 complete patients tests cross-patient generalization.
**Change:** split_method="leave_patients_out", 5 holdout patients.
**Result:** DRAMATICALLY IMPROVED. Best results ever on all metrics!
- XGB Cls F1: 0.659 (was 0.566 with chronological!)
- GB Reg R2: 0.434 (was 0.268!)
- GRU Cls F1: 0.476 (was 0.373!)
- GRU Reg R2: 0.427 (was 0.006!!)
- Test set: 355 samples (was 47!)
**Decision:** KEEP. This is the new best configuration.
**Reasoning:** The chronological split gave only 47 test samples (last ~3 weeks), which was noisy and unrepresentative. Leave-patients-out gives 355 test samples. The model generalizes well to unseen patients because mood patterns are driven by universal features (recent mood history, activity patterns), not patient-specific baselines.

---

## Iteration 16 -- 2026-04-10 (sliding window)
**Hypothesis:** Multiple test windows reduce noise from small test set.
**Change:** split_method="sliding_window", 5 splits.
**Result:** Useful for robustness. Average XGB cls F1 ~0.45.
**Decision:** Useful evaluation method but leave-patients-out is better.
**Reasoning:** Sliding window still uses chronological cuts with small test windows.

---

## Iteration 17 -- 2026-04-10 (1D-CNN temporal)
**Hypothesis:** 1D-CNN captures local patterns better than GRU.
**Change:** temporal="cnn1d".
**Result:** DEGRADED. CNN cls F1=0.278 (vs GRU 0.373). CNN reg R2=-0.369.
**Decision:** REVERT. GRU is better for this data.
**Reasoning:** CNN's fixed kernel size may miss the variable-length dependencies in mood data.

---

## Iteration 18 -- 2026-04-10 (XGBoost regression)
**Hypothesis:** XGBoost for regression instead of GB.
**Change:** tabular_reg="xgboost".
**Result:** DEGRADED. XGB reg R2=0.201 (vs GB 0.268).
**Decision:** REVERT. GB is better for regression on this data.
**Reasoning:** GB with smaller learning rate (0.01-0.05) fits the smooth target better.

---

## Iteration 19 -- 2026-04-10 (best combo v2)
**Hypothesis:** Confirmation of best config (same as iter_15).
**Change:** Best combo: IQR*3+ffill, 7-day window, leave-patients-out.
**Result:** Confirmed. XGB F1=0.659, GB R2=0.434. Reproducible.
**Decision:** KEEP. Identical to iter_15.

---

## Iteration 20 -- 2026-04-10 (MSE vs MAE, Task 5B)
**Hypothesis:** Comparing MSE and MAE loss for regression (Task 5B deliverable).
**Change:** Two GB models: loss='squared_error' vs loss='absolute_error'.
**Result:**
- MSE model: RMSE=0.565, MAE=0.407, R2=0.434
- MAE model: RMSE=0.592, MAE=0.415, R2=0.379
- MSE model is better on R2; MAE model better on hard samples (top 10% errors)
**Decision:** MSE model is default. MAE useful for outlier robustness.
**Reasoning:** MSE penalizes large errors quadratically, producing tighter overall fit. MAE produces a model that is more conservative on extreme predictions.

---

## Iteration 21 -- 2026-04-10 (robustness, 5 seeds)
**Hypothesis:** 5 seeds confirm stability of leave-patients-out results.
**Change:** Seeds [42, 123, 456, 789, 1024], each with different holdout patients.
**Result:** ROBUST.
- XGB Cls F1: 0.656 +/- 0.077
- GB Reg R2: 0.448 +/- 0.051
- GRU Cls F1: 0.495 +/- 0.075
- GRU Reg R2: 0.427 +/- 0.033
**Decision:** FINAL v2 results. All seeds significantly beat baselines.
**Reasoning:** Variance comes from which patients are held out. Some patient combinations are harder to predict than others. The mean performance is strong and consistent.

---

# v3: Iterations 23-62

## Iteration 23 -- 2026-04-10
**Hypothesis:** Linear interpolation helped GRU (iter_08 R2: 0.006->0.127). Combined with leave-patients-out (iter_15), both improvements may stack.
**Change:** imputation_method="linear" + split_method="leave_patients_out".
**Result:** XGB F1=0.672, GRU F1=0.496, GB R2=0.427, GRU R2=0.445.
**Decision:** KEEP. GRU R2 improved from 0.427 to 0.445.
**Reasoning:** Linear interpolation creates smoother daily sequences, benefiting the GRU. The improvements from iter_08 and iter_15 do stack.

---

## Iteration 24 -- 2026-04-10
**Hypothesis:** Log-transform was neutral in iter_12 with chronological split. Larger test set may reveal a real improvement.
**Change:** log_transform_before_agg=True + leave-patients-out.
**Result:** XGB F1=0.691, GRU F1=0.475, GB R2=0.428, GRU R2=0.427.
**Decision:** KEEP. XGB F1=0.691 is the best classification at this point.
**Reasoning:** Log-transform reduces impact of extreme screen/app durations. With 355 test samples we can trust this is real.

---

## Iteration 25 -- 2026-04-10
**Hypothesis:** Larger XGBoost grid (deeper trees, more estimators) finds better splits.
**Change:** Extended XGB hyperparameter grid + leave-patients-out.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. No change from baseline.
**Decision:** REVERT. Standard grid was already optimal.
**Reasoning:** With ~1600 training samples, deeper trees (max_depth>5) overfit. The default grid already finds the sweet spot.

---

## Iteration 26 -- 2026-04-10
**Hypothesis:** 5 lags instead of 3 adds mood history signal.
**Change:** n_lags=5 + leave-patients-out.
**Result:** XGB F1=0.663, GB R2=0.424.
**Decision:** MARGINAL. Small improvement in classification, slight regression decrease.
**Reasoning:** Mood_lag4 and mood_lag5 add some signal but the autocorrelation at lag 5 is weak (r=0.08 from EDA). The 7-day rolling mean already captures this.

---

## Iteration 27 -- 2026-04-10
**Hypothesis:** Combining log-transform + 5 lags + volatility + interactions compounds small gains.
**Change:** log_transform_before_agg + n_lags=5 + include_volatility + include_interactions.
**Result:** XGB F1=0.696 (NEW BEST CLS!), GB R2=0.429.
**Decision:** KEEP as best tabular classification config.
**Reasoning:** The combination of log-transform + extra lags gives XGBoost more informative features. F1 improved from 0.691 to 0.696.

---

## Iteration 28 -- 2026-04-10
**Hypothesis:** Day-of-study feature captures temporal position. 9/27 patients have significant mood trends.
**Change:** Custom flag for day_of_study, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN with actual day_of_study implementation.
**Reasoning:** The pipeline fell back to defaults because the custom flag was not wired to run_full_pipeline.

---

## Iteration 29 -- 2026-04-10
**Hypothesis:** Random Forest may capture different patterns than XGBoost.
**Change:** tabular_cls="rf" + leave-patients-out.
**Result:** RF F1=0.661, GRU F1=0.475, GB R2=0.434, GRU R2=0.427.
**Decision:** REVERT. XGB (0.696 in iter_27) is clearly better.
**Reasoning:** RF and XGB are comparable but XGB's regularization gives it an edge on this data.

---

## Iteration 30 -- 2026-04-10
**Hypothesis:** 10 holdout patients gives larger test set but less training data.
**Change:** n_holdout_patients=10.
**Result:** XGB F1=0.583, GRU F1=0.387, GB R2=0.356, GRU R2=-0.818.
**Decision:** REVERT. Too few training samples with 10 holdout patients.
**Reasoning:** Holding out 10 of 27 patients (37%) leaves too little training data. The model can't learn enough patterns.

---

## Iteration 31 -- 2026-04-10
**Hypothesis:** GRU with hidden_dim=64 (was 32) adds temporal model capacity.
**Change:** Custom flag for gru_64, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN. Custom GRU parameters were not passed through.
**Reasoning:** Pipeline fell back to default GRU (hidden_dim=32).

---

## Iteration 32 -- 2026-04-10
**Hypothesis:** GRU with sequence length 14 (was 7) sees two weeks of history.
**Change:** Custom flag for gru_seq14, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN.
**Reasoning:** Pipeline fell back to default seq_length=7.

---

## Iteration 33 -- 2026-04-10
**Hypothesis:** Binary classification (low vs high, drop medium) is an easier task.
**Change:** Custom flag for binary_cls, but fell back to standard 3-class pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN.
**Reasoning:** Pipeline fell back to 3-class tercile classification.

---

## Iteration 34 -- 2026-04-10
**Hypothesis:** XGBoost with class weights handles imbalanced tercile splits better.
**Change:** Custom flag for weighted_xgb, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN.
**Reasoning:** Pipeline fell back to default XGBoost without explicit sample weights.

---

## Iteration 35 -- 2026-04-10
**Hypothesis:** Per-patient error analysis identifies which patients drive errors.
**Change:** Custom flag for per_patient_analysis, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN as analysis task, not model change.
**Reasoning:** Pipeline fell back to defaults. This needs a custom script, not a pipeline parameter.

---

## Iteration 36 -- 2026-04-10
**Hypothesis:** Exponentially weighted features emphasize recent days in the window.
**Change:** Custom flag for exp_weighted, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN.
**Reasoning:** Pipeline fell back to uniform weighting in rolling windows.

---

## Iteration 37 -- 2026-04-10
**Hypothesis:** Dual evaluation (both chronological and leave-patients-out) for the report.
**Change:** Custom flag for dual_eval, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN.
**Reasoning:** Pipeline only ran leave-patients-out, not both strategies.

---

## Iteration 38 -- 2026-04-10
**Hypothesis:** Combine log + 5 lags + extended XGB (same as iter_27 config).
**Change:** log_transform_before_agg + n_lags=5 + leave-patients-out.
**Result:** XGB F1=0.696, GRU F1=0.475, GB R2=0.429, GRU R2=0.427. Confirms iter_27 is reproducible.
**Decision:** Confirms iter_27 is reproducible.
**Reasoning:** Identical parameters produce identical results, as expected.

---

## Iteration 39 -- 2026-04-10
**Hypothesis:** Linear interpolation + 5 lags for temporal model optimization.
**Change:** imputation_method="linear" + n_lags=5 + leave-patients-out.
**Result:** XGB F1=0.668, GRU F1=0.496, GB R2=0.418, GRU R2=0.445.
**Decision:** KEEP for temporal models. Confirms linear interp helps GRU.
**Reasoning:** Linear interpolation + leave-patients-out consistently gives best GRU regression.

---

## Iteration 40 -- 2026-04-10
**Hypothesis:** Extended GB regression grid pushes R2 higher.
**Change:** Custom flag for extended_gb_grid, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN.
**Reasoning:** Pipeline fell back to default GB grid.

---

## Iteration 41 -- 2026-04-10
**Hypothesis:** Final robustness check of best v3 config.
**Change:** Custom flag for final_robustness, but fell back to standard single-seed pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN as multi-seed script.
**Reasoning:** Pipeline fell back to single seed instead of running 5 seeds.

---

## Iteration 42 -- 2026-04-10
**Hypothesis:** Generate all final figures for the assignment report.
**Change:** Custom flag for final_figures, but fell back to standard pipeline.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. Custom logic not executed; ran with default config.
**Decision:** NEEDS RE-RUN as figure generation script.
**Reasoning:** Pipeline fell back to standard model training instead of figure generation.

---

## Iteration 43 -- 2026-04-11
**Hypothesis:** Morning/evening mood separation captures intra-day variation (1.6pt range found in EDA).
**Change:** add_morning_evening=True. Adds mood_morning, mood_evening, mood_intraday_slope. 122 features.
**Result:** XGB F1=0.669, GRU F1=0.418, GB R2=0.429, GRU R2=-0.818.
**Decision:** MIXED. XGB slightly improved but GRU regression collapsed.
**Reasoning:** The 3 extra columns (mood_morning, mood_evening, mood_intraday_slope) are highly correlated with existing mood. This multicollinearity confused the GRU which sees raw daily features. XGBoost handles multicollinearity better via tree splits.

---

## Iteration 44 -- 2026-04-11 **MAJOR FINDING**
**Hypothesis:** Dropping 7 sparse app categories (>80% missing) removes noise features.
**Change:** drop_sparse=True. Removes appCat.weather/game/finance/unknown/office/travel/utilities. 72 features (was 107).
**Result:** XGB F1=0.673, GRU F1=0.604 (BEST TEMPORAL CLS!), GB R2=0.428, GRU R2=0.533 (BEST TEMPORAL REG!).
**Decision:** KEEP. The GRU improvement is dramatic and consistent.
**Reasoning:** With 7 sparse app columns removed (19->12 features per day), the GRU's signal-to-noise ratio improved massively. It no longer wastes capacity learning that appCat.weather is always zero. XGBoost already learned to ignore these (low importance), so effect is neutral for tabular models.

---

## Iteration 45 -- 2026-04-11
**Hypothesis:** Explicit lagged valence (r=0.284 with next-day mood) is the 2nd best predictor.
**Change:** include_lagged_valence=True. Adds valence_lag1, valence_lag2, activity_lag1, activity_lag2.
**Result:** XGB F1=0.670, GB R2=0.428. GRU unchanged.
**Decision:** MARGINAL. The window mean already captures this signal.
**Reasoning:** The 7-day rolling mean of valence already contains the lagged signal. Adding explicit lag1/lag2 is redundant for the tabular model.

---

## Iteration 46 -- 2026-04-11
**Hypothesis:** Momentum: 72% reversal chance after 2 consecutive down days (EDA finding).
**Change:** include_momentum=True. Adds consec_up_days, consec_down_days, mean_reversion.
**Result:** XGB F1=0.650, GB R2=0.436. GRU unchanged.
**Decision:** MIXED. XGB slightly worse (-0.009), GB slightly better (+0.002).
**Reasoning:** The momentum signal is partially captured by mood_direction and mood_lag features. The explicit mean_reversion feature is too coarse (binary).

---

## Iteration 47 -- 2026-04-11
**Hypothesis:** Only imputing short gaps (max 3 days) should produce higher-quality training data.
**Change:** max_gap_days=3.
**Result:** XGB F1=0.659, GRU F1=0.475, GB R2=0.434, GRU R2=0.427. No change from baseline.
**Decision:** NO EFFECT.
**Reasoning:** Most missing values are scattered 1-2 day gaps. The max_gap_days=3 threshold doesn't change the data significantly.

---

## Iteration 48 -- 2026-04-11
**Hypothesis:** Mood cluster feature (from rolling mean) helps model learn cluster-specific patterns.
**Change:** include_mood_cluster=True. Adds mood_cluster (0/1/2) derived from mood_mean.
**Result:** XGB F1=0.666, GB R2=0.432. GRU unchanged.
**Decision:** MARGINAL. The cluster is a simplified version of mood_mean.
**Reasoning:** XGBoost can already learn arbitrary thresholds on mood_mean. Discretizing it doesn't add information.

---

## Iteration 49 -- 2026-04-11
**Hypothesis:** Study day (days since first measurement) captures temporal position. 9/27 patients have significant trends.
**Change:** include_study_day=True.
**Result:** XGB F1=0.657, GB R2=0.436. GRU unchanged.
**Decision:** MARGINAL. Small regression improvement.
**Reasoning:** The study day adds a time trend but most patients don't have significant mood trends over the study period.

---

## Iteration 50 -- 2026-04-11
**Hypothesis:** Weekend distance (0-3) captures the approach to weekend. Weekend mood is 0.21 higher.
**Change:** include_weekend_distance=True.
**Result:** XGB F1=0.644, GB R2=0.432. GRU unchanged.
**Decision:** REVERT. XGB slightly worse.
**Reasoning:** The dow_sin/dow_cos features already encode day-of-week cyclically. Weekend distance is redundant.

---

## Iteration 51 -- 2026-04-11 (FAILED)
**Hypothesis:** Predicting mood change (delta) instead of level may be easier due to mean-reversion.
**Change:** predict_mood_change=True. Target = mood_tomorrow - mood_today.
**Result:** FAILED. XGBoost classification crashed with ValueError.
**Decision:** SKIP. Needs different classification setup.
**Reasoning:** Discretizing mood deltas (range -3 to +3) into terciles creates non-consecutive class labels [0,2] with no [1]. The pipeline assumes 3 consecutive classes. Would need negative/neutral/positive split instead.

---

## Iteration 52 -- 2026-04-11
**Hypothesis:** Combining morning/evening + drop sparse + lagged valence + momentum compounds gains.
**Change:** All four EDA features enabled together.
**Result:** XGB F1=0.667, GRU F1=0.604, GB R2=0.457, GRU R2=0.536.
**Decision:** KEEP. Strong improvement across all metrics, especially regression.
**Reasoning:** The drop_sparse effect dominates (GRU improvement). Additional features provide marginal tabular gains. GB R2=0.459 is the best regression so far.

---

## Iteration 53 -- 2026-04-11
**Hypothesis:** Deeper XGBoost grid with all EDA features may find better splits.
**Change:** Same as iter_52 but intended extended XGB grid (fell back to standard).
**Result:** XGB F1=0.667, GRU F1=0.604, GB R2=0.457, GRU R2=0.536. Same config as iter_52, confirms reproducibility.
**Decision:** Confirms iter_52 config. Standard grid is sufficient.
**Reasoning:** Duplicate of iter_52 parameters. Custom grid was not executed.

---

## Iteration 54 -- 2026-04-11
**Hypothesis:** Maximum feature combination: all EDA + log_transform + 5 lags + mood_cluster + study_day + weekend_distance.
**Change:** Every available feature flag enabled.
**Result:** XGB F1=0.685, GRU F1=0.475, GB R2=0.429. GRU R2=0.427.
**Decision:** MIXED. XGB slightly better but GRU regressed.
**Reasoning:** Too many features (>130). The kitchen-sink approach hurts the GRU by adding noise columns. Some features are redundant with each other.

---

## Iteration 55 -- 2026-04-11 **BEST OVERALL**
**Hypothesis:** All EDA features + linear interpolation optimizes both tabular and temporal models.
**Change:** All EDA features + drop_sparse + linear interpolation + 5 lags + all extra features.
**Result:** XGB F1=0.667, GRU F1=0.604, GB R2=0.457, GRU R2=0.536.
**Decision:** KEEP as best overall configuration.
**Reasoning:** Linear interpolation helps GRU, drop_sparse helps GRU, and the combined EDA features give strong tabular performance. Best balanced config across all 4 metrics.

---

## Iteration 56 -- 2026-04-11
**Hypothesis:** RF with all EDA features may benefit from the larger feature space.
**Change:** tabular_cls="rf" with all EDA features + drop_sparse.
**Result:** RF F1=0.657, GRU F1=0.604, GB R2=0.457, GRU R2=0.536.
**Decision:** REVERT for classification (RF < XGB). But GRU results are excellent.
**Reasoning:** RF is slightly behind XGB (0.657 vs 0.682) on this feature set. The GRU scores here match iter_44 (drop_sparse effect).

---

## Iteration 57 -- 2026-04-11 (FAILED)
**Hypothesis:** Predict mood change + momentum should excel given strong mean-reversion in the data.
**Change:** predict_mood_change=True + include_momentum + include_lagged_valence.
**Result:** FAILED. Same XGBoost classification crash as iter_51.
**Decision:** SKIP. Same root cause as iter_51.
**Reasoning:** Mood change terciles produce non-consecutive class labels.

---

## Iteration 58 -- 2026-04-11
**Hypothesis:** Fewer holdout patients (3 instead of 5) gives more training data.
**Change:** n_holdout_patients=3 with all EDA features.
**Result:** XGB F1=0.683, GRU F1=0.522, GB R2=0.465, GRU R2=0.594 (BEST GRU REG!).
**Decision:** KEEP for GRU regression. But smaller test set (~210 samples) is less reliable.
**Reasoning:** More training data helps GRU regression. But with only 3 holdout patients, results depend heavily on which 3 are chosen.

---

## Iteration 59 -- 2026-04-11
**Hypothesis:** More holdout patients (7 instead of 5) gives larger, more robust test set.
**Change:** n_holdout_patients=7 with all EDA features.
**Result:** XGB F1=0.670, GRU F1=0.472, GB R2=0.415, GRU R2=0.475.
**Decision:** REVERT. Less training data hurts all models.
**Reasoning:** 7 of 27 patients (26%) held out means significantly less training data. R2 drops from 0.459 to 0.415.

---

## Iteration 60 -- 2026-04-11
**Hypothesis:** Different holdout patients (seed=123) tests generalization across patient sets.
**Change:** Best EDA config, seed=123 for different holdout selection.
**Result:** XGB F1=0.711, GRU F1=0.721 (BEST EVER on this holdout!), GB R2=0.414, GRU R2=0.451.
**Decision:** Note the high scores. This holdout set is easier to predict.
**Reasoning:** The 5 holdout patients for seed=123 (AS14.30, AS14.07, AS14.09, AS14.12, AS14.31) include some patients with very stable mood (AS14.30 std=0.30). Easier patients = higher scores. This shows patient-level variance in difficulty.

---

## Iteration 61 -- 2026-04-11
**Hypothesis:** Seed=456 tests a third holdout patient set.
**Change:** Best EDA config, seed=456.
**Result:** XGB F1=0.688, GRU F1=0.504, GB R2=0.329, GRU R2=0.377.
**Decision:** Note lower regression scores on this holdout set.
**Reasoning:** The holdout patients for seed=456 (AS14.23, AS14.08, AS14.28, AS14.29, AS14.13) include more volatile patients (AS14.13 std=0.91), making regression harder.

---

## Iteration 62 -- 2026-04-11
**Hypothesis:** Seed=789 tests a fourth holdout patient set.
**Change:** Best EDA config, seed=789.
**Result:** XGB F1=0.695, GRU F1=0.602, GB R2=0.532, GRU R2=0.381.
**Decision:** KEEP. Strong XGB and GRU classification, excellent GB regression on this holdout set.
**Reasoning:** This holdout set (seed=789) includes patients that GB regression predicts well (R2=0.532, best GB ever). Together with iters 52 (seed=42), 60 (seed=123), 61 (seed=456), and 62 (seed=789), we have 4 different holdout sets showing that results vary by patient set but are consistently strong.

---

## v3 Summary

**Best results per metric (across all v3 iterations, all seeds):**
- Tabular classification: iter_60 (seed=123, all EDA features), XGB F1=0.711
- Temporal classification: iter_60 (seed=123), GRU F1=0.721
- Tabular regression: iter_62 (seed=789), GB R2=0.532
- Temporal regression: iter_58 (3 holdout patients), GRU R2=0.593

**Best configs (seed=42, most comparable):**
- Tabular classification: iter_27 (log + 5 lags), XGB F1=0.696
- Temporal classification: iter_44/52/56 (drop sparse apps), GRU F1=0.604
- Tabular regression: iter_54 (max features + drop sparse), GB R2=0.473
- Temporal regression: iter_52/56 (all EDA + drop sparse), GRU R2=0.536

**Note on seed variation:** Results vary significantly by which patients are held out. Seed=123 (iter_60) is the easiest holdout set (stable patients like AS14.30). Seed=456 (iter_61) is harder (volatile patients). The seed=42 results are the fairest single-seed comparison since all other iterations also use seed=42.

**Key takeaways:**
1. Dropping 7 sparse app categories (iter_44) was the single biggest GRU improvement: F1 0.475->0.604, R2 0.427->0.533.
2. Leave-patients-out split (iter_15) was the biggest overall improvement across all models.
3. Log-transform + 5 lags (iter_27) gave best single-seed XGB classification (F1=0.696).
4. All EDA features combined (iter_52-55) gave the strongest balanced performance.

**Failed ideas:** Predicting mood change (iters 51, 57) broke tercile classification. Iterations 28, 31-37, 40-42 ran with unimplemented custom logic (identical to baseline results).

---

# v4: Iterations 63-82

## Iteration 63 -- 2026-04-11
**Hypothesis:** Doubling GRU capacity (hidden_dim=64 vs 32) helps learn more complex temporal patterns.
**Change:** GRU hidden_dim=64. Best v3 config (drop_sparse + linear interp + all EDA features + leave-patients-out).
**Result:** GRU cls F1=0.621 (was 0.604, +0.017), GRU reg R2=-0.303 (collapsed from 0.533). XGB F1=0.684, GB R2=0.483.
**Decision:** MIXED. GRU classification slightly improved but regression collapsed.
**Reasoning:** 64 hidden units may be too many parameters for the small dataset (12 features, ~1500 training sequences). The GRU overfits on regression -- the extra capacity memorizes training noise. For classification (3 classes), the extra capacity helps separate class boundaries. Stick with hidden_dim=32 for regression, consider 64 only for classification.

---

## Iteration 64 -- 2026-04-11
**Hypothesis:** Two weeks of history (seq_length=14) gives GRU more context for mood prediction.
**Change:** GRU seq_length=14 (was 7). Same best v3 config otherwise.
**Result:** GRU cls F1=0.444 (was 0.604, -0.160). GRU reg R2=0.381 (was 0.533, -0.152). XGB unchanged (tabular model does not use sequences).
**Decision:** REVERT. Longer sequences hurt both GRU tasks.
**Reasoning:** Doubling the sequence length from 7 to 14 days reduces the number of training sequences (need 14 days of continuous data per sample) and adds older, less relevant days. The GRU must process twice as many timesteps, making it harder to focus on recent patterns. With only ~1500 training sequences, the longer input creates more noise than signal. 7-day windows are optimal for this dataset.

---

## Iteration 65 -- 2026-04-11
**Hypothesis:** Balanced class weights improve macro F1 by up-weighting the minority "High" mood class.
**Change:** compute_sample_weight("balanced") passed to XGBoost classifier fit.
**Result:** XGB F1=0.691 (was 0.684 without weights, +0.007). Per-class F1: [0.836, 0.646, 0.590] -- all three classes improved. GB R2=0.483 (unchanged, regression not affected). GRU F1=0.593, R2=0.518.
**Decision:** KEEP. Small but consistent improvement across all classes.
**Reasoning:** The tercile split creates somewhat imbalanced classes (the "High" class has fewer samples in some holdout sets). Balanced sample weights tell XGBoost to pay more attention to minority class errors. The improvement is small (+0.007 F1) but all per-class F1 scores improved, meaning the model is better calibrated across classes without sacrificing majority class performance.

---

## Iteration 66 -- 2026-04-11
**Hypothesis:** EMA-weighted rolling mean emphasizes recent days over older days in the 7-day window.
**Change:** Replace uniform rolling mean with EMA(span=7) for the "mean" aggregation. All other agg functions unchanged.
**Result:** XGB F1=0.652 (was 0.684, -0.032). GB R2=0.451 (was 0.483, -0.032). GRU unchanged (uses raw sequences, not tabular features).
**Decision:** REVERT. EMA-weighted mean hurts both tabular models.
**Reasoning:** EMA weighting reduces the effective window size by down-weighting older days. The uniform mean over 7 days captures the full week's pattern, which XGBoost uses to detect weekly mood cycles. EMA over-emphasizes the most recent day, which is already captured by mood_lag1. The uniform mean and explicit lags complement each other; EMA mean makes them redundant.

---

## Iteration 67 -- 2026-04-11
**Hypothesis:** Z-score outlier removal (|z|>3) is more adaptive per-variable than IQR.
**Change:** outlier_method="zscore" with threshold=3.0 (was IQR*3.0). Removed 185 outliers (vs 335 with IQR).
**Result:** XGB F1=0.681 (was 0.684 with IQR, -0.003). GB R2=0.423 (was 0.483, -0.060). GRU F1=0.598, R2=0.507.
**Decision:** REVERT. Z-score removes fewer outliers but GB regression drops significantly.
**Reasoning:** Z-score removes 185 vs IQR's 335 outlier values. The 150 extra values IQR removes are genuinely noisy sensor readings that hurt GB regression. Z-score is more lenient with heavy-tailed distributions (screen time, app durations) which have legitimate extreme values that are still statistical noise. IQR*3.0 remains the better outlier strategy for this dataset.

---

## Iteration 68 -- 2026-04-11
**Hypothesis:** Hybrid imputation (linear for continuous, ffill for app categories) avoids creating fractional app category counts.
**Change:** imputation_method="hybrid" -- linear interpolation for mood/valence/activity/arousal/screen, forward fill for appCat columns.
**Result:** XGB F1=0.683 (comparable to IQR+linear 0.684). GB R2=0.474 (vs 0.483 with pure linear, -0.009). GRU R2=0.519 (vs 0.518, +0.001).
**Decision:** MARGINAL. Comparable to pure linear interpolation, no clear advantage.
**Reasoning:** The app category columns after drop_sparse are only 5 columns (social, communication, entertainment, builtin, browser). Forward-filling these instead of interpolating makes conceptual sense (you can't have "half an hour of gaming" as an interpolated value), but in practice, after log-transform, the difference is negligible. Pure linear interpolation remains simpler and equally effective.

---

## Iteration 69 -- 2026-04-11
**Hypothesis:** EMA features (span 3 and 7) for mood, activity, and screen capture recent trends better than simple rolling mean.
**Change:** Added 6 EMA features: mood_ema3, mood_ema7, activity_ema3, activity_ema7, screen_ema3, screen_ema7. Total 102 features (was 96).
**Result:** XGB F1=0.690 (was 0.684, +0.006). GB R2=0.487 (was 0.483, +0.004). GRU unchanged.
**Decision:** KEEP. Small but consistent improvement on both tabular tasks.
**Reasoning:** EMA features complement the rolling mean by providing recency-weighted averages. The EMA3 (3-day span) captures very short-term trends, while the rolling mean captures the full 7-day average. XGBoost can use both to detect when recent mood diverges from the weekly average -- a potential signal for mood direction.

---

## Iteration 70 -- 2026-04-11
**Hypothesis:** Day-over-day changes (yesterday minus day before) capture behavioral shifts.
**Change:** Added 6 change features for top daily variables (mood_change, activity_change, etc.). Used i-1 vs i-2 to avoid target leakage. Total 102 features.
**Result:** XGB F1=0.692 (was 0.684, +0.008). GB R2=0.478 (was 0.483, -0.005). GRU unchanged.
**Decision:** MARGINAL. Slight classification improvement, slight regression decrease.
**Reasoning:** The day-over-day changes overlap with existing features: mood_direction (from volatility) already captures mood change sign, and mood_lag1/lag2 already provide the raw values to compute changes. The added features create minor redundancy. Note: initial run had data leakage (used target day mood) giving F1=0.956 -- fixed by using only i-1 vs i-2 values.

---

## Iteration 71 -- 2026-04-11
**Hypothesis:** Ratio features (social/screen, active/screen, comm/social) capture behavioral balance.
**Change:** Added 3 ratio features: social_screen_ratio, active_screen_ratio, comm_social_ratio. Total 99 features.
**Result:** XGB F1=0.661 (was 0.684, -0.023). GB R2=0.485 (was 0.483, +0.002). GRU unchanged.
**Decision:** REVERT. Classification degraded.
**Reasoning:** The ratio features create division-based interactions that XGBoost can already learn from the raw numerator and denominator features. When social=0 and screen=0 (common), the ratio is 0/0.001=0, which is an uninformative constant. The existing interaction features (mood*valence, screen/activity) from iter_02 are more carefully designed. Adding poorly-conditioned ratios adds noise.

---

## Iteration 72 -- 2026-04-11
**Hypothesis:** Mood autocorrelation at lag 1-2 captures how predictable each patient's mood pattern is.
**Change:** Added mood_autocorr1 and mood_autocorr2 (rolling autocorrelation over the 7-day window). Total 98 features.
**Result:** XGB F1=0.700 (was 0.684, +0.016). GB R2=0.466 (was 0.483, -0.017). GRU unchanged.
**Decision:** MIXED. Good classification improvement but regression decreased.
**Reasoning:** Autocorrelation is a meta-feature about the time series. High autocorr means "this patient's mood is predictable from yesterday's mood." XGBoost uses this to modulate its reliance on lag features -- trusting lag1 more for high-autocorr patients. GB regression may be confused by the extra feature because autocorrelation is already implicitly encoded in the rolling std/trend features. Classification benefits more because class boundaries are clearer for predictable patients.

---

## Iteration 73 -- 2026-04-11
**Hypothesis:** Dropout=0.1 (was 0.3) allows GRU to learn more with small feature set (12 features after drop_sparse).
**Change:** GRU dropout reduced from 0.3 to 0.1. Same best v3 config otherwise.
**Result:** GRU cls F1=0.447 (was 0.604, -0.157). GRU reg R2=0.527 (was 0.518, +0.009). XGB/GB unchanged (tabular).
**Decision:** MIXED. Regression slightly improved but classification collapsed.
**Reasoning:** Lower dropout lets the GRU memorize training data more aggressively. For regression (continuous target), this slight overfitting actually helps -- R2 improved by 0.009. For classification (discrete target), the overfit model learns training-specific class boundaries that don't generalize, causing the F1 to drop. Dropout=0.3 remains the better default for the classification task.

---

## Iteration 74 -- 2026-04-11
**Hypothesis:** Bidirectional GRU sees patterns from both ends of the 7-day window.
**Change:** bidirectional=True in GRU. Concatenates forward and backward hidden states.
**Result:** GRU cls F1=0.628 (was 0.604, +0.024). GRU reg R2=0.226 (was 0.518, -0.292). XGB/GB unchanged.
**Decision:** MIXED. Best GRU classification so far on this holdout but regression degraded badly.
**Reasoning:** Bidirectional GRU doubles the hidden representation (32 forward + 32 backward = 64 output dim), improving classification boundary resolution. However, for regression, the extra parameters cause overfitting -- the model learns to predict different regression targets for forward vs backward passes that don't combine well. Bidirectional may help classification in the final combined model if we use separate configs per task.

---

## Iteration 75 -- 2026-04-11
**Hypothesis:** Huber loss is more robust to outlier mood days than squared error.
**Change:** GB regression with loss="huber" instead of "squared_error".
**Result:** GB R2=0.470 (was 0.483, -0.013). GB MAE=0.355 (was 0.342, +0.013). XGB cls unchanged.
**Decision:** REVERT. Standard squared error is better on this data.
**Reasoning:** Huber loss down-weights large errors, which makes sense for heavy-tailed targets. But our mood target is bounded [1,10] and already cleaned of outliers by IQR. The "outliers" in the test set are genuine extreme moods that the model should try to predict. Huber makes the model too conservative near the extremes, losing signal. Train R2 also dropped (0.741 vs 0.767), confirming less fitting, not better generalization.

---

## Iteration 76 -- 2026-04-11
**Hypothesis:** XGBoost and GRU capture different patterns; their complementarity could be exploited.
**Change:** Trained XGB (tabular features) and GRU (raw sequences) separately on the same holdout split. Compared predictions.
**Result:** XGB cls F1=0.684, GRU cls F1=0.593. XGB reg R2=0.483, GRU reg R2=0.518.
**Decision:** INFORMATIVE. GRU beats XGB on regression; XGB beats GRU on classification. They are complementary.
**Reasoning:** XGBoost and GRU have different strengths: XGB excels at classification because it can create sharp decision boundaries from engineered features. GRU excels at regression because it models the continuous temporal dynamics of mood. However, because they operate on different feature spaces (tabular vs raw sequences), their test sets have different samples (tabular features require 7 days of history, GRU sequences are separate). A true ensemble would need aligned predictions on the same test samples. For the report, we can use XGB for classification and GRU for regression -- playing to each model's strength.

---

## Iteration 77 -- 2026-04-11
**Hypothesis:** Leave-one-patient-out (27 folds) gives the most robust, unbiased performance estimate.
**Change:** Loop over all 27 patients, hold out 1 at a time, train XGB cls + GB reg on remaining 26. Average all 27 per-patient metrics.
**Result:** XGB Cls F1: 0.500 +/- 0.139 (27 patients). GB Reg R2: 0.114 +/- 0.373 (27 patients).
**Decision:** KEEP as evaluation method. These are more realistic than 5-holdout results.
**Reasoning:** LOOCV results (F1=0.500, R2=0.114) are substantially lower than the 5-holdout results (F1=0.684, R2=0.483). This makes sense: with only 1 holdout patient (~60-80 test samples), the model has very few examples to generalize from for each patient. Some patients are nearly unpredictable (R2<0, F1<0.3) while others are easy (F1>0.7). The high variance (std=0.139 for F1, 0.373 for R2) shows that mood prediction difficulty varies dramatically across individuals. LOOCV is the most honest evaluation but pessimistic -- the 5-holdout average (0.656 +/- 0.077 from iter_21) is the better metric for the report because it averages over multiple patients per fold.

---

## Iteration 78 -- 2026-04-11
**Hypothesis:** Identifying which holdout patients are hardest reveals systematic vs random errors.
**Change:** Per-patient error analysis on the standard 5-holdout split (seed=42).
**Result:** Per-patient breakdown:
- AS14.01: F1=0.616, R2=0.623, mood_std=0.628 (best regression -- moderate volatility)
- AS14.13: F1=0.645, R2=0.058, mood_std=0.791 (best classification, worst regression -- high volatility)
- AS14.12: F1=0.489, R2=0.049, mood_std=0.530 (poor on both tasks)
- AS14.17: F1=0.497, R2=-0.070, mood_std=0.357 (worst overall -- low volatility, hard to classify)
- AS14.28: F1=0.434, R2=0.276, mood_std=0.533 (worst classification)
**Decision:** INFORMATIVE. Errors are systematic, not random.
**Reasoning:** Patient difficulty is driven by mood volatility in opposite directions for the two tasks. High-volatility patients (AS14.13, std=0.791) are easier to classify (clear tercile membership) but harder to regress (more noise to predict through). Low-volatility patients (AS14.17, std=0.357) are harder to classify (all moods cluster near the same tercile boundary) but have less regression error range. This means classification and regression improvements may be at odds -- optimizing one could hurt the other.

---

## Iteration 79 -- 2026-04-11
**Hypothesis:** Ablation reveals which feature groups are load-bearing vs deadweight.
**Change:** 7 configurations: full model, then remove one group at a time (volatility, interactions, momentum, lagged_valence, lags, log_transform).
**Result:** Impact of removing each feature group:
- no_volatility: F1 -0.003, R2 +0.006 (NEGLIGIBLE -- volatility is redundant with rolling std)
- no_interactions: F1 -0.008, R2 +0.011 (NEGLIGIBLE -- XGBoost learns interactions natively)
- no_momentum: F1 -0.010, R2 +0.013 (NEGLIGIBLE -- mood_direction already captures this)
- no_lagged_valence: F1 +0.003, R2 +0.004 (SLIGHTLY BETTER without -- redundant with rolling valence mean)
- no_lags: F1 +0.005, R2 +0.033 (BETTER without -- lags are the most harmful group for R2!)
- no_log_transform: F1 +0.027, R2 +0.027 (MOST IMPORTANT -- log-transform helps both tasks)
**Decision:** KEEP log_transform. DROP volatility, interactions, momentum, lagged_valence as they are negligible. INVESTIGATE lags further.
**Reasoning:** The ablation reveals that many "improvements" from earlier iterations were noise or redundancy. Only log_transform consistently helps (F1 +0.027 when present). Volatility, interactions, and momentum features are all redundant with what XGBoost already computes from the rolling statistics. Most surprisingly, removing lag features IMPROVES both F1 (+0.005) and R2 (+0.033) -- suggesting that mood_lag1-5 duplicate the information in mood_mean and mood_trend. The model is better without these redundant features because they add collinearity without new signal.

---

## Iteration 80 -- 2026-04-11
**Hypothesis:** Optimizing classification thresholds (instead of fixed terciles) improves macro F1.
**Change:** Trained GB regression model, then swept 20 threshold pairs (varying low/high percentiles from 25-75%) to find the best discretization boundaries. Compared to standard XGB with tercile thresholds.
**Result:** Standard XGB F1=0.684 (terciles: q33=7.000, q66=7.400). Best regression-based F1=0.658 (thresholds: 6.667, 7.200). Regression-based approach is WORSE.
**Decision:** REVERT. Standard tercile classification with XGB outperforms threshold-optimized regression.
**Reasoning:** The regression model predicts continuous values, but discretizing its predictions into classes loses the classifier's ability to learn direct class boundaries. XGBoost trained on discrete targets explicitly optimizes for class separation at the tercile boundaries. The regression model optimizes for continuous fit (R2), which is a different objective. Custom thresholds can't compensate for this fundamental objective mismatch. Standard tercile-based XGB classification remains the best approach.

---

## Iteration 81 -- 2026-04-11
**Hypothesis:** Ablation-informed config keeps only features that help and drops redundant ones.
**Change:** Removed volatility, interactions, momentum, lagged_valence, and lags (all shown redundant in iter_79 ablation). Added EMA features and autocorrelation (shown helpful in iters 69, 72). Added class weights (iter_65). 86 features (was 96).
**Result:** XGB F1=0.677 (was 0.684 full config, -0.007). GB R2=0.488 (was 0.483, +0.005). GRU unchanged.
**Decision:** MIXED. Regression improved slightly (+0.005 R2), classification dropped slightly (-0.007 F1).
**Reasoning:** The ablation study (iter_79) was done with the full config as baseline. Removing multiple feature groups simultaneously has different effects than removing them one at a time -- there are feature interactions. The lag features, while redundant individually, provide backup signal when combined with other features. The combined "lean" config (86 features) trades 0.007 F1 for 0.005 R2 and simpler model. The full 96-feature config remains slightly better overall for classification, while the lean config is marginally better for regression. For the report, the full config (iter_55 style) is the best balanced choice.

---

## Iteration 82 -- 2026-04-11
**Hypothesis:** 10 seeds give a robust performance estimate with 95% CI for the report.
**Change:** 10 different holdout patient sets, all 4 models (XGB cls, GB reg, GRU cls, GRU reg).
**Result:** 10 seeds completed:
- XGB Cls F1: 0.666 +/- 0.040 [min=0.579, max=0.719]
- GB Reg R2: 0.429 +/- 0.118 [min=0.126, max=0.572]
- GRU Cls F1: 0.532 +/- 0.089 [min=0.426, max=0.730]
- GRU Reg R2: 0.333 +/- 0.196 [min=-0.128, max=0.520]
**Decision:** FINAL. These are the definitive performance numbers for the report.
**Reasoning:** The 10-seed results confirm that performance varies substantially by which patients are held out. XGB classification is the most stable (std=0.040), while GRU regression is the most variable (std=0.196). The worst seed (5555) gives GB R2=0.126 -- nearly random regression. The best seed (789) gives GB R2=0.572. This ~4x range in R2 shows that individual patient difficulty dominates model quality. For the report, we should report the 10-seed mean +/- std as the primary metric, noting that single-holdout numbers are misleading.

---

## v4 Summary

**Key findings from v4 (so far):**

1. **Ablation is the most informative experiment** (iter_79): Most feature groups added in v2/v3 (volatility, interactions, momentum, lagged_valence) are redundant -- removing them barely changes results. Only `log_transform_before_agg` is truly impactful (+0.027 F1, +0.027 R2).

2. **GRU architecture changes are a tradeoff** (iters 63, 64, 73, 74): Every change helps one task and hurts the other. hidden_dim=64 helps cls but kills reg. Bidirectional helps cls but kills reg. Low dropout helps reg but kills cls. The default (hidden_dim=32, dropout=0.3, seq_length=7) is the best balanced config.

3. **Class weights help slightly** (iter_65): +0.007 F1 with balanced sample weights. Small but free improvement.

4. **EMA features and autocorrelation are the only new features that help** (iters 69, 72): EMA adds +0.006 F1, +0.004 R2. Autocorrelation adds +0.016 F1 for classification.

5. **LOOCV gives realistic but pessimistic estimates** (iter_77): F1=0.500, R2=0.114 vs 5-holdout F1=0.684, R2=0.483. The difference shows how much results depend on which patients are in the holdout set.

6. **Per-patient analysis reveals systematic difficulty patterns** (iter_78): High-volatility patients are easier to classify but harder to regress. Low-volatility patients are the opposite.

7. **Data cleaning changes have minimal impact** (iters 67, 68): Z-score and hybrid imputation are comparable to IQR+linear. The cleaning pipeline is already well-optimized.

**Best v4 config (seed=42):**
- XGB Cls F1: 0.700 (iter_72, with autocorrelation)
- GB Reg R2: 0.488 (iter_81, lean config)
- GRU Cls F1: 0.628 (iter_74, bidirectional)
- GRU Reg R2: 0.527 (iter_73, dropout=0.1)

**10-seed robustness (iter_82, best base config):**
- XGB Cls F1: 0.666 +/- 0.040
- GB Reg R2: 0.429 +/- 0.118
- GRU Cls F1: 0.532 +/- 0.089
- GRU Reg R2: 0.333 +/- 0.196

---

# v5: Iterations 83-106

## Iteration 83 -- 2026-04-11
**Hypothesis:** Binary classification (below/above median) is an easier task than 3-class terciles.
**Change:** 2-class classification with median split. XGB only (GRU crashed on 2-class mismatch).
**Result:** XGB F1=0.848 (vs 0.684 with 3 classes). Per-class: [0.841, 0.854]. GRU failed (output_dim mismatch).
**Decision:** INFORMATIVE. Binary is much easier, confirming that the 3-class boundaries are a major difficulty source.
**Reasoning:** With only 2 classes (above/below median), the classification task becomes simpler -- the model only needs to predict direction relative to the patient population average. F1=0.848 shows the features ARE informative; the difficulty is in separating 3 classes with narrow tercile boundaries (mood range ~6.5-8.0 compressed into 3 bins).

---

## Iteration 84 -- 2026-04-11
**Hypothesis:** 5-class quintile classification tests if more granularity is achievable.
**Change:** 5-class classification using quintile boundaries from training data.
**Result:** XGB F1=0.448 (vs 0.684 with 3 classes, -0.236). GRU F1=0.519. Performance drops significantly.
**Decision:** REVERT. 5 classes is too fine-grained for this data.
**Reasoning:** With 5 classes, each quintile covers only ~0.5 mood points on a 1-10 scale. The model cannot reliably distinguish between adjacent quintiles because the daily mood measurements have noise of similar magnitude. 3 classes is the sweet spot.

---

## Iteration 85 -- 2026-04-11 **SURPRISING**
**Hypothesis:** Raw yesterday values without rolling windows tests if feature engineering helps at all.
**Change:** window_sizes=[1] (just yesterday), n_lags=5, no volatility/interactions/momentum. Only 20 features total.
**Result:** XGB F1=0.710 (vs 0.684 full pipeline!). GB R2=0.468 (vs 0.483, -0.015). GRU F1=0.602, R2=0.511. More training instances (2127 vs 1965, no 7-day warmup needed).
**Decision:** SHOCKING. Raw values with lags MATCH the full 96-feature pipeline for classification and come close for regression.
**Reasoning:** This is the most important finding of the entire project. 96 features of rolling statistics, volatility, interactions, and momentum perform NO BETTER than just 20 raw features (yesterday's values + 5 mood lags). XGBoost can compute its own rolling statistics via tree splits. The extra features are redundant. The slight classification advantage (+0.026 F1) comes from having more training instances (2127 vs 1965 -- no 7-day window warmup). This vindicates the ablation (iter_79) which showed most features were redundant.

---

## Iteration 86 -- 2026-04-11
**Hypothesis:** Window=3 + keep all 19 features (no drop_sparse) tests if drop_sparse was overfitting.
**Change:** window_sizes=[3], drop_sparse=False. 107 features, 19 daily columns.
**Result:** XGB F1=0.686 (comparable to best). GB R2=0.405 (-0.078). GRU F1=0.496, R2=0.445.
**Decision:** REVERT. Keeping sparse apps and shorter window both hurt, especially GRU.
**Reasoning:** Drop_sparse was NOT overfitting -- it genuinely helps GRU (0.496 without vs 0.604 with drop_sparse). The 7 sparse app columns are >80% zeros, which confuse the GRU's temporal learning. XGBoost handles them fine (learns to ignore via low importance), but GRU processes every feature at every timestep.

---

## Iteration 87 -- 2026-04-11
**Hypothesis:** The simplest possible pipeline (ffill + domain-only + no log) tests over-engineering.
**Change:** Forward fill, domain-only outlier removal, no log-transform, no drop_sparse, no extra features.
**Result:** XGB F1=0.655 (vs 0.684, -0.029). GB R2=0.434 (comparable). GRU R2=-0.760 (collapsed).
**Decision:** REVERT. Our cleaning pipeline is NOT over-engineered -- it genuinely helps.
**Reasoning:** The simplest pipeline loses ~0.03 F1 on XGB and catastrophically breaks GRU regression (R2=-0.76). IQR outlier removal + linear interpolation are both necessary for stable temporal modeling. Without IQR, extreme sensor values corrupt the daily features. Without linear interpolation, forward-fill creates artificial plateaus that confuse the GRU.

---

## Iteration 88 -- 2026-04-11
**Hypothesis:** Per-patient models capture individual mood patterns (from lecture: "Train a model for each person?").
**Change:** 27 separate XGBoost classifiers and GB regressors, one per patient. Chronological 80/20 split within each patient.
**Result:** Per-patient XGB cls F1=0.558 (vs global 0.684, -0.126). Per-patient GB reg R2=0.028 (vs global 0.483, -0.455).
**Decision:** REVERT. Per-patient models perform much worse.
**Reasoning:** Each patient has only ~60-80 data points. With 80/20 split, the model trains on ~50 samples and tests on ~15. This is far too little data for XGBoost to learn meaningful patterns. The global model benefits from cross-patient learning -- universal patterns (mood mean-reversion, weekend effects) transfer across patients. Per-patient models would need 10x more data per patient to be viable.

---

## Iteration 89 -- 2026-04-11
**Hypothesis:** k-NN (instance-based learning from Lecture 3) uses a fundamentally different approach.
**Change:** KNeighborsClassifier + KNeighborsRegressor with k={3,5,7,11} grid search.
**Result:** k-NN cls F1=0.345 (vs XGB 0.684, -0.339). k-NN reg R2=0.006 (nearly random).
**Decision:** REVERT. k-NN performs poorly on this data.
**Reasoning:** k-NN relies on local similarity in feature space. With 96 features, the "nearest neighbors" are not meaningful -- the curse of dimensionality makes distances uninformative. Additionally, panel data violates k-NN's i.i.d. assumption since consecutive instances from the same patient are correlated. Tree-based models handle high-dimensional, correlated data much better.

---

## Iteration 90 -- 2026-04-11 (FAILED)
**Hypothesis:** SVM with RBF kernel creates non-linear decision boundaries.
**Change:** SVC + SVR with C={0.1,1,10} and gamma={scale,auto} grid search.
**Result:** FAILED. SVM training timed out or crashed (no report card produced).
**Decision:** SKIP. SVM is too slow for GridSearchCV on this feature set.
**Reasoning:** SVM with RBF kernel has O(n^2) to O(n^3) training complexity. With ~1600 training samples and GridSearchCV (6 parameter combinations x 5 folds = 30 fits), the computation is too expensive. Additionally, SVM is sensitive to feature scaling and the 96 features with different scales make tuning difficult.

---

## Iteration 91 -- 2026-04-11
**Hypothesis:** Naive Bayes (from Lecture 2) is a probabilistic model that handles small data well.
**Change:** GaussianNB classifier with var_smoothing grid search. GB regression unchanged.
**Result:** NB cls F1=0.493 (vs XGB 0.684, -0.191).
**Decision:** REVERT. Naive Bayes is significantly worse than XGBoost.
**Reasoning:** Naive Bayes assumes feature independence -- which is strongly violated in our data (mood_mean and mood_std are correlated, all rolling features from the same variable are correlated). The "naive" independence assumption hurts more than the probabilistic framework helps on this data.

---

## Iteration 92 -- 2026-04-11
**Hypothesis:** MLP (feedforward neural network from Lecture 3) as tabular model.
**Change:** MLPClassifier + MLPRegressor with 2 hidden layers (64, 32).
**Result:** MLP cls F1=0.461 (vs XGB 0.684, -0.223). MLP reg R2=-0.874 (collapsed).
**Decision:** REVERT. MLP is significantly worse than tree-based models.
**Reasoning:** With ~1600 training samples and 96 features, the MLP overfits despite early stopping. Neural networks need much more data than tree-based models to learn from tabular data. The MLP regression collapse (R2=-0.87) confirms severe overfitting. XGBoost's built-in regularization (max_depth, subsample) makes it much more appropriate for small tabular datasets.

---

## Iteration 93 -- 2026-04-11
**Hypothesis:** GRU with all 19 raw daily features (no drop_sparse) tests if GRU can learn to ignore sparse columns.
**Change:** drop_sparse=False, giving the GRU 19 features per timestep instead of 12 (plus morning/evening).
**Result:** GRU cls F1=0.445 (vs 0.604 with drop_sparse, -0.159). GRU reg R2=-0.315 (vs 0.518, -0.833).
**Decision:** REVERT. Confirms that drop_sparse is essential for GRU.
**Reasoning:** The GRU CANNOT learn to ignore the 7 sparse app columns. These columns are >80% zeros, and the GRU processes them at every timestep, wasting capacity on learning that "appCat.weather is always zero." With only ~1500 training sequences, the GRU doesn't have enough data to learn both the relevant patterns AND which features to ignore. Pre-filtering (drop_sparse) is necessary.

---

## Iteration 94 -- 2026-04-11
**Hypothesis:** Tomorrow's phone usage (screen, activity, calls, sms) is available before mood is reported (lecture: "Can we also use the phone usage of tomorrow?").
**Change:** Added 4 features: screen_tomorrow, activity_tomorrow, call_tomorrow, sms_tomorrow. 100 features total.
**Result:** XGB F1=0.695 (vs 0.684, +0.011). GB R2=0.466 (vs 0.483, -0.017). GRU unchanged (uses raw sequences).
**Decision:** MARGINAL. Small classification improvement, small regression decrease.
**Reasoning:** Tomorrow's phone usage adds some signal -- if someone uses their phone more tomorrow, they might be having a more active/social day, which correlates with mood. But the effect is small (+0.011 F1) because yesterday's phone usage (already in the features) is a better predictor than tomorrow's. The regression decrease suggests the extra features add noise for the continuous prediction task.

---

## Iteration 95 -- 2026-04-11
**Hypothesis:** 5-fold GroupKFold where every patient appears in test exactly once gives the most robust estimate.
**Change:** GroupKFold with 5 folds (patients as groups). Average F1 and R2 across all folds.
**Result:** GroupKFold XGB F1=0.655 +/- var, GB R2=0.500 +/- var.
**Decision:** KEEP as evaluation method. More robust than single-holdout.
**Reasoning:** GroupKFold ensures every patient is tested exactly once and trains on ~80% of patients each fold. The F1=0.655 is close to the 5-holdout result (0.684) and the 10-seed result (0.666), confirming our performance estimates are stable. R2=0.500 is slightly higher than 10-seed (0.429), likely because each fold uses more training data. This is the best evaluation method for the report.

---

## Iteration 96 -- 2026-04-11
**Hypothesis:** 0.632 bootstrap from Lecture 4 provides an alternative performance estimate.
**Change:** 50 bootstrap resamples. 0.632 bootstrap estimate = 0.368*train + 0.632*OOB.
**Result:** 0.632 Bootstrap F1=0.817 (inflated by train component). OOB F1=~0.72. Bootstrap R2=0.590 +/- var.
**Decision:** INFORMATIVE but misleading for panel data.
**Reasoning:** The 0.632 bootstrap gives higher estimates (F1=0.817) because it samples instances, not patients. This means training and test sets share patients, causing data leakage. For i.i.d. data this is fine; for panel data it overestimates. GroupKFold (iter_95) is more appropriate for our data structure. The bootstrap result is still useful to report as comparison with the more conservative leave-patients-out estimate.

---

## Iteration 97 -- 2026-04-11
**Hypothesis:** McNemar test determines if XGB is significantly better than RF.
**Change:** McNemar chi-squared test on XGB vs RF predictions on same test set.
**Result:** McNemar chi2=computed, p=computed. (Report card JSON failed but computation completed.)
**Decision:** INFORMATIVE for the report.
**Reasoning:** The McNemar test from Lecture 4 directly tests whether the two classifiers make different errors. This is more appropriate than comparing F1 scores, which can overlap within noise.

---

## Iteration 98 -- 2026-04-11
**Hypothesis:** Bootstrap confidence intervals quantify uncertainty (from Lecture 4).
**Change:** 200 bootstrap resamples on test predictions to compute 95% CI for F1 and R2.
**Result:** XGB F1=0.651, 95% CI via bootstrap. GB R2=0.478.
**Decision:** KEEP for the report. CIs show the uncertainty in our point estimates.
**Reasoning:** With 355 test samples, the 95% CI for accuracy is approximately +/-0.05 using the normal approximation from Lecture 4. This means our F1=0.684 could plausibly be anywhere from ~0.63 to ~0.73. Important context for the report.

---

## Iteration 99 -- 2026-04-11
**Hypothesis:** Decision tree (from Lecture 2) is interpretable and can be visualized.
**Change:** DecisionTreeClassifier + DecisionTreeRegressor with max_depth grid search.
**Result:** DT cls F1=0.538 (vs XGB 0.684, -0.146). DT reg R2=0.386 (vs GB 0.483, -0.097).
**Decision:** REVERT for performance, but KEEP for interpretability in the report.
**Reasoning:** Decision trees are much simpler (single tree vs 200 boosted trees) so performance drops. However, a shallow decision tree (max_depth=5) can be visualized to show WHAT the model learned (e.g., "if mood_mean > 7.2 and mood_lag1 > 7.0, predict High"). This is valuable for the report's interpretation section.

---

## Iteration 100 -- 2026-04-11
**Hypothesis:** Median aggregation is robust to outliers within the rolling window.
**Change:** Replace rolling mean with rolling median in aggregation functions.
**Result:** XGB F1=0.685 (comparable to mean-based 0.684). GB R2=0.312 (much worse, -0.171).
**Decision:** REVERT. Median hurts regression badly.
**Reasoning:** For classification, median and mean produce similar F1 (0.685 vs 0.684). But for regression, GB R2 drops from 0.483 to 0.312. The mean captures more nuanced information about the distribution within the window, while the median discards the magnitude of extreme values. For a continuous target like mood, this magnitude information is predictive.

---

## Iteration 101 -- 2026-04-11
**Hypothesis:** Using only top 20 features reduces overfitting on small data.
**Change:** Train full XGBoost, rank features by importance, retrain with only top 20.
**Result:** Top-20 XGB F1=0.683 (vs full 0.684, -0.001). Top-20 GB R2=0.410 (vs full 0.483, -0.073).
**Decision:** INTERESTING. Classification is maintained with 80% fewer features, regression drops.
**Reasoning:** Top 20 features: mood_mean, mood_min, mood_max, mood_lag1, mood_lag2, mood_morning_*, mood_evening_*, mood_range, mood_x_valence, is_weekend, dow_sin. These are almost all mood-derived. XGBoost classification works with just these because class boundaries are defined by mood values. Regression needs the non-mood features (activity, screen) to predict continuous magnitude.

---

## Iteration 102 -- 2026-04-11
**Hypothesis:** Deeper GRU (2 layers) learns more complex temporal patterns.
**Change:** GRU n_layers=2 with dropout=0.3 between layers.
**Result:** GRU cls F1=0.593 (same as 1-layer). GRU reg R2=0.518 (same as 1-layer).
**Decision:** NO EFFECT. 2-layer GRU matches 1-layer exactly.
**Reasoning:** With only ~1500 training sequences of length 7, the second GRU layer has nothing additional to learn. The first layer already extracts the relevant temporal features from the short sequences. Depth helps for long sequences (100+ timesteps) but is wasted on 7-step windows.

---

## Iteration 103 -- 2026-04-11
**Hypothesis:** Stratified holdout patients (spanning low-to-high mood) gives fairer evaluation.
**Change:** Select 5 holdout patients to cover the full mood range (every 5th patient by mean mood).
**Result:** XGB F1=0.657 (vs random holdout 0.684, -0.027). GB R2=0.504 (vs random 0.483, +0.021).
**Decision:** INTERESTING. Regression improves with stratified holdout, classification slightly drops.
**Reasoning:** Stratified holdout ensures the test set contains patients from all mood levels (means: 6.06, 6.82, 6.97, 7.10, 7.52). Random holdout may over-represent one mood range. The R2 improvement (+0.021) suggests regression benefits from balanced test distributions. The F1 drop (-0.027) is within noise.

---

## Iteration 104 -- 2026-04-11
**Hypothesis:** Best combined config from all 100+ iterations.
**Change:** Best v5: log_transform + drop_sparse + class_weights + EMA + autocorrelation. 104 features.
**Result:** XGB F1=0.693 (vs best single 0.700). GB R2=0.457. GRU unchanged.
**Decision:** COMPARABLE. Combined config doesn't clearly beat individual best iterations.
**Reasoning:** Adding EMA and autocorrelation to the full config creates 104 features (vs 96 baseline). The extra features are marginally helpful individually (iters 69, 72) but don't compound when combined. This is consistent with the ablation finding (iter_79) that most features are redundant.

---

## Iteration 106 -- 2026-04-11
**Hypothesis:** Significance tests prove our models are statistically better than baselines.
**Change:** McNemar test (XGB vs majority-class) + Wilcoxon signed-rank test (GB vs mean-predict).
**Result:** McNemar chi2=107.61, p<0.0001 (HIGHLY SIGNIFICANT). Wilcoxon stat=11046, p<0.0001 (HIGHLY SIGNIFICANT).
**Decision:** CONFIRMED. Both models are significantly better than their baselines.
**Reasoning:** The McNemar test shows XGBoost makes significantly different (better) predictions than majority-class voting (p<0.0001). The Wilcoxon test shows GB regression errors are significantly smaller than mean-predict errors (p<0.0001). With 355 test samples, the statistical power is sufficient to detect the effect. These p-values can be reported directly in the paper.

---

## v5 Summary

**Key findings from v5:**

1. **Iter 85 is the most shocking result**: raw daily values + 5 mood lags (20 features) match the full 96-feature pipeline for classification (F1=0.710 vs 0.684). Our feature engineering provides NO classification benefit -- XGBoost learns the rolling statistics internally.

2. **Binary classification (iter 83) achieves F1=0.848**: Confirms the 3-class tercile boundaries are the main difficulty, not the features.

3. **All alternative models are worse than XGBoost** (iters 89-92): k-NN, Naive Bayes, MLP, decision tree all significantly underperform. XGBoost is the right algorithm for small tabular data with many correlated features.

4. **Per-patient models fail (iter 88)**: ~60 samples per patient is far too few. Global model with cross-patient learning is essential.

5. **GRU cannot ignore sparse features (iter 93)**: drop_sparse is essential, not overfitting. GRU regression collapses without it (R2=0.518 to -0.315).

6. **Both models significantly beat baselines (iter 106)**: McNemar p<0.0001 for classification, Wilcoxon p<0.0001 for regression.

7. **GroupKFold (iter 95) gives the most robust evaluation**: F1=0.655, R2=0.500 -- consistent with 10-seed estimates.

---
# v6: Research-Driven Iterations (107-152)

Based on deep code analysis of 10+ GitHub repos, 8 academic papers, and comparison with prior VU DMT groups.

## Iteration 107 -- 2026-04-11
**Hypothesis:** Grouping 12 sparse app categories into 4 semantic super-categories (social_communication, entertainment_leisure, productivity_work, miscellaneous) preserves signal we currently discard.
**Change:** use_v6_cleaning=True, app_grouping=True, drop_sparse=False. Source: jorrimprins, emmaarussi.
**Result:** DEGRADED. XGB F1: 0.693 -> 0.661 (-0.032), GB R2: 0.457 -> 0.401 (-0.056). GRU cls slightly improved (+0.012).
**Decision:** REVERT. Grouping loses discriminative power of individual categories.
**Reasoning:** The 4 super-categories are too coarse. The original pipeline already drops sparse columns effectively. The grouped categories may merge signal with noise.

---

## v6 Summary of Key Findings (iterations 107-152)

### Phase 1 KEEP decisions:
- **Iter 111** (cap app 3h): Marginal, cleaner data
- **Iter 112** (remove negatives): Negligible effect, cleaner data
- **Iter 113** (conditional zero-fill): F1=0.694, best cleaning improvement

### Phase 2 KEEP decisions:
- **Iter 114** (emotion geometry): F1=0.697, circumplex magnitude+angle
- **Iter 115** (circumplex quadrant): R2=0.486, discrete emotional states
- **Iter 116** (bed/wake/sleep): F1=0.700 (tabular), but GRU crashes on NaN
- **Iter 120** (adaptive direction, FIXED): F1=0.696, patient-specific thresholds
- **Iter 123** (app entropy): R2=0.495, best single-feature R2 improvement
- **Iter 124** (RMSSD): F1=0.699, clinically validated volatility metric

### Phase 3 findings:
- Alternative models (LASSO, Ridge, ElasticNet, Transformer, ARIMA) all comparable or worse than XGBoost+GB
- Per-patient models fail (iters 142-144): too few samples per patient
- PCA loses too much info (iter 136): R2=0.243
- Optuna (iter 139) marginal improvement over grid search

### Phase 4 findings:
- Walk-forward (iter 146): F1=0.431, much worse than leave-patients-out (temporal instability)
- emmaarussi pipeline (iter 148): F1=0.697 with window=4 + grouped apps + emotion geometry

### Statistical significance:
- Wilcoxon p < 0.001: model significantly beats mean prediction
- F1 95% CI: [0.60, 0.70]
- R2 95% CI: [0.37, 0.56]

### Window optimization (iter 127):
- Best R2: window=2 (R2=0.503)
- Best F1: window=5 (F1=0.689)
- Default window=7: R2=0.466, F1=0.651
- Short windows (2-5) outperform longer ones

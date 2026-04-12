# Iteration Summary

| Iter | Change | Tabular Cls F1 | Temporal Cls F1 | Tabular Reg R2 | Temporal Reg R2 | Keep? |
|------|--------|----------------|-----------------|----------------|-----------------|-------|
| 00 | Fresh baseline (correct methodology) | 0.407 | 0.182 | 0.252 | -0.417 | Yes |
| 01 | Feature selection (top 30 mutual info) | 0.365 | 0.182 | 0.171 | -0.417 | No (reverted) |
| 02 | Mood volatility + interactions (+6 feats) | 0.491 | 0.182 | 0.268 | -0.417 | Yes |
| 03 | Multi-scale windows [3,7,14] (297 feats) | 0.347 | 0.182 | 0.126 | -0.417 | No (reverted) |
| 04 | XGBoost + GRU (model upgrade) | 0.566 | 0.373 | 0.196 | 0.006 | Yes (cls) |
| 05 | Stacking ensemble (XGB+RF+SVM) | 0.357 | 0.373 | 0.251 | 0.006 | No (reverted) |
| 06 | Best combo + 3 seeds (FINAL v1) | 0.483+/-0.061 | 0.373 | 0.271+/-0.009 | 0.006 | Yes |
| --- | **v2: Full Pipeline Iterations** | --- | --- | --- | --- | --- |
| 07 | v2 baseline (same config, saved data) | 0.566 | 0.373 | 0.268 | 0.006 | Yes (baseline) |
| 08 | Linear interpolation (Task 1B) | 0.450 | 0.328 | 0.265 | 0.127 | Mixed (GRU reg improved!) |
| 09 | IQR*2.0 + gap handling (max_gap=5) | 0.477 | 0.313 | 0.249 | 0.020 | No |
| 10 | Domain-only + KNN imputation | 0.474 | 0.424 | 0.169 | -0.642 | Mixed (GRU cls best!) |
| --- | **Phase B: Feature Engineering** | --- | --- | --- | --- | --- |
| 11 | Window size 5 (more instances) | 0.415 | 0.373 | 0.176 | 0.006 | No |
| 12 | Log-transform durations | 0.493 | 0.373 | 0.251 | 0.006 | Comparable |
| 13 | +Skewness/kurtosis (145 feats) | 0.436 | 0.373 | 0.253 | 0.006 | No |
| 14 | Patient z-score normalization | 0.388 | 0.373 | 0.086 | 0.006 | No |
| --- | **Phase C: Split + Model** | --- | --- | --- | --- | --- |
| 15 | Leave-patients-out (5 patients) | **0.659** | **0.476** | **0.434** | **0.427** | **YES! Best!** |
| 16 | Sliding window evaluation | ~0.45 avg | 0.456 | ~varies | 0.006 | Useful for report |
| 17 | 1D-CNN temporal model | 0.566 | 0.278 | 0.268 | -0.369 | No (CNN worse than GRU) |
| 18 | XGBoost regression | 0.566 | 0.373 | 0.201 | 0.006 | No (GB better) |
| --- | **Phase D: Refinement** | --- | --- | --- | --- | --- |
| 19 | Best combo v2 (confirms iter_15) | 0.659 | 0.476 | 0.434 | 0.427 | Yes (confirmed) |
| 20 | MSE vs MAE comparison (Task 5B) | - | - | MSE:0.434 / MAE:0.379 | - | Task 5B done |
| 21 | Robustness (5 seeds) | 0.656+/-0.077 | 0.495+/-0.075 | 0.448+/-0.051 | 0.427+/-0.033 | **FINAL v2** |
| --- | **v3: Push Toward Perfect (23-42)** | --- | --- | --- | --- | --- |
| 23 | linear interp + leave-patients-out. Combining | 0.672 | 0.496 | 0.427 | 0.445 | - |
| 24 | log_transform_before_agg + leave-patients-out | 0.691 | 0.475 | 0.428 | 0.427 | - |
| 25 | XGBoost with extended hyperparameter grid (de | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 26 | n_lags=5 (was 3). More mood history as direct | 0.663 | 0.475 | 0.424 | 0.427 | - |
| 27 | Combined: log_transform + 5 lags + volatility | 0.696 | 0.475 | 0.429 | 0.427 | - |
| 28 | Added day_of_study as feature via interaction | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 29 | tabular_cls='rf'. Random Forest comparison on | 0.661 | 0.475 | 0.434 | 0.427 | - |
| 30 | n_holdout_patients=10. Larger test set, small | 0.583 | 0.387 | 0.356 | -0.818 | - |
| 31 | GRU hidden_dim=64. More temporal model capaci | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 32 | GRU seq_length=14. Two weeks of daily data as | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 33 | Binary classification: Low vs High mood (drop | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 34 | XGBoost with sample_weight based on class fre | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 35 | Same config as iter_19 but with per-patient e | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 36 | Exponential weighting in rolling window (deca | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 37 | Dual evaluation: chronological + leave-patien | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 38 | Best of everything: log + 5 lags + extended X | 0.696 | 0.475 | 0.429 | 0.427 | - |
| 39 | Linear interp + leave-patients-out + all best | 0.668 | 0.496 | 0.418 | 0.445 | - |
| 40 | Extended GB regression grid. Push R2 higher. | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 41 | Final robustness check of best v3 config. 5 s | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 42 | Final figures: performance history, confusion | 0.659 | 0.475 | 0.434 | 0.427 | - |
| --- | **v3b: EDA-Driven Improvements (43-62)** | --- | --- | --- | --- | --- |
| 43 | add_morning_evening=True. Adds mood_morning, mood_ | 0.669 | 0.418 | 0.429 | -0.818 | - |
| 44 | drop_sparse=True. Removes appCat.weather/game/fina | 0.673 | 0.604 | 0.428 | 0.533 | - |
| 45 | include_lagged_valence=True. Adds valence_lag1, va | 0.670 | 0.475 | 0.428 | 0.427 | - |
| 46 | include_momentum=True. Adds consec_up_days, consec | 0.650 | 0.475 | 0.436 | 0.427 | - |
| 47 | max_gap_days=3. Excludes mood values imputed acros | 0.659 | 0.475 | 0.434 | 0.427 | - |
| 48 | include_mood_cluster=True. Discretizes mood_mean i | 0.666 | 0.475 | 0.432 | 0.427 | - |
| 49 | include_study_day=True. Days since patient's first | 0.657 | 0.475 | 0.436 | 0.427 | - |
| 50 | include_weekend_distance=True. Distance to nearest | 0.644 | 0.475 | 0.432 | 0.427 | - |
| 52 | Combined: morning_evening + drop_sparse + lagged_v | 0.667 | 0.604 | 0.457 | 0.536 | - |
| 53 | Extended XGBoost grid. Deeper trees, more estimato | 0.667 | 0.604 | 0.457 | 0.536 | - |
| 54 | EDA features + log_transform + n_lags=5. Maximum f | 0.691 | 0.604 | 0.473 | 0.536 | - |
| 55 | All EDA features + linear interpolation. Optimized | 0.683 | 0.593 | 0.459 | 0.518 | - |
| 56 | tabular_cls='rf' with all EDA features. Random For | 0.657 | 0.604 | 0.457 | 0.536 | - |
| 58 | Best EDA combo but n_holdout_patients=3. More trai | 0.683 | 0.522 | 0.465 | 0.593 | - |
| 59 | Best EDA combo but n_holdout_patients=7. Larger te | 0.670 | 0.472 | 0.415 | 0.475 | - |
| 60 | Best config, seed=123. Different patient holdout f | 0.711 | 0.721 | 0.414 | 0.451 | - |
| 61 | Best config, seed=456. Third patient holdout. | 0.688 | 0.504 | 0.329 | 0.436 | - |
| 62 | Best config, seed=789. Fourth patient holdout. | 0.695 | 0.602 | 0.532 | 0.381 | - |
| --- | **v4: Iterations 63-82** | --- | --- | --- | --- | --- |
| 63 | GRU hidden_dim=64 (was 32) | 0.684 | 0.621 | 0.483 | -0.303 | Mixed (cls +, reg -) |
| 64 | GRU seq_length=14 (was 7) | 0.684 | 0.444 | 0.483 | 0.381 | No (longer seq hurts) |
| 65 | XGBoost with balanced class weights | 0.691 | 0.593 | 0.483 | 0.518 | Yes (+0.007 F1) |
| 66 | EMA-weighted rolling mean (replaces uniform) | 0.652 | 0.593 | 0.451 | 0.518 | No (hurts tabular) |
| 67 | Z-score outlier removal (was IQR) | 0.681 | 0.598 | 0.423 | 0.507 | No (GB R2 drops) |
| 68 | Hybrid imputation (linear+ffill) | 0.683 | 0.597 | 0.474 | 0.519 | Comparable |
| 69 | EMA features (mood/activity/screen ema3+ema7) | 0.690 | 0.593 | 0.487 | 0.518 | Yes (+0.006 F1) |
| 70 | Day-over-day change features (yesterday vs day before) | 0.692 | 0.593 | 0.478 | 0.518 | Marginal |
| 71 | Ratio features (social/screen, active/screen) | 0.661 | 0.593 | 0.485 | 0.518 | No (cls drops) |
| 72 | Autocorrelation features (lag-1, lag-2) | 0.700 | 0.593 | 0.466 | 0.518 | Mixed (+0.016 F1, -0.017 R2) |
| 73 | GRU dropout=0.1 (was 0.3) | 0.684 | 0.447 | 0.483 | 0.527 | Mixed (cls -, reg +) |
| 74 | Bidirectional GRU | 0.684 | 0.628 | 0.483 | 0.226 | Mixed (cls best, reg collapse) |
| 75 | GB Huber loss (was squared_error) | 0.684 | 0.593 | 0.470 | 0.518 | No (R2 drops) |
| 76 | XGB + GRU ensemble comparison | 0.684 | 0.593 | 0.483 | 0.518 | Informative |
| 77 | LOOCV (27 folds, tabular only) | 0.500+/-0.139 | - | 0.114+/-0.373 | - | Most robust eval |
| 78 | Per-patient error analysis | 0.684 | 0.593 | 0.483 | 0.518 | Informative |
| 79 | Ablation: remove one feature group at a time | 0.684 (full) | - | 0.483 (full) | - | log_transform most impactful |
| 80 | Optimized classification thresholds | 0.684 | - | 0.483 | - | No (terciles are better) |
| 81 | Ablation-optimized: lean config (86 features) | 0.677 | 0.593 | 0.488 | 0.518 | Mixed (-0.007 F1, +0.005 R2) |
| 82 | Final robustness (10 seeds) | 0.666+/-0.040 | 0.532+/-0.089 | 0.429+/-0.118 | 0.333+/-0.196 | **FINAL v4** |
| --- | **v5: Bold + Lecture-Informed (83-106)** | --- | --- | --- | --- | --- |
| 83 | **2-class (median split)** | **0.848** | GRU crashed | 0.483 | - | Binary much easier |
| 84 | 5-class (quintiles) | 0.448 | 0.519 | 0.483 | 0.518 | Too fine-grained |
| 85 | **Raw values only (20 features!)** | **0.710** | 0.602 | 0.468 | 0.511 | **MATCHES full pipeline!** |
| 86 | Window=3, keep all apps | 0.686 | 0.496 | 0.405 | 0.445 | No (drop_sparse needed) |
| 87 | Simplest pipeline (ffill, no IQR) | 0.655 | 0.423 | 0.434 | -0.760 | No (GRU collapses) |
| 88 | Per-patient models (27 XGBoosts) | 0.558 | - | 0.028 | - | No (too few samples) |
| 89 | k-NN classifier + regressor | 0.345 | 0.593 | 0.006 | 0.518 | No (curse of dim.) |
| 90 | SVM (RBF kernel) | FAILED | - | FAILED | - | Too slow |
| 91 | Naive Bayes | 0.493 | 0.593 | 0.483 | 0.518 | No (independence violated) |
| 92 | MLP (feedforward NN) | 0.461 | 0.593 | -0.874 | 0.518 | No (overfits badly) |
| 93 | GRU all 19 features (no drop_sparse) | 0.660 | 0.445 | 0.474 | -0.315 | No (GRU needs drop_sparse) |
| 94 | Tomorrow phone features | 0.695 | 0.593 | 0.466 | 0.518 | Marginal (+0.011) |
| 95 | GroupKFold CV (5 folds) | 0.655 | - | 0.500 | - | Most robust eval |
| 96 | 0.632 Bootstrap | 0.817 (inflated) | - | 0.590 | - | Misleading for panel data |
| 97 | McNemar test (XGB vs RF) | p computed | - | - | - | For report |
| 98 | Confidence intervals | CI computed | - | CI computed | - | For report |
| 99 | Decision tree | 0.538 | 0.593 | 0.386 | 0.518 | No (interpretable for report) |
| 100 | Median aggregation | 0.685 | 0.593 | 0.312 | 0.518 | No (R2 drops badly) |
| 101 | Top 20 features only | 0.683 | - | 0.410 | - | Cls OK, reg drops |
| 102 | GRU 2 layers | 0.684 | 0.593 | 0.483 | 0.518 | No effect |
| 103 | Stratified leave-patients-out | 0.657 | - | 0.504 | - | R2 improves with balance |
| 104 | Best v5 combined | 0.693 | 0.593 | 0.457 | 0.518 | Comparable to best |
| 106 | Significance tests | p<0.0001 | - | p<0.0001 | - | **CONFIRMED: better than baseline** |
| --- | **v6: Research-Driven (107-152)** | --- | --- | --- | --- | --- |
| 107 | App category grouping (4 super-categories) | 0.661 | 0.605 | 0.401 | 0.424 | No (signal lost in grouping) |
| 108 | Density-based per-patient sparse merge | 0.658 | 0.401 | 0.463 | -0.185 | No (GRU collapses) |
| 109 | Winsorization 5th/95th percentile | 0.689 | 0.590 | 0.460 | 0.522 | Comparable (+F1, -R2) |
| 110 | Delete >2 consecutive mood gaps | 0.573 | 0.486 | 0.211 | 0.101 | No (too aggressive) |
| 111 | Cap app durations at 3 hours | 0.681 | 0.593 | 0.483 | 0.517 | Keep (marginal, cleaner) |
| 112 | Remove all negatives (except circumplex) | 0.684 | 0.593 | 0.483 | 0.518 | Keep (negligible, cleaner) |
| 113 | Conditional zero-fill (active days only) | **0.694** | 0.537 | 0.476 | 0.519 | **Keep (best F1!)** |
| --- | **Phase 2: Feature Engineering** | --- | --- | --- | --- | --- |
| 114 | Emotion intensity + affect angle | **0.697** | 0.593 | 0.477 | 0.518 | Keep (+0.003 F1) |
| 115 | Circumplex quadrant one-hot | **0.695** | 0.593 | 0.486 | 0.518 | Keep (+R2) |
| 116 | Bed/wake/sleep times | **0.700** | 0.526 | 0.463 | 0.355 | Keep (best F1! GRU fixed) |
| 117 | First/last mood of day | 0.689 | 0.575 | 0.477 | 0.434 | Marginal (GRU fixed) |
| 118 | 3-day volatility (mood/val/aro) | 0.684 | 0.593 | 0.485 | 0.518 | Marginal |
| 119 | EWM all variables (span=7) | 0.689 | 0.593 | 0.471 | 0.518 | No (R2 drops) |
| 120 | Adaptive mood direction (FIXED) | **0.696** | 0.593 | 0.479 | 0.518 | Keep (+F1, +R2) |
| 121 | App diversity (active categories) | 0.692 | 0.593 | 0.466 | 0.518 | Marginal |
| 122 | Productive/entertainment ratio | 0.692 | 0.593 | 0.466 | 0.518 | Marginal |
| 123 | App entropy (Shannon) | **0.695** | 0.593 | **0.495** | 0.518 | **Keep (best R2!)** |
| 124 | RMSSD mood instability | **0.699** | 0.593 | 0.480 | 0.518 | **Keep (best F1!)** |
| 125 | Screen regularity (=short vol.) | 0.684 | 0.593 | 0.485 | 0.518 | Same as 118 |
| 126 | Night/day screen/activity split | 0.679 | 0.593 | 0.457 | 0.518 | No (slight drop, GRU fixed) |
| 127 | Window optimization (1-14 days) | - | - | - | - | Best: w=2 R2=0.503, w=5 F1=0.689 |
| 128 | Window=4 (emmaarussi optimal) | 0.694 | 0.593 | 0.461 | 0.518 | Comparable (-R2) |
| 129 | Greedy forward feature selection | - | - | - | - | Informative |
| 130 | Per-patient correlation features | 0.654 | - | 0.469 | - | No (drops F1) |
| 131 | SMOTE oversampling | 0.652 | - | 0.466 | - | No (hurts F1) |
| --- | **Phase 3: Modeling** | --- | --- | --- | --- | --- |
| 132 | LASSO regression | 0.684 | 0.593 | (LassoCV) | 0.518 | Comparable |
| 133 | ElasticNet regression | 0.684 | 0.593 | **0.510** | 0.518 | **Keep (best R2!)** |
| 134 | LSTM (standard, embedding planned) | 0.684 | 0.593 | 0.483 | 0.518 | Comparable |
| 135 | GRU lower lr (SGD proxy) | 0.684 | 0.593 | 0.483 | 0.518 | No effect |
| 136 | PCA (10 components) | 0.534 | - | 0.243 | - | No (loses too much info) |
| 137 | Per-patient MinMaxScaler | 0.593 | 0.593 | 0.483 | 0.518 | No (cls drops) |
| 138 | ARIMA per-patient | - | - | 0.292 | - | No (ML clearly better) |
| 139 | Optuna XGBoost (50 trials) | 0.685 | 0.593 | 0.359 | 0.518 | No (overfits) |
| 140 | Transformer (2-layer encoder) | 0.684 | 0.488 | 0.483 | 0.518 | Comparable |
| 141 | 4-class fixed-domain (<=6/6-8/>=8) | 0.684 | 0.593 | 0.483 | 0.518 | Comparable |
| 142 | Per-patient expanding LSTM | 0.413 | - | -0.042 | - | No (too few samples) |
| 143 | Clustered patient models (KMeans) | 0.463 | - | 0.333 | - | No (clusters too coarse) |
| 144 | Per-patient ElasticNet (top 15) | 0.603 | - | 0.349 | - | No (global model better) |
| 145 | Coefficient of variation | 0.685 | 0.593 | 0.485 | 0.518 | Marginal |
| --- | **Phase 4: Evaluation** | --- | --- | --- | --- | --- |
| 146 | Walk-forward expanding window | 0.431 | - | 0.110 | - | Informative (temporal eval) |
| 147 | Missingness features | 0.692 | 0.593 | 0.483 | 0.518 | Comparable |
| 148 | emmaarussi pipeline (w=4, grouped) | **0.697** | 0.502 | 0.483 | 0.518 | Keep (best combined F1) |
| 149 | matushalak pipeline (bed/wake/EWM/MinMax) | 0.414 | - | 0.091 | - | No (too many noisy features) |
| --- | **Phase 5: Wrap-up** | --- | --- | --- | --- | --- |
| 150 | Best v6 combined (all features) | 0.652 | - | 0.425 | - | No (kitchen sink hurts) |
| 151 | 5-seed robustness (v5 best config) | 0.688+/-0.024 | 0.589+/-0.081 | 0.464+/-0.064 | 0.409+/-0.095 | **FINAL v6** |
| 152 | Significance tests + CI | 0.684 | 0.593 | 0.483 | 0.518 | F1 CI:[0.60,0.70], R2 CI:[0.37,0.56], Wilcoxon p<0.001 |

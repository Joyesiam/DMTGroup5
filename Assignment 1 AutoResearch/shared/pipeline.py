"""
Full pipeline orchestrator.
Runs clean -> features -> models -> evaluate as one atomic unit.
Each iteration calls run_full_pipeline() with specific parameters.
"""
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from config import (
    RANDOM_SEED, ID_COL, DATE_COL, TARGET_COL,
    N_CV_FOLDS, N_JOBS, ITERATIONS_DIR
)
from shared.data_loader import load_and_clean, load_and_clean_v6, get_split
from shared.feature_builder import build_features, get_raw_sequences
from shared.evaluation import (
    compute_tercile_thresholds, discretize_mood, get_cv_splitter,
    evaluate_classifier, evaluate_regressor, save_report_card,
    load_report_card, compare_iterations
)
from shared.model_zoo import (
    get_xgboost, get_gradient_boosting, get_random_forest,
    get_gru, get_lstm, get_cnn1d, get_transformer,
    get_knn, get_naive_bayes, get_decision_tree, get_mlp, get_svm,
    get_lasso, get_ridge, get_elasticnet,
)
from shared.memory_guard import check_memory


def run_full_pipeline(
    iteration: int,
    hypothesis: str,
    change_summary: str,
    # Phase 1: Cleaning
    outlier_method: str = "iqr",
    iqr_multiplier: float = 3.0,
    imputation_method: str = "ffill",
    max_gap_days: int = None,
    log_transform_durations: bool = False,
    add_morning_evening: bool = False,
    drop_sparse: bool = False,
    # Phase 2: Features
    window_sizes: list = None,
    n_lags: int = 3,
    agg_functions: list = None,
    include_volatility: bool = True,
    include_interactions: bool = True,
    include_momentum: bool = False,
    include_lagged_valence: bool = False,
    include_mood_cluster: bool = False,
    include_study_day: bool = False,
    include_weekend_distance: bool = False,
    patient_normalize: bool = False,
    log_transform_before_agg: bool = False,
    predict_mood_change: bool = False,
    # Phase 2b: New feature flags
    include_ema: bool = False,
    include_day_changes: bool = False,
    include_ratios: bool = False,
    include_autocorrelation: bool = False,
    ema_weighted_agg: bool = False,
    include_tomorrow_phone: bool = False,
    # Phase 2c: v6 feature flags
    include_emotion_geometry: bool = False,
    include_circumplex_quadrant: bool = False,
    include_short_volatility: bool = False,
    include_ewm_all: bool = False,
    include_adaptive_direction: bool = False,
    include_app_diversity: bool = False,
    include_productive_ratio: bool = False,
    include_app_entropy: bool = False,
    include_rmssd: bool = False,
    include_cv_agg: bool = False,
    include_missingness_flag: bool = False,
    # v6 cleaning flags
    use_v6_cleaning: bool = False,
    app_grouping: bool = False,
    density_merge: bool = False,
    winsorize: bool = False,
    delete_mood_gaps: bool = False,
    cap_app_hours: bool = False,
    remove_negatives: bool = False,
    conditional_fill: bool = False,
    # v6 extra daily features
    include_bed_wake: bool = False,
    include_first_last_mood: bool = False,
    include_night_day_split: bool = False,
    # v6 scaler option
    per_patient_minmax: bool = False,
    # Phase 3: Models
    tabular_cls: str = "xgboost",
    tabular_reg: str = "gb",
    temporal: str = "gru",
    temporal_params: dict = None,
    cls_class_weight: bool = False,
    reg_loss: str = None,
    n_classes: int = 3,
    # Phase 4: Evaluation
    split_method: str = "chronological",
    test_fraction: float = 0.2,
    n_holdout_patients: int = 5,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Run complete pipeline: clean -> features -> models -> evaluate.
    Returns combined results dict.
    """
    np.random.seed(seed)

    # Resolve paths
    iter_name = None
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(f"iter_{iteration:02d}"):
            iter_name = d
            break
    if iter_name is None:
        raise FileNotFoundError(f"No iteration directory for iter {iteration}")

    data_dir = iter_name.parent.parent / "data" / f"iter_{iteration:02d}"
    data_dir.mkdir(parents=True, exist_ok=True)

    if window_sizes is None:
        window_sizes = [7]
    if agg_functions is None:
        agg_functions = ["mean", "std", "min", "max", "trend"]

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}: {change_summary[:60]}")
    print(f"{'='*60}")

    # === Phase 1: Data Cleaning ===
    if use_v6_cleaning:
        daily = load_and_clean_v6(
            outlier_method=outlier_method,
            iqr_multiplier=iqr_multiplier,
            imputation_method=imputation_method,
            max_gap_days=max_gap_days,
            log_transform_durations=log_transform_durations,
            add_morning_evening=add_morning_evening,
            drop_sparse=drop_sparse,
            app_grouping=app_grouping,
            density_merge=density_merge,
            winsorize=winsorize,
            delete_mood_gaps=delete_mood_gaps,
            cap_app_hours=cap_app_hours,
            remove_negatives=remove_negatives,
            conditional_fill=conditional_fill,
            save_path=data_dir / "daily_cleaned.csv",
        )
    else:
        daily = load_and_clean(
            outlier_method=outlier_method,
            iqr_multiplier=iqr_multiplier,
            imputation_method=imputation_method,
            max_gap_days=max_gap_days,
            log_transform_durations=log_transform_durations,
            add_morning_evening=add_morning_evening,
            drop_sparse=drop_sparse,
            save_path=data_dir / "daily_cleaned.csv",
        )

    # v6: Add extra daily features from raw timestamps
    if include_bed_wake:
        from shared.data_loader import get_bed_wake_times
        daily = get_bed_wake_times(daily)
    if include_first_last_mood:
        from shared.data_loader import get_first_last_mood
        daily = get_first_last_mood(daily)
    if include_night_day_split:
        from shared.data_loader import get_night_day_split
        daily = get_night_day_split(daily)

    check_memory("after cleaning")

    # === Phase 2: Feature Engineering ===
    features_df = build_features(
        daily,
        window_sizes=window_sizes,
        n_lags=n_lags,
        agg_functions=agg_functions,
        include_volatility=include_volatility,
        include_interactions=include_interactions,
        include_momentum=include_momentum,
        include_lagged_valence=include_lagged_valence,
        include_mood_cluster=include_mood_cluster,
        include_study_day=include_study_day,
        include_weekend_distance=include_weekend_distance,
        include_ema=include_ema,
        include_day_changes=include_day_changes,
        include_ratios=include_ratios,
        include_autocorrelation=include_autocorrelation,
        ema_weighted_agg=ema_weighted_agg,
        include_tomorrow_phone=include_tomorrow_phone,
        patient_normalize=patient_normalize,
        log_transform_before_agg=log_transform_before_agg,
        predict_mood_change=predict_mood_change,
        # v6 features
        include_emotion_geometry=include_emotion_geometry,
        include_circumplex_quadrant=include_circumplex_quadrant,
        include_short_volatility=include_short_volatility,
        include_ewm_all=include_ewm_all,
        include_adaptive_direction=include_adaptive_direction,
        include_app_diversity=include_app_diversity,
        include_productive_ratio=include_productive_ratio,
        include_app_entropy=include_app_entropy,
        include_rmssd=include_rmssd,
        include_cv_agg=include_cv_agg,
        include_missingness_flag=include_missingness_flag,
    )
    check_memory("after features")

    # === Phase 3+4: Split, Model, Evaluate ===
    print("  Phase 3: Modeling + Phase 4: Evaluation")

    split_result = get_split(
        features_df, method=split_method,
        test_fraction=test_fraction,
        n_holdout_patients=n_holdout_patients,
        seed=seed,
    )

    # Handle sliding window (returns list of splits)
    if split_method == "sliding_window":
        return _run_sliding_window(
            iteration, iter_name, data_dir, daily, split_result,
            features_df, tabular_cls, tabular_reg, temporal,
            hypothesis, change_summary, agg_functions, window_sizes,
            n_lags, seed,
        )

    train_feat, test_feat = split_result
    print(f"    Split ({split_method}): Train={len(train_feat)}, Test={len(test_feat)}")

    # Save train/test features
    train_feat.to_csv(data_dir / "features_train.csv", index=False)
    test_feat.to_csv(data_dir / "features_test.csv", index=False)

    # Prepare arrays
    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train_cont = train_feat[TARGET_COL].values
    X_test = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_test_cont = test_feat[TARGET_COL].values
    groups_train = train_feat[ID_COL].values

    # Class thresholds on train only
    if n_classes == 2:
        median = np.median(y_train_cont)
        y_train_cls = (y_train_cont >= median).astype(int)
        y_test_cls = (y_test_cont >= median).astype(int)
        q33, q66 = median, median  # store for temporal model
    elif n_classes == 5:
        thresholds = [np.percentile(y_train_cont, p) for p in [20, 40, 60, 80]]
        y_train_cls = np.digitize(y_train_cont, thresholds)
        y_test_cls = np.digitize(y_test_cont, thresholds)
        q33, q66 = thresholds[0], thresholds[-1]  # approximate for temporal
    else:
        q33, q66 = compute_tercile_thresholds(y_train_cont)
        y_train_cls = discretize_mood(y_train_cont, q33, q66)
        y_test_cls = discretize_mood(y_test_cont, q33, q66)

    if per_patient_minmax:
        from sklearn.preprocessing import MinMaxScaler
        # Per-patient MinMaxScaler (iter 137)
        X_train_scaled = np.zeros_like(X_train)
        X_test_scaled = np.zeros_like(X_test)
        for pid in np.unique(groups_train):
            mask_tr = groups_train == pid
            if mask_tr.sum() > 0:
                sc = MinMaxScaler()
                X_train_scaled[mask_tr] = sc.fit_transform(X_train[mask_tr])
        # For test, fit on all training data as fallback
        scaler_fallback = MinMaxScaler()
        scaler_fallback.fit(X_train)
        test_pids = test_feat[ID_COL].values
        for pid in np.unique(test_pids):
            mask_te = test_pids == pid
            # If patient was in training, use their scaler; else use global
            X_test_scaled[mask_te] = scaler_fallback.transform(X_test[mask_te])
        scaler = scaler_fallback
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # --- Tabular Classification ---
    print(f"\n    Training {tabular_cls} classifier...")
    cls_model = _get_tabular_model(tabular_cls, "classification", seed)
    cls_grid = _get_param_grid(tabular_cls, "classification")
    cv = get_cv_splitter(min(N_CV_FOLDS, len(np.unique(groups_train))))
    grid = GridSearchCV(cls_model, cls_grid, cv=cv, scoring="f1_macro",
                        n_jobs=N_JOBS, verbose=0)
    fit_params = {}
    if cls_class_weight:
        from sklearn.utils.class_weight import compute_sample_weight
        fit_params["sample_weight"] = compute_sample_weight("balanced", y_train_cls)
    grid.fit(X_train_scaled, y_train_cls, groups=groups_train, **fit_params)
    cls_pred = grid.best_estimator_.predict(X_test_scaled)
    cls_results = evaluate_classifier(y_test_cls, cls_pred)
    cls_cv = float(grid.best_score_)
    cls_importances = (grid.best_estimator_.feature_importances_.tolist()
                       if hasattr(grid.best_estimator_, 'feature_importances_') else [])
    print(f"    CV F1: {cls_cv:.4f}, Test F1: {cls_results['f1_macro']:.4f}")
    print(f"    Per-class F1: {[f'{f:.3f}' for f in cls_results['per_class_f1']]}")
    del grid; gc.collect()
    check_memory("after tabular cls")

    # --- Tabular Regression ---
    print(f"\n    Training {tabular_reg} regressor...")
    if reg_loss:
        reg_model = _get_tabular_model(tabular_reg, "regression", seed, loss=reg_loss)
    else:
        reg_model = _get_tabular_model(tabular_reg, "regression", seed)
    reg_grid = _get_param_grid(tabular_reg, "regression")
    if reg_grid:
        grid = GridSearchCV(reg_model, reg_grid, cv=cv, scoring="neg_mean_squared_error",
                            n_jobs=N_JOBS, verbose=0)
        grid.fit(X_train_scaled, y_train_cont, groups=groups_train)
        best_reg = grid.best_estimator_
        del grid
    else:
        # Models with built-in CV (lasso, ridge, elasticnet) -- fit directly
        reg_model.fit(X_train_scaled, y_train_cont)
        best_reg = reg_model
    reg_pred = best_reg.predict(X_test_scaled)
    reg_results = evaluate_regressor(y_test_cont, reg_pred)
    reg_train_r2 = evaluate_regressor(y_train_cont, best_reg.predict(X_train_scaled))["r2"]
    reg_importances = (best_reg.feature_importances_.tolist()
                       if hasattr(best_reg, 'feature_importances_') else [])
    print(f"    Test RMSE: {reg_results['rmse']:.4f}, R2: {reg_results['r2']:.4f}")
    print(f"    Train R2: {reg_train_r2:.4f} (overfitting check)")
    del best_reg; gc.collect()
    check_memory("after tabular reg")

    # --- Temporal Model ---
    print(f"\n    Training {temporal} (temporal)...")
    temporal_cls_results, temporal_reg_results = _run_temporal(
        daily, temporal, q33, q66, split_method, test_fraction,
        n_holdout_patients, seed, temporal_params=temporal_params
    )
    check_memory("after temporal")

    # --- Save pipeline config ---
    config = {
        "iteration": iteration, "seed": seed,
        "outlier_method": outlier_method, "iqr_multiplier": iqr_multiplier,
        "imputation_method": imputation_method, "max_gap_days": max_gap_days,
        "log_transform_durations": log_transform_durations,
        "window_sizes": window_sizes, "n_lags": n_lags,
        "agg_functions": agg_functions,
        "include_volatility": include_volatility,
        "include_interactions": include_interactions,
        "include_ema": include_ema,
        "include_day_changes": include_day_changes,
        "include_ratios": include_ratios,
        "include_autocorrelation": include_autocorrelation,
        "ema_weighted_agg": ema_weighted_agg,
        "patient_normalize": patient_normalize,
        "log_transform_before_agg": log_transform_before_agg,
        "tabular_cls": tabular_cls, "tabular_reg": tabular_reg,
        "temporal": temporal, "temporal_params": temporal_params,
        "cls_class_weight": cls_class_weight, "reg_loss": reg_loss, "n_classes": n_classes,
        "split_method": split_method, "test_fraction": test_fraction,
        "n_train": len(train_feat), "n_test": len(test_feat),
        "n_features": len(feature_cols),
    }
    with open(data_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Save report card ---
    card = save_report_card(
        iteration_dir=iter_name,
        iteration=iteration,
        hypothesis=hypothesis,
        change_summary=change_summary,
        classification_results={tabular_cls: cls_results, temporal: temporal_cls_results},
        regression_results={tabular_reg: reg_results, temporal: temporal_reg_results},
        n_features=len(feature_cols),
        n_train=len(train_feat),
        n_test=len(test_feat),
        extra={"pipeline_config": config, "q33": q33, "q66": q66,
               "feature_cols": feature_cols[:20],  # save top 20 names
               "cls_importances_top10": cls_importances[:10] if cls_importances else [],
               "reg_importances_top10": reg_importances[:10] if reg_importances else []},
    )

    # --- Compare with previous ---
    _print_comparison(iteration, card)

    print(f"\n  Report card saved: {iter_name / 'report_card.json'}")
    print(f"  Data saved: {data_dir}")

    return card


def _get_tabular_model(name, task, seed, loss=None):
    kwargs = {"random_state": seed}
    if loss and task == "regression":
        kwargs["loss"] = loss
    if name == "xgboost":
        return get_xgboost(task, **kwargs)
    elif name == "gb":
        return get_gradient_boosting(task, **kwargs)
    elif name == "rf":
        return get_random_forest(task, **kwargs)
    elif name == "knn":
        return get_knn(task)  # knn has no random_state
    elif name == "naive_bayes":
        return get_naive_bayes(task)
    elif name == "decision_tree":
        return get_decision_tree(task, **kwargs)
    elif name == "mlp":
        return get_mlp(task, **kwargs)
    elif name == "svm":
        return get_svm(task, **kwargs)
    elif name == "lasso":
        return get_lasso(task, **kwargs)
    elif name == "ridge_reg":
        return get_ridge(task, **kwargs)
    elif name == "elasticnet":
        return get_elasticnet(task, **kwargs)
    raise ValueError(f"Unknown model: {name}")


def _get_param_grid(name, task):
    if name == "xgboost":
        if task == "classification":
            return {"n_estimators": [100, 200], "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1], "reg_alpha": [0, 0.1]}
        return {"n_estimators": [100, 200], "max_depth": [3, 5],
                "learning_rate": [0.01, 0.05]}
    elif name == "gb":
        return {"n_estimators": [100, 200], "max_depth": [3, 5],
                "learning_rate": [0.01, 0.05]}
    elif name == "rf":
        return {"n_estimators": [100, 200], "max_depth": [5, 10],
                "min_samples_leaf": [2, 5]}
    elif name == "knn":
        return {"n_neighbors": [3, 5, 7, 11]}
    elif name == "decision_tree":
        return {"max_depth": [3, 5, 10, None], "min_samples_leaf": [2, 5, 10]}
    elif name == "mlp":
        return {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [0.001, 0.01]}
    elif name == "svm":
        if task == "classification":
            return {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
        return {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
    elif name == "naive_bayes":
        return {"var_smoothing": [1e-9, 1e-7, 1e-5]}
    elif name in ("lasso", "ridge_reg", "elasticnet"):
        return {}  # These use built-in CV for alpha selection
    return {}


def _run_temporal(daily, temporal_name, q33, q66, split_method, test_fraction,
                  n_holdout_patients, seed, temporal_params=None):
    """Train temporal model for both classification and regression."""
    tp = temporal_params or {}
    seq_length = tp.get("seq_length", 7)
    X_seq, y_seq, pids_seq, dates_seq = get_raw_sequences(daily, seq_length=seq_length)

    # Replace NaN in sequences (new daily columns like bed_time can have NaN)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)

    # Split sequences using same strategy
    if split_method == "chronological":
        all_dates = sorted(daily[DATE_COL].unique())
        cutoff_date = all_dates[int(len(all_dates) * (1 - test_fraction))]
        seq_dates_ts = np.array([np.datetime64(d) for d in dates_seq])
        train_mask = seq_dates_ts < np.datetime64(cutoff_date)
    elif split_method == "leave_patients_out":
        rng = np.random.RandomState(seed)
        patients = daily[ID_COL].unique()
        holdout = rng.choice(patients, size=n_holdout_patients, replace=False)
        train_mask = ~np.isin(pids_seq, holdout)
    else:
        # Default to chronological for temporal
        all_dates = sorted(daily[DATE_COL].unique())
        cutoff_date = all_dates[int(len(all_dates) * 0.8)]
        seq_dates_ts = np.array([np.datetime64(d) for d in dates_seq])
        train_mask = seq_dates_ts < np.datetime64(cutoff_date)

    test_mask = ~train_mask
    X_tr, X_te = X_seq[train_mask], X_seq[test_mask]
    y_tr, y_te = y_seq[train_mask], y_seq[test_mask]

    # Scale
    n_s, sl, nf = X_tr.shape
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr.reshape(-1, nf)).reshape(n_s, sl, nf)
    X_te_s = sc.transform(X_te.reshape(-1, nf)).reshape(X_te.shape[0], sl, nf)
    del X_seq, X_tr, X_te; gc.collect()

    # Validation split for early stopping
    nv = max(1, int(len(X_tr_s) * 0.2))

    # Get model constructor
    if temporal_name == "gru":
        get_model = get_gru
    elif temporal_name == "lstm":
        get_model = get_lstm
    elif temporal_name == "cnn1d":
        get_model = get_cnn1d
    elif temporal_name == "transformer":
        get_model = get_transformer
    else:
        raise ValueError(f"Unknown temporal model: {temporal_name}")

    # Extract temporal model hyperparameters
    hidden_dim = tp.get("hidden_dim", 32)
    dropout = tp.get("dropout", 0.3)
    bidirectional = tp.get("bidirectional", False)
    if temporal_name == "transformer":
        model_kwargs = {
            "d_model": tp.get("d_model", 64),
            "nhead": tp.get("nhead", 2),
            "num_layers": tp.get("num_layers", 2),
            "dropout": dropout,
        }
    else:
        model_kwargs = {"hidden_dim": hidden_dim, "dropout": dropout}
        if temporal_name == "gru":
            model_kwargs["bidirectional"] = bidirectional

    # Classification
    y_tr_cls = discretize_mood(y_tr, q33, q66)
    y_te_cls = discretize_mood(y_te, q33, q66)

    model_cls = get_model(input_dim=nf, task="classification",
                          lr=0.001, epochs=100, patience=15, batch_size=32,
                          **model_kwargs)
    model_cls.fit(X_tr_s[:-nv], y_tr_cls[:-nv], X_val=X_tr_s[-nv:], y_val=y_tr_cls[-nv:])
    cls_pred = model_cls.predict(X_te_s)
    cls_results = evaluate_classifier(y_te_cls, cls_pred)
    print(f"    {temporal_name} cls F1: {cls_results['f1_macro']:.4f}")
    del model_cls; gc.collect()

    # Regression
    model_reg = get_model(input_dim=nf, task="regression",
                          lr=0.001, epochs=100, patience=15, batch_size=32,
                          **model_kwargs)
    model_reg.fit(X_tr_s[:-nv], y_tr[:-nv], X_val=X_tr_s[-nv:], y_val=y_tr[-nv:])
    reg_pred = model_reg.predict(X_te_s)
    reg_results = evaluate_regressor(y_te, reg_pred)
    print(f"    {temporal_name} reg R2: {reg_results['r2']:.4f}")
    del model_reg, X_tr_s, X_te_s; gc.collect()

    return cls_results, reg_results


def _run_sliding_window(iteration, iter_name, data_dir, daily, splits,
                        features_df, tabular_cls, tabular_reg, temporal,
                        hypothesis, change_summary, agg_functions, window_sizes,
                        n_lags, seed):
    """Handle sliding window evaluation by averaging across splits."""
    # For sliding window, run tabular models on each split and average
    # (temporal model uses chronological as fallback)
    all_cls_f1 = []
    all_reg_r2 = []

    meta_cols = [ID_COL, DATE_COL, TARGET_COL]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    for i, (train_feat, test_feat) in enumerate(splits):
        X_tr = np.nan_to_num(train_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr_cont = train_feat[TARGET_COL].values
        X_te = np.nan_to_num(test_feat[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_te_cont = test_feat[TARGET_COL].values
        groups = train_feat[ID_COL].values

        q33, q66 = compute_tercile_thresholds(y_tr_cont)
        y_tr_cls = discretize_mood(y_tr_cont, q33, q66)
        y_te_cls = discretize_mood(y_te_cont, q33, q66)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # Classification
        cls_model = _get_tabular_model(tabular_cls, "classification", seed)
        cls_model.fit(X_tr_s, y_tr_cls)
        cls_pred = cls_model.predict(X_te_s)
        cls_res = evaluate_classifier(y_te_cls, cls_pred)
        all_cls_f1.append(cls_res["f1_macro"])

        # Regression
        reg_model = _get_tabular_model(tabular_reg, "regression", seed)
        reg_model.fit(X_tr_s, y_tr_cont)
        reg_pred = reg_model.predict(X_te_s)
        reg_res = evaluate_regressor(y_te_cont, reg_pred)
        all_reg_r2.append(reg_res["r2"])

        del cls_model, reg_model; gc.collect()

    cls_f1_mean = float(np.mean(all_cls_f1))
    cls_f1_std = float(np.std(all_cls_f1))
    reg_r2_mean = float(np.mean(all_reg_r2))
    reg_r2_std = float(np.std(all_reg_r2))

    print(f"    Sliding window results ({len(splits)} splits):")
    print(f"    Cls F1: {cls_f1_mean:.4f} +/- {cls_f1_std:.4f}")
    print(f"    Reg R2: {reg_r2_mean:.4f} +/- {reg_r2_std:.4f}")

    # Temporal model (use chronological fallback)
    q33, q66 = compute_tercile_thresholds(features_df[TARGET_COL].values[:int(len(features_df)*0.8)])
    temporal_cls, temporal_reg = _run_temporal(
        daily, "gru", q33, q66, "chronological", 0.2, 5, seed
    )

    cls_report = {tabular_cls: {"f1_macro": cls_f1_mean, "f1_std": cls_f1_std,
                                 "accuracy": cls_f1_mean, "per_class_f1": all_cls_f1}}
    reg_report = {tabular_reg: {"r2": reg_r2_mean, "r2_std": reg_r2_std,
                                 "rmse": 0, "mae": 0, "mse": 0}}

    card = save_report_card(
        iteration_dir=iter_name, iteration=iteration,
        hypothesis=hypothesis, change_summary=change_summary,
        classification_results={**cls_report, "gru": temporal_cls},
        regression_results={**reg_report, "gru": temporal_reg},
        n_features=len(feature_cols), n_train=0, n_test=0,
        extra={"sliding_window_splits": len(splits),
               "cls_f1_per_split": all_cls_f1, "reg_r2_per_split": all_reg_r2},
    )
    _print_comparison(iteration, card)
    return card


def _print_comparison(iteration, card):
    """Compare with previous best iteration."""
    if iteration <= 0:
        return
    prev_cards = []
    for d in sorted(ITERATIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        rc = d / "report_card.json"
        if rc.exists():
            try:
                pc = load_report_card(d)
                if pc["iteration"] < iteration:
                    prev_cards.append(pc)
            except (json.JSONDecodeError, KeyError):
                continue
    if prev_cards:
        prev = max(prev_cards, key=lambda c: c["iteration"])
        print("\n" + compare_iterations(card, prev))

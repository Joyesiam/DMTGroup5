# Iteration 7: Baseline v2 (Full Pipeline with Saved Data)

## Diagnosis
Previous iterations (0-6) never saved intermediate data and used hardcoded
cleaning parameters. This iteration establishes the v2 baseline with:
- Same best config from v1 (IQR*3 + ffill + 7-day window + volatility + interactions)
- But now using the parameterized pipeline that saves all intermediate CSVs
- Also generates EDA analysis for Task 1A

## Hypothesis
Using the same parameters as the best v1 config through the new pipeline
should produce similar results (F1 ~0.48-0.57, R2 ~0.27). The main value
is establishing traceable data artifacts.

## Change
- First iteration using shared/pipeline.py orchestrator
- Saves daily_cleaned.csv, features_train.csv, features_test.csv, pipeline_config.json
- Same cleaning + features + models as v1 best (iter_02 features + iter_04 models)

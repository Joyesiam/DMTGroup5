# Assignment 2 - Personalized Hotel Ranking

A from-scratch learning-to-rank pipeline for the Expedia / VU DM 2014 in-class
Kaggle competition (ICDM 2013 dataset). Six numbered notebooks under
`notebooks/` take the raw five million row CSVs through exploration, cleaning,
feature engineering, Optuna-tuned modelling, locked-holdout evaluation, and
final submission. Each notebook is self-contained and is meant to be read
top-to-bottom; decision rationale lives in the markdown cells of the notebooks
themselves.

## Final result

| Metric | Value |
|---|---|
| Locked holdout NDCG@5 (19,980 searches) | **0.41521** |
| Kaggle public NDCG@5 | **0.41706** |

The submission is a weighted blend (`w_xgb = 0.5`) of a three-seed LightGBM
LambdaRank ensemble and a single XGBoost `rank:ndcg` model, both tuned with a
50-trial Optuna TPE study on a locked validation slice. Final score per
`(srch_id, prop_id)` is then optionally adjusted by a val-tuned popularity-tertile
exposure rerank (Singh and Joachims 2018) for the bias-mitigation deliverable.

## Notebooks

| Notebook | What happens here |
|---|---|
| `01_eda.ipynb` | First look at the data, target structure, missingness, group sizes, multi-panel EDA figures |
| `02_data_cleaning.ipynb` | Missing-value strategy, `log1p` price transform, competitor aggregates, dtype optimisation |
| `03_feature_engineering.ipynb` | Five-block feature ablation, within-search relativisations, leakage-safe 5-fold OOF target encoding on `prop_id`, `srch_destination_id`, and the pair |
| `04_modelling.ipynb` | LightGBM LambdaRank and XGBoost `rank:ndcg` with 50-trial Optuna TPE tuning, three-seed ensemble, blend weight sweep on val |
| `05_evaluation.ipynb` | Holdout NDCG@5 per predictor, segment analysis, worst-case characterisation, popularity-tertile bias audit, val-tuned inference-time rerank |
| `06_final_pipeline_and_submission.ipynb` | End-to-end full-train refit and write of `results/submission.csv` |

## Deliverables

- `results/submission.csv` - the Kaggle ranking submission
- `results/figures/` - all figures referenced in the report
- `data/processed/*.json` and `*.csv` - small artefacts that document the pipeline outcome (Optuna best parameters, evaluation summary, realised Kaggle score)
- `report/` - written deliverable (PDF compiled from the Springer Nature template on Overleaf)

## Reproducing the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw competition CSVs in `data/raw/`:

- `data/raw/training_set_VU_DM.csv`
- `data/raw/test_set_VU_DM.csv`

Then execute the notebooks in order:

```bash
for nb in notebooks/0{1..6}*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

The full pipeline runs end-to-end in roughly three hours on a 64 GB laptop with
`n_jobs = 4` and `gc.collect()` between model fits. Notebook 04 is the heaviest
stage because of the 100 Optuna trials (50 per model).

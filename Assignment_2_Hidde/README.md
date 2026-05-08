# Assignment 2 — Personalized Hotel Ranking

A from-scratch attempt at the Expedia / VU DM 2014 personalized hotel ranking problem.
Everything lives in the six numbered notebooks under `notebooks/`. Each notebook is
self-contained and is meant to be read top-to-bottom as a record of how the project
was actually built, including the experiments that did not pan out.

## Contents

| Notebook | What happens here |
|---|---|
| `01_eda.ipynb` | First look at the data, target structure, missingness, group sizes |
| `02_data_cleaning.ipynb` | Missing-value strategy, outlier handling, dtype optimisation |
| `03_feature_engineering.ipynb` | Raw feature selection, leakage-safe historical priors, custom features |
| `04_modelling.ipynb` | LightGBM ranker, XGBoost rank baseline, hyperparameter exploration, ensemble |
| `05_evaluation.ipynb` | Holdout NDCG@5, segment analysis, error analysis, position-bias check |
| `06_final_pipeline_and_submission.ipynb` | End-to-end run from raw CSVs to `results/submission.csv` |

Per-notebook decision logs are written to `notes/`. Generated figures go to
`results/figures/`. The final ranked submission ends up at `results/submission.csv`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw competition CSVs in `data/raw/`:

- `data/raw/training_set_VU_DM.csv`
- `data/raw/test_set_VU_DM.csv`

Then run notebooks in order. The full pipeline takes a few hours on a single machine;
notebook 06 alone trains three seeds on the full training set.

## Reproducing the submission

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_modelling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_final_pipeline_and_submission.ipynb
```

The final submission file is `results/submission.csv` with header `srch_id,prop_id`.

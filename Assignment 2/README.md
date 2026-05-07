# Assignment 2: Personalized Hotel Search Ranking (Expedia 2013)

Course: Data Mining Techniques (X_400111), VU Amsterdam, 2026.

Team: VU-DM-2026-Group-5
- Hendrico Burger (2758955)
- Hidde Franke (2840386)
- Joanne Pals (2864742)

## Task

Learning-to-rank on the Expedia 2013 dataset. For each `srch_id` we
order the candidate `prop_id` values so that booked hotels are ranked
above clicked-only and non-interacted hotels. The competition metric
is NDCG@5 with relevance grades 5 (booked), 1 (clicked), 0 (otherwise).

## Final submission

- File: `results/submit_iter07_prior_3seed.csv`
- Kaggle public NDCG@5: 0.39690
- Method: 3-seed LightGBM LambdaMART on a 49-feature anchor plus 12
  leakage-safe historical priors over `prop_id`,
  `srch_destination_id`, and `prop_id x srch_destination_id`.
  Score-averaged over seeds.

## How to reproduce

1. Create the environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place the raw data files in `data/raw/`:
   - `training_set_VU_DM.csv`
   - `test_set_VU_DM.csv`
3. Convert raw CSVs to processed parquet under
   `data/processed/{train_clean,test_clean}.parquet` (any standard
   pandas + pyarrow conversion script will work).
4. Run the pipeline:
   ```bash
   python src/pipeline.py --train --predict
   ```
   The submission CSV is written to `results/submit_final.csv` with
   header `srch_id,prop_id`, ordered from highest predicted relevance
   to lowest within each `srch_id`.

To validate the included submission CSV without retraining:
```bash
python -m src.pipeline --validate results/submit_iter07_prior_3seed.csv
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_eda.ipynb` | Dataset description, distributions, metric definition |
| `notebooks/02_feature_engineering.ipynb` | Anchor features and historical priors with leakage analysis |
| `notebooks/03_modeling.ipynb` | LightGBM LambdaMART, ensembling, second technique (XGBoost) |
| `notebooks/04_evaluation_bias_deployment.ipynb` | Holdout NDCG@5, public score, bias detection and mitigation |
| `notebooks/05_final_pipeline_and_submission.ipynb` | End-to-end reproduction and submission validation |

## Compute

Trained on a MacBook Pro M4 (14 cores, 36 GB RAM). All multi-threaded
libraries are pinned to safe thread counts; we do not use `n_jobs=-1`
or `num_threads=-1`. Memory usage stays well below 24 GB.

## Constraints

- Original 2013 Kaggle answer files are not used.
- External data sources are not used in the final pipeline.
- Public leaderboard scores are reported only after model selection.

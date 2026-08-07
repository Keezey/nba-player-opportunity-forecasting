# Reproducibility Guide

This guide rebuilds the historical data, leakage-safe modeling rows,
experiments, and V2.3 production artifacts described in the research report.
Run every command from the repository root.

Generated data, model binaries, manifests, tuning trials, and experiment
reports are local artifacts and are intentionally excluded from Git.

## 1. Environment

The finalized environment used Python 3.13.2.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\activate`.

## 2. Historical Warehouse

An internet connection is required for uncached NBA Stats requests.

```bash
python -m scripts.build_historical_store \
  2021-22 2022-23 2023-24 2024-25 2025-26

python -m scripts.historical_store_status
```

Successful responses are cached under `data/raw/cache`. If an endpoint times
out, rerun the same command; completed responses will be reused. Do not pass
`--refresh` when resuming unless every response should be downloaded again.

Expected season partitions follow this pattern:

```text
data/processed/player_games/season=2021-22/player_games.parquet
data/processed/player_games/season=2021-22/manifest.json
```

The five completed partitions should contain 131,279 player-game rows in
total, with 1,230 regular-season games in each season.

## 3. Reproducible Target Pools

First recreate the original stratified 90-player anchor and its eligibility
audit. The eligible universe uses player-season participation and role data
from 2021-22 through 2023-24.

```bash
python -m scripts.build_v2_player_pool \
  data/processed/v2_target_pool_90.csv \
  2021-22 2022-23 2023-24 \
  --players-per-stratum 10 \
  --seed 42
```

Then construct the nested 30, 60, 90, 150, and 250-player pools:

```bash
python -m scripts.build_v2_nested_pools \
  data/processed/v2_target_pool_90_eligible.csv \
  data/processed/v2_1_pools \
  --anchor-pool data/processed/v2_target_pool_90.csv \
  --sizes 30 60 90 150 250 \
  --prefix v2_1_target_pool \
  --seed 42
```

The target pool determines which player-games become modeling rows. Similar
player searches continue to use the full eligible league population available
before each target game.

## 4. Leakage-Safe Modeling Rows

Build the earlier-season development rows and the later 2025-26 rows for the
250-player pool:

```bash
python -m scripts.build_v2_training_dataset \
  data/processed/v2_training_90.parquet \
  2021-10-19 \
  2025-04-13 \
  --player-pool data/processed/v2_target_pool_90.csv \
  --workers 4

python -m scripts.build_v2_training_dataset \
  data/processed/v2_1_training_250.parquet \
  2021-10-19 \
  2025-04-13 \
  --player-pool data/processed/v2_1_pools/v2_1_target_pool_250.csv \
  --workers 4

python -m scripts.build_v2_training_dataset \
  data/processed/v2_1_test_250_2025-26.parquet \
  2025-10-21 \
  2026-04-12 \
  --player-pool data/processed/v2_1_pools/v2_1_target_pool_250.csv \
  --workers 4
```

Each feature row uses the previous calendar day as its cutoff. Actual outcomes
are attached only after the pregame row passes validation. Skipped-game reasons
are written beside each Parquet output.

Combine the two periods for rolling experiments:

```bash
python -m scripts.merge_v2_training_datasets \
  data/processed/v2_1_all_250.parquet \
  data/processed/v2_1_training_250.parquet \
  data/processed/v2_1_test_250_2025-26.parquet
```

The expected combined dataset contains 59,156 eligible rows for 250 target
players across 5,788 distinct games.

## 5. Research Experiments

Run the experiments in order. Each stage freezes a decision used by the next.

### 5.0 Fixed development parameters

The pool-size experiment holds model and trust parameters fixed so that it
isolates the effect of adding target players. Recreate the earlier tuning
subset and the three benchmark parameter files:

```bash
python -c "import pandas as pd; d=pd.read_parquet('data/processed/v2_training_90.parquet'); d[d['season'].isin(['2021-22','2022-23','2023-24'])].to_parquet('data/processed/v2_training_90_tuning.parquet', index=False)"

python -m scripts.tune_v2_parameters \
  data/processed/v2_training_90_tuning.parquet \
  reports/v2_parameters_90.json \
  --model-type hist_gradient_boosting \
  --n-iter 20 \
  --n-splits 3

python -m scripts.tune_v2_parameters \
  data/processed/v2_training_90_tuning.parquet \
  reports/v2_parameters_90_elastic.json \
  --model-type elastic_net \
  --n-iter 20 \
  --n-splits 3

python -m scripts.tune_v2_parameters \
  data/processed/v2_training_90_tuning.parquet \
  reports/v2_parameters_90_neural.json \
  --model-type pytorch_multitask \
  --n-iter 6 \
  --n-splits 3
```

### 5.1 Pool-size learning curve

```bash
python -m scripts.run_v2_pool_size_experiment \
  data/processed/v2_1_training_250.parquet \
  data/processed/v2_1_test_250_2025-26.parquet \
  reports/v2_1_pool_size_experiment \
  --pool-dir data/processed/v2_1_pools \
  --pool-prefix v2_1_target_pool \
  --pool-sizes 30 60 90 150 250 \
  --elastic-parameters reports/v2_parameters_90_elastic.json \
  --boosting-parameters reports/v2_parameters_90.json \
  --neural-parameters reports/v2_parameters_90_neural.json \
  --trust-parameters reports/v2_parameters_90_elastic.json \
  --compute-threads 2
```

### 5.2 Rolling outer-season model comparison

```bash
python -m scripts.run_v2_rolling_season_experiment \
  data/processed/v2_1_all_250.parquet \
  reports/v2_1_rolling_seasons \
  --test-seasons 2023-24 2024-25 2025-26 \
  --models elastic_net pytorch_multitask \
  --trust-n-iter 20 \
  --elastic-n-iter 20 \
  --neural-n-iter 6 \
  --n-splits 3 \
  --compute-threads 2 \
  --neural-ensemble-size 5
```

### 5.3 Normal-minutes conditional experiment

```bash
python -m scripts.run_v2_regular_minutes_experiment \
  data/processed/v2_1_all_250.parquet \
  reports/v2_1_regular_minutes \
  --test-seasons 2023-24 2024-25 2025-26 \
  --models elastic_net \
  --training-cohorts all regular \
  --absolute-tolerance 3 \
  --relative-tolerance 0.15 \
  --trust-n-iter 20 \
  --elastic-n-iter 20 \
  --n-splits 3 \
  --compute-threads 2
```

The `regular` label uses actual minutes only to define research cohorts. It is
never an inference feature.

### 5.4 Robust multiplier bounds

```bash
python -m scripts.run_v2_multiplier_bounds_experiment \
  data/processed/v2_1_all_250.parquet \
  reports/v2_2_multiplier_bounds \
  --test-seasons 2023-24 2024-25 2025-26 \
  --bound-strategies unbounded fixed_0.50_1.50 \
    fixed_0.67_1.50 training_p01_p99 \
  --training-cohorts all regular \
  --models elastic_net \
  --trust-n-iter 20 \
  --elastic-n-iter 20 \
  --n-splits 3 \
  --compute-threads 2
```

### 5.5 Rebound-conversion experiment

```bash
python -m scripts.run_v2_conversion_experiment \
  data/processed/v2_1_all_250.parquet \
  reports/v2_2_multiplier_bounds/predictions.parquet \
  reports/v2_3_rebound_conversion \
  --test-seasons 2023-24 2024-25 2025-26 \
  --training-cohorts all regular \
  --base-model-type elastic_net \
  --base-bound-strategy fixed_0.50_1.50 \
  --elastic-n-iter 20 \
  --n-splits 3 \
  --bootstrap-samples 2000 \
  --compute-threads 2
```

## 6. Final Production Artifacts

Extract the settings selected in the rolling 2025-26 fold. Those settings were
chosen using only seasons through 2024-25.

```bash
python -m scripts.prepare_v2_production_parameters \
  reports/v2_2_multiplier_bounds/parameters.json \
  reports/v2_3_rebound_conversion/parameters.json \
  reports/v2_3_production
```

Fit the default all-games artifact:

```bash
python -m scripts.finalize_v2_release \
  data/processed/v2_1_training_250.parquet \
  data/processed/v2_1_test_250_2025-26.parquet \
  data/processed/releases/v2_3_250_all_games_training.parquet \
  models/v2_3_250_all_games.joblib \
  --parameters-json reports/v2_3_production/parameters_all.json \
  --manifest models/v2_3_250_all_games_manifest.json \
  --release-version 2.3.0 \
  --evaluation-report reports/v2_3_rebound_conversion/report.md
```

Fit the normal-minutes conditional artifact:

```bash
python -m scripts.finalize_v2_release \
  data/processed/v2_1_training_250.parquet \
  data/processed/v2_1_test_250_2025-26.parquet \
  data/processed/releases/v2_3_250_normal_minutes_training.parquet \
  models/v2_3_250_normal_minutes.joblib \
  --parameters-json reports/v2_3_production/parameters_regular.json \
  --manifest models/v2_3_250_normal_minutes_manifest.json \
  --release-version 2.3.0 \
  --evaluation-report reports/v2_3_rebound_conversion/report.md
```

The final fitting step includes 2025-26 outcomes. Therefore, 2025-26 is
validation evidence for model selection but is not an untouched test set for
the retrained production artifacts.

## 7. Inference Smoke Test

```bash
python -m scripts.predict_player_v2 \
  models/v2_3_250_all_games.joblib \
  "Amen Thompson" \
  2026-04-07
```

Use `models/v2_3_250_normal_minutes.joblib` only when the question explicitly
assumes that the player receives approximately the expected workload.

## 8. Figure Regeneration

```bash
python -m pip install -r requirements-report.txt
python -m scripts.generate_research_report_figures
```

Curated figures are written to `docs/figures`. The script reads local generated
experiment outputs, which must exist before figure regeneration.

## 9. External Dependencies

The project uses NBA Stats through the unofficial `nba_api` wrapper. Endpoint
availability, schemas, and request behavior can change. Users are responsible
for following the data provider's applicable terms.

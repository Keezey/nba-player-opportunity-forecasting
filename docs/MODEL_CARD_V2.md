# NBA Player Opportunity Forecasting V2.3 Model Card

## Release Summary

- **Release:** V2.3.0
- **Primary model:** Elastic Net residual corrections
- **Outputs:** field-goal attempts, rebound chances, and rebounds
- **Target population:** 250 reproducibly selected NBA players
- **Historical range:** 2021-22 through 2025-26 regular seasons
- **Production rows:** 59,156 eligible player-games
- **Conditional production rows:** 34,891 normal-minute player-games
- **Validation design:** nested chronological tests on 2023-24, 2024-25,
  and 2025-26

V2 preserves V1's basketball structure. V1 establishes the player's recent
baseline and an opponent adjustment from statistically similar players. V2
learns corrections to V1's FGA and rebound-chance projections. A separate
chance-weighted Elastic Net estimates rebound conversion, and final rebounds
are corrected rebound chances multiplied by corrected conversion.

## Available Artifacts

Two final model bundles and their manifests are distributed under
`artifacts/v2.3.0/`:

- `v2_3_250_all_games.joblib`: default unconditional model.
- `v2_3_250_normal_minutes.joblib`: conditional model trained only on
  games in which actual minutes remained near the pregame expectation.

The normal-minutes model does not predict whether normal minutes will occur.
Use it only when the projection question explicitly assumes approximately the
expected workload.

## Intended Use

Appropriate uses include:

- Historical NBA regular-season player-game analysis.
- Pregame FGA, rebound-chance, and rebound projections when recent tracking
  data is available.
- Educational demonstration of API ingestion, leakage-safe feature
  engineering, chronological validation, residual learning, and model release.

The model is not a guarantee of player performance, an injury-news system, or
a complete wagering system. Users are responsible for decisions made from its
output.

## Inputs

Every predictive feature is available before the target game:

- Recent qualifying games, minutes, FGA/minute, rebound chances/minute, and
  rebound conversion.
- Usage, touches/minute, listed position, and starter rate.
- Similar-player distances and sample counts.
- Raw and sample-shrunken opponent multipliers.
- V1 minutes, FGA, rebound-chance, and rebound projections.

Actual outcomes are attached only after the pregame feature row passes the
cutoff check. `as_of_date` must be strictly earlier than `game_date`.

## Data

The local historical store contains regular-season box-score, advanced, and
tracking data fetched through the unofficial `nba_api` wrapper. Raw NBA data,
cached responses, processed Parquet datasets, intermediate models, and
generated experiment outputs are not distributed in the Git repository.

The final fitting data contains 59,156 eligible rows for 250 players across
5,788 games. Eligibility requires enough prior qualifying tracking games to
construct a pregame baseline. Reported metrics therefore apply to eligible
player-games rather than every NBA appearance.

## Validation Design

Three outer test seasons were kept later than their corresponding training
rows:

| Outer test season | Earlier training rows | Test rows | Test players |
| --- | ---: | ---: | ---: |
| 2023-24 | 25,622 | 13,088 | 240 |
| 2024-25 | 38,710 | 11,099 | 223 |
| 2025-26 | 49,809 | 9,347 | 196 |

Trust and model parameters were retuned only inside earlier seasons for each
outer fold. Equal-season summaries give each held-out season one vote.

## Opportunity-Model Results

The selected Elastic Net residual model improved V1 in every held-out season.

| Output | Mean V1 MAE | Mean V2 MAE | Relative improvement |
| --- | ---: | ---: | ---: |
| FGA | 3.136 | 3.062 | 2.36% |
| Rebound chances | 3.063 | 2.934 | 4.22% |
| Rebounds with baseline conversion | 2.122 | 2.066 | 2.64% |

A PyTorch multitask network produced similar MAE but materially worse RMSE in
some folds and highly correlated errors. Elastic Net was selected for its
stability, simplicity, interpretability, and lower operational cost.

## Robust Multiplier Bounds

Observed matchup multipliers are clipped to `0.50-1.50` before sample-size
shrinkage. Average MAE changed very little, while the worst rebound-chance error
fell from `55.8` to `22.4` and the worst rebound error fell from `30.2` to
`20.4`. Only about `0.19%` of rebound-chance rows were clipped.

## Rebound-Conversion Results

With robust opportunity forecasts frozen, the learned conversion correction
was compared with the recent player ratio and empirical shrinkage.

| Evaluation | Current conversion MAE | Learned conversion MAE | Improvement |
| --- | ---: | ---: | ---: |
| Complete held-out seasons | 2.0645 | 2.0467 | 0.86% |
| Normal-minute games, matching training | 1.9876 | 1.9631 | 1.23% |

The learned method improved all three test seasons. A paired player-cluster
bootstrap gave a 95% interval of approximately `0.0132-0.0225` rebounds for
the complete-season MAE gain and `0.0189-0.0303` for the conditional gain.

## Production Fitting

After the rolling results and model choices were frozen, both production
artifacts were fit on all available seasons through 2025-26. Consequently,
2025-26 is validation evidence for model selection, not an untouched test set
for the final retrained artifact.

Each local manifest records:

- Training cohort, seasons, rows, players, and dates.
- Opportunity and conversion feature schemas.
- Frozen trust, multiplier-bound, and model parameters.
- Conversion bounds learned from the training cohort.
- Runtime package versions and SHA-256 hashes.

## Known Limitations

- Expected minutes are based on recent role and do not ingest real-time injury
  or lineup news.
- No calibrated prediction intervals are currently produced.
- Early-season, returning, or low-minute players may fail the five-game
  eligibility rule.
- Opponent effects rely on a small recent sample and remain noisy despite
  similarity weighting, bounds, and shrinkage.
- Public tracking availability and endpoint behavior can change.
- The 250-player target pool is broad but is not every possible NBA player.
- Results cover regular-season games; playoffs are outside the evaluated scope.

## Monitoring Recommendations

For each new season, report coverage, MAE, RMSE, bias, 95th/99th-percentile
absolute error, and results by player, position, minutes stability, and sample
count. Retraining should remain chronological, and parameters should be changed
only after a new future-period evaluation.

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete build and
validation sequence. Package versions are pinned in `requirements.txt`, and
the offline test suite runs with:

```bash
python -m pytest -q
```

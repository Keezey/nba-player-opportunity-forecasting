# NBA API Endpoint Decisions

The current model predicts FGA, rebound chances, and rebounds. Assist-related
data is intentionally outside this version.

## Automatic Prediction

`PlayerGameLog` resolves the target player's exact regular-season game and
opponent from the supplied player and date. The date also determines the NBA
season. The prediction cutoff is always the preceding day.

League-wide comparison profiles use date-filtered dashboard endpoints:

- `LeagueDashPlayerStats`, `Base`: games, minutes, FGA, and rebounds.
- `LeagueDashPlayerStats`, `Advanced`: usage percentage.
- `LeagueDashPlayerStats`, `StarterBench=Starters`: recent starts divided by
  total games to create `starter_rate`.
- `LeagueDashPlayerStats`, position filters `G`, `F`, and `C`: broad listed
  position memberships. Multiple memberships become labels such as `G-F`.
- `LeagueDashPtStats`, `Rebounding`: rebound chances and tracking minutes.
- `LeagueDashPtStats`, `Possessions`: touches and tracking minutes.

Opponent matchup rates use the same `Base` and `Rebounding` dashboards with
`OpponentTeamID` populated. This provides each comparison player's FGA/min,
rebound-chances/min, and number of games against that opponent in the lookback
window.

## Target Baseline

The target player's prior game rows come from:

- `PlayerGameLog`: minutes, FGA, rebounds, date, and opponent.
- `BoxScorePlayerTrackV3`: `reboundChancesTotal` and touches.

The baseline keeps games near the player's normal workload, requires at least
five qualifying tracking games by default, and computes rebound conversion as
paired total rebounds divided by paired total rebound chances.

## Saved Backtest Datasets

`BoxScoreAdvancedV3` supplies per-game `usagePercentage` when building saved
datasets. Its position field and the equivalent V3 tracking position field also
provide the game-level starter signal: a populated starting position means the
player started.

## Caching

Every request is cached as Parquet under `data/raw/cache`. Historical endpoint
parameters are hashed into deterministic filenames. `--refresh` bypasses these
files when a response needs to be fetched again.

## Supported Range

This version requires public player tracking, so its automatic workflow starts
with 2013-14 and supports completed regular seasons through 2025-26. A game may
still be unprojectable near the beginning of a season if the target has fewer
than the configured minimum number of prior qualifying games.

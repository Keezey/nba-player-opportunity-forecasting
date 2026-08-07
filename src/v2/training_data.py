"""Historical training-dataset construction for version two.

Each target game is sent through the same automatic V1 workflow used by the
single-game command. Successful predictions become leakage-safe training rows;
expected unavailable games are retained separately with their failure reason.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import pandas as pd

from ..dataset import build_player_game_df, clean_gamelog
from ..fetch_nba import NBADataFetchError, fetch_player_gamelog
from ..live_data import resolve_player, season_from_game_date
from ..predict import PredictionUnavailableError, predict_player_game
from .features import TRAINING_ROW_COLUMNS, FeatureLeakageError, build_training_row


SKIPPED_ROW_COLUMNS = [
    "player_id",
    "player_name",
    "game_id",
    "game_date",
    "season",
    "opp_abbr",
    "error_type",
    "message",
]


@dataclass(frozen=True)
class TrainingDatasetResult:
    """Successful training rows and games that could not produce a prediction."""

    rows: pd.DataFrame
    skipped: pd.DataFrame

    @property
    def games_found(self) -> int:
        return len(self.rows) + len(self.skipped)

    @property
    def predictions_built(self) -> int:
        return len(self.rows)


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=TRAINING_ROW_COLUMNS)


def _empty_skipped() -> pd.DataFrame:
    return pd.DataFrame(columns=SKIPPED_ROW_COLUMNS)


def _season_start_year(season: str) -> int:
    try:
        return int(str(season).split("-", maxsplit=1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid NBA season string: {season!r}") from exc


def seasons_for_date_range(start_date, end_date) -> list[str]:
    """Return every supported NBA season touched by an inclusive date range."""
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date.")

    first = season_from_game_date(start)
    last = season_from_game_date(end)
    first_year = _season_start_year(first)
    last_year = _season_start_year(last)
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(first_year, last_year + 1)]


def _target_games(
    player_id: int,
    *,
    season: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    refresh: bool,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    from ..historical_store import load_season_player_games, season_store_exists

    if not refresh and season_store_exists(season, store_dir):
        games = load_season_player_games(
            season,
            player_ids=[int(player_id)],
            date_from=start_date,
            date_to=end_date,
            store_dir=store_dir,
        )
        if not games.empty:
            games["season"] = season
        return games.sort_values("game_date")

    raw = fetch_player_gamelog(
        int(player_id),
        season,
        "Regular Season",
        refresh=refresh,
    )
    games = clean_gamelog(raw)
    if games.empty:
        return games

    games = games[
        (games["game_date"].dt.normalize() >= start_date)
        & (games["game_date"].dt.normalize() <= end_date)
    ].copy()
    games["season"] = season
    return games.sort_values("game_date")


def _actual_games(
    player_id: int,
    *,
    season: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    refresh: bool,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    actuals = build_player_game_df(
        [int(player_id)],
        season=season,
        season_type="Regular Season",
        date_from=start_date,
        date_to=end_date,
        include_advanced=False,
        refresh=refresh,
        store_dir=store_dir,
    )
    if actuals.empty:
        return actuals
    return actuals.drop_duplicates("game_id", keep="first").set_index("game_id", drop=False)


def _actual_for_game(actuals: pd.DataFrame, game: pd.Series) -> pd.Series:
    game_id = str(game["game_id"])
    if not actuals.empty and game_id in actuals.index:
        actual = actuals.loc[game_id]
        if isinstance(actual, pd.DataFrame):
            actual = actual.iloc[0]
        return actual

    # Box-score labels remain usable when the tracking request failed. The
    # rebound-chance label will be missing and excluded only from that model.
    actual = game.copy()
    actual["reb_chances"] = pd.NA
    return actual


def _skipped_row(
    *,
    player_id: int,
    player_name: str,
    game: pd.Series,
    season: str,
    error: Exception,
) -> dict:
    return {
        "player_id": player_id,
        "player_name": player_name,
        "game_id": game.get("game_id", pd.NA),
        "game_date": pd.to_datetime(game.get("game_date"), errors="coerce"),
        "season": season,
        "opp_abbr": game.get("opp_abbr", pd.NA),
        "error_type": type(error).__name__,
        "message": str(error),
    }


def build_player_training_dataset(
    player_query: str | int,
    start_date,
    end_date,
    *,
    candidate_player_ids: Optional[Iterable[int]] = None,
    top_n_similar: int = 10,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
    feature_weights: Optional[Mapping[str, float]] = None,
    position_weight: float = 0.35,
    require_same_starter_flag: bool = False,
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> TrainingDatasetResult:
    """Build chronological V2 training rows for one player over a date range."""
    player_id, player_name = resolve_player(player_query)
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    seasons = seasons_for_date_range(start, end)

    rows: list[pd.Series] = []
    skipped: list[dict] = []
    expected_errors = (
        PredictionUnavailableError,
        NBADataFetchError,
        FeatureLeakageError,
    )

    for season in seasons:
        games = _target_games(
            player_id,
            season=season,
            start_date=start,
            end_date=end,
            refresh=refresh,
            store_dir=store_dir,
        )
        if games.empty:
            continue
        actuals = _actual_games(
            player_id,
            season=season,
            start_date=start,
            end_date=end,
            refresh=refresh,
            store_dir=store_dir,
        )

        for _, game in games.iterrows():
            game_date = pd.to_datetime(game["game_date"]).normalize()
            try:
                prediction_result = predict_player_game(
                    player_id,
                    game_date,
                    season=season,
                    candidate_player_ids=candidate_player_ids,
                    top_n_similar=top_n_similar,
                    lookback_days=lookback_days,
                    min_games=min_games,
                    min_minutes_ratio=min_minutes_ratio,
                    feature_weights=feature_weights,
                    position_weight=position_weight,
                    require_same_starter_flag=require_same_starter_flag,
                    refresh=refresh,
                    store_dir=store_dir,
                )
                actual = _actual_for_game(actuals, game)
                rows.append(build_training_row(prediction_result, actual))
            except expected_errors as exc:
                skipped.append(
                    _skipped_row(
                        player_id=player_id,
                        player_name=player_name,
                        game=game,
                        season=season,
                        error=exc,
                    )
                )

    row_frame = pd.DataFrame(rows).reindex(columns=TRAINING_ROW_COLUMNS)
    if not row_frame.empty:
        row_frame.sort_values(["game_date", "player_id"], inplace=True)
        row_frame.reset_index(drop=True, inplace=True)

    skipped_frame = pd.DataFrame(skipped).reindex(columns=SKIPPED_ROW_COLUMNS)
    if not skipped_frame.empty:
        skipped_frame.sort_values(["game_date", "player_id"], inplace=True)
        skipped_frame.reset_index(drop=True, inplace=True)

    return TrainingDatasetResult(rows=row_frame, skipped=skipped_frame)


def build_training_dataset(
    player_queries: Iterable[str | int],
    start_date,
    end_date,
    *,
    progress: Callable[[int, int, str | int, int, int], None] | None = None,
    workers: int = 1,
    **kwargs,
) -> TrainingDatasetResult:
    """Combine chronological V2 training rows for multiple target players."""
    if workers <= 0:
        raise ValueError("workers must be positive.")
    queries = list(player_queries)
    all_rows: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    successful_rows = 0
    skipped_rows = 0

    def collect_result(index, player_query, result):
        nonlocal successful_rows, skipped_rows
        if not result.rows.empty:
            all_rows.append(result.rows)
            successful_rows += len(result.rows)
        if not result.skipped.empty:
            all_skipped.append(result.skipped)
            skipped_rows += len(result.skipped)
        if progress is not None:
            progress(
                index,
                len(queries),
                player_query,
                successful_rows,
                skipped_rows,
            )

    if workers == 1:
        for index, player_query in enumerate(queries, start=1):
            result = build_player_training_dataset(
                player_query,
                start_date,
                end_date,
                **kwargs,
            )
            collect_result(index, player_query, result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_queries = {
                executor.submit(
                    build_player_training_dataset,
                    player_query,
                    start_date,
                    end_date,
                    **kwargs,
                ): player_query
                for player_query in queries
            }
            for index, future in enumerate(as_completed(future_queries), start=1):
                player_query = future_queries[future]
                collect_result(index, player_query, future.result())

    rows = pd.concat(all_rows, ignore_index=True) if all_rows else _empty_rows()
    skipped = (
        pd.concat(all_skipped, ignore_index=True) if all_skipped else _empty_skipped()
    )
    if not rows.empty:
        rows.sort_values(["game_date", "player_id"], inplace=True)
        rows.reset_index(drop=True, inplace=True)
    if not skipped.empty:
        skipped.sort_values(["game_date", "player_id"], inplace=True)
        skipped.reset_index(drop=True, inplace=True)
    return TrainingDatasetResult(rows=rows, skipped=skipped)


def write_training_dataset(
    result: TrainingDatasetResult,
    output_path: str | Path,
    *,
    skipped_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Write training rows to Parquet and optional skip details to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.rows.to_parquet(output, index=False)

    written_skipped: Path | None = None
    if skipped_path is not None:
        written_skipped = Path(skipped_path)
        written_skipped.parent.mkdir(parents=True, exist_ok=True)
        result.skipped.to_csv(written_skipped, index=False)

    return output, written_skipped

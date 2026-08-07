"""Cached, retrying access to the NBA Stats endpoints used by the project."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from nba_api.stats.endpoints import (
    BoxScoreAdvancedV3,
    BoxScorePlayerTrackV3,
    LeagueDashPlayerStats,
    LeagueDashPtStats,
    PlayerGameLog,
    PlayerGameLogs,
)

from .config import (
    CACHE_DIR,
    REQUEST_RETRIES,
    REQUEST_SLEEP_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)


class NBADataFetchError(RuntimeError):
    """Raised when a required NBA Stats request fails after all retries."""


def _nba_date(value) -> str:
    if value in (None, ""):
        return ""
    return pd.to_datetime(value).strftime("%m/%d/%Y")


def _cache_path(namespace: str, parameters: dict) -> Path:
    payload = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return CACHE_DIR / f"{namespace}_{digest}.parquet"


def _cached_request(
    namespace: str,
    parameters: dict,
    request: Callable[[], pd.DataFrame],
    *,
    refresh: bool = False,
    retries: int = REQUEST_RETRIES,
    sleep_seconds: float = REQUEST_SLEEP_SECONDS,
) -> pd.DataFrame:
    """Return a cached response or execute and cache a retrying request."""
    path = _cache_path(namespace, parameters)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            frame = request()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            return frame
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))

    raise NBADataFetchError(
        f"NBA Stats request {namespace!r} failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def get_player_gamelog(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Fetch one player's season game log without caching."""
    endpoint = PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return endpoint.get_data_frames()[0]


def fetch_player_gamelog(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    params = {
        "player_id": int(player_id),
        "season": season,
        "season_type": season_type,
    }
    return _cached_request(
        "player_gamelog",
        params,
        lambda: get_player_gamelog(int(player_id), season, season_type),
        refresh=refresh,
    )


def fetch_players_gamelogs(
    player_ids: Iterable[int],
    season: str,
    season_type: str = "Regular Season",
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and concatenate cached game logs for multiple players."""
    frames = [
        fetch_player_gamelog(
            int(player_id),
            season,
            season_type,
            refresh=refresh,
        )
        for player_id in sorted(set(player_ids))
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_season_player_game_logs(
    season: str,
    season_type: str = "Regular Season",
    measure_type: str = "Base",
) -> pd.DataFrame:
    """Fetch league-wide player-game rows for one season without caching."""
    endpoint = PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        measure_type_player_game_logs_nullable=measure_type,
        per_mode_simple_nullable="Totals",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return endpoint.get_data_frames()[0]


def fetch_season_player_game_logs(
    season: str,
    season_type: str = "Regular Season",
    measure_type: str = "Base",
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache every player-game row for a regular-season measure."""
    params = {
        "season": season,
        "season_type": season_type,
        "measure_type": measure_type,
    }
    frame = _cached_request(
        "season_player_game_logs",
        params,
        lambda: get_season_player_game_logs(season, season_type, measure_type),
        refresh=refresh,
    )
    if frame.empty:
        _cache_path("season_player_game_logs", params).unlink(missing_ok=True)
        raise NBADataFetchError(
            f"NBA Stats returned no {measure_type} player-game rows for {season}."
        )
    return frame


def get_game_player_tracking(game_id: str) -> pd.DataFrame:
    endpoint = BoxScorePlayerTrackV3(
        game_id=str(game_id),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    frame = endpoint.get_data_frames()[0]
    frame["GAME_ID"] = str(game_id)
    return frame


def get_game_player_advanced(game_id: str) -> pd.DataFrame:
    endpoint = BoxScoreAdvancedV3(
        game_id=str(game_id),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    frame = endpoint.get_data_frames()[0]
    frame["GAME_ID"] = str(game_id)
    return frame


def _fetch_games(
    game_ids: Iterable[str],
    *,
    namespace: str,
    getter: Callable[[str], pd.DataFrame],
    refresh: bool,
    verbose: bool = False,
    progress_every: int = 100,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    unique_game_ids = sorted({str(game_id) for game_id in game_ids})
    for position, game_id in enumerate(unique_game_ids, start=1):
        parameters = {"game_id": game_id}
        cache_path = _cache_path(namespace, parameters)
        network_request = refresh or not cache_path.exists()
        try:
            frame = _cached_request(
                namespace,
                parameters,
                lambda game_id=game_id: getter(game_id),
                refresh=refresh,
            )
            if frame.empty:
                cache_path.unlink(missing_ok=True)
                raise NBADataFetchError(
                    f"NBA Stats returned no rows for game {game_id}."
                )
            frames.append(frame)
        except NBADataFetchError as exc:
            print(f"{namespace.replace('_', ' ').title()} failed for {game_id}: {exc}")
        finally:
            if network_request and position < len(unique_game_ids):
                time.sleep(REQUEST_SLEEP_SECONDS)
        if verbose and (
            position == len(unique_game_ids)
            or (progress_every > 0 and position % progress_every == 0)
        ):
            print(
                f"{namespace.replace('_', ' ').title()}: "
                f"{position}/{len(unique_game_ids)} games"
            )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_tracking_for_games(
    game_ids: Iterable[str],
    *,
    refresh: bool = False,
    verbose: bool = False,
    progress_every: int = 100,
) -> pd.DataFrame:
    return _fetch_games(
        game_ids,
        namespace="boxscore_tracking",
        getter=get_game_player_tracking,
        refresh=refresh,
        verbose=verbose,
        progress_every=progress_every,
    )


def fetch_advanced_for_games(
    game_ids: Iterable[str],
    *,
    refresh: bool = False,
    verbose: bool = False,
    progress_every: int = 100,
) -> pd.DataFrame:
    return _fetch_games(
        game_ids,
        namespace="boxscore_advanced",
        getter=get_game_player_advanced,
        refresh=refresh,
        verbose=verbose,
        progress_every=progress_every,
    )


def fetch_league_player_stats(
    *,
    season: str,
    date_from,
    date_to,
    measure_type: str = "Base",
    opponent_team_id: int = 0,
    starter_bench: str = "",
    player_position: str = "",
    season_type: str = "Regular Season",
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a league-wide player dashboard slice for a fixed date window."""
    params = {
        "season": season,
        "date_from": _nba_date(date_from),
        "date_to": _nba_date(date_to),
        "measure_type": measure_type,
        "opponent_team_id": int(opponent_team_id),
        "starter_bench": starter_bench,
        "player_position": player_position,
        "season_type": season_type,
    }

    def request() -> pd.DataFrame:
        endpoint = LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            date_from_nullable=params["date_from"],
            date_to_nullable=params["date_to"],
            measure_type_detailed_defense=measure_type,
            per_mode_detailed="Totals",
            opponent_team_id=int(opponent_team_id),
            starter_bench_nullable=starter_bench,
            player_position_abbreviation_nullable=player_position,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return endpoint.get_data_frames()[0]

    return _cached_request("league_player_stats", params, request, refresh=refresh)


def fetch_league_tracking_stats(
    *,
    season: str,
    date_from,
    date_to,
    measure_type: str,
    opponent_team_id: int = 0,
    season_type: str = "Regular Season",
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a league-wide player tracking dashboard slice."""
    params = {
        "season": season,
        "date_from": _nba_date(date_from),
        "date_to": _nba_date(date_to),
        "measure_type": measure_type,
        "opponent_team_id": int(opponent_team_id),
        "season_type": season_type,
    }

    def request() -> pd.DataFrame:
        endpoint = LeagueDashPtStats(
            season=season,
            season_type_all_star=season_type,
            date_from_nullable=params["date_from"],
            date_to_nullable=params["date_to"],
            per_mode_simple="Totals",
            player_or_team="Player",
            pt_measure_type=measure_type,
            opponent_team_id=int(opponent_team_id),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return endpoint.get_data_frames()[0]

    return _cached_request("league_tracking_stats", params, request, refresh=refresh)

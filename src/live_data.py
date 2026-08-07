"""Automatic player/game lookup and league dashboard normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams

from .config import MAX_SUPPORTED_SEASON_START, MIN_TRACKING_SEASON_START
from .dataset import clean_gamelog
from .fetch_nba import (
    fetch_league_player_stats,
    fetch_league_tracking_stats,
    fetch_player_gamelog,
)
from .historical_store import (
    build_local_dashboard_profiles,
    build_local_opponent_rates,
    load_season_player_games,
    season_store_exists,
)


def resolve_player(player_query: str | int) -> tuple[int, str]:
    """Resolve an NBA player name or numeric ID using nba_api's static data."""
    query = str(player_query).strip()
    if query.isdigit():
        player_id = int(query)
        match = nba_players.find_player_by_id(player_id)
        return player_id, match["full_name"] if match else query

    matches = nba_players.find_players_by_full_name(query)
    if not matches:
        raise ValueError(f"No NBA player found for {player_query!r}.")

    exact = [match for match in matches if match["full_name"].casefold() == query.casefold()]
    if len(exact) == 1:
        match = exact[0]
        return int(match["id"]), match["full_name"]
    if len(matches) == 1:
        match = matches[0]
        return int(match["id"]), match["full_name"]

    options = ", ".join(f"{match['full_name']} ({match['id']})" for match in matches[:10])
    raise ValueError(
        f"Multiple players matched {player_query!r}. Use a full name or NBA ID. Matches: {options}"
    )


def season_from_game_date(game_date) -> str:
    """Convert a regular-season date into an NBA season string."""
    date = pd.to_datetime(game_date)
    start_year = date.year if date.month >= 9 else date.year - 1
    if start_year < MIN_TRACKING_SEASON_START:
        raise ValueError(
            f"This model requires rebound tracking, available here from "
            f"{MIN_TRACKING_SEASON_START}-{str(MIN_TRACKING_SEASON_START + 1)[-2:]} onward."
        )
    if start_year > MAX_SUPPORTED_SEASON_START:
        raise ValueError(
            f"The automatic workflow currently supports seasons through "
            f"{MAX_SUPPORTED_SEASON_START}-{str(MAX_SUPPORTED_SEASON_START + 1)[-2:]}."
        )
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def resolve_team_id(team_abbreviation: str) -> int:
    match = nba_teams.find_team_by_abbreviation(str(team_abbreviation).upper())
    if not match:
        raise ValueError(f"Unknown NBA team abbreviation: {team_abbreviation!r}")
    return int(match["id"])


def lookup_player_game(
    player_id: int,
    game_date,
    *,
    season: str | None = None,
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> pd.Series:
    """Find an exact regular-season player-game and its opponent."""
    game_date = pd.to_datetime(game_date).normalize()
    season = season or season_from_game_date(game_date)
    if not refresh and season_store_exists(season, store_dir):
        local_games = load_season_player_games(
            season,
            player_ids=[int(player_id)],
            date_from=game_date,
            date_to=game_date,
            store_dir=store_dir,
        )
        if local_games.empty:
            raise ValueError(
                f"No regular-season game was found for player {player_id} on "
                f"{game_date.date()} in the local {season} store."
            )
        game = local_games.iloc[0].copy()
        game["season"] = season
        return game

    raw = fetch_player_gamelog(
        int(player_id),
        season,
        "Regular Season",
        refresh=refresh,
    )
    games = clean_gamelog(raw)
    matches = games[games["game_date"].dt.normalize() == game_date]
    if matches.empty:
        raise ValueError(
            f"No regular-season game was found for player {player_id} on {game_date.date()} "
            f"in season {season}. The player may not have appeared in that game."
        )

    game = matches.iloc[0].copy()
    if pd.isna(game.get("opp_abbr")):
        raise ValueError("The game was found, but its opponent could not be parsed.")
    game["season"] = season
    game["opp_abbr"] = str(game["opp_abbr"]).upper()
    game["opp_team_id"] = resolve_team_id(game["opp_abbr"])
    return game


def _one_row_per_player(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer league-total rows when a dashboard includes team-split duplicates."""
    if frame.empty or "PLAYER_ID" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()
    result["PLAYER_ID"] = pd.to_numeric(result["PLAYER_ID"], errors="coerce")
    result = result.dropna(subset=["PLAYER_ID"]).copy()
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)
    team_count = result["TEAM_COUNT"] if "TEAM_COUNT" in result.columns else pd.Series(1, index=result.index)
    minutes = result["MIN"] if "MIN" in result.columns else pd.Series(0, index=result.index)
    result["_team_count"] = pd.to_numeric(team_count, errors="coerce").fillna(1)
    result["_minutes_sort"] = pd.to_numeric(minutes, errors="coerce").fillna(0)
    result.sort_values(
        ["PLAYER_ID", "_team_count", "_minutes_sort"],
        ascending=[True, False, False],
        inplace=True,
    )
    return result.drop_duplicates("PLAYER_ID", keep="first").set_index("PLAYER_ID")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.div(denominator.where(denominator > 0))


def _numeric_column(frame: pd.DataFrame, column: str, index: pd.Index) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").reindex(index)


def build_dashboard_profiles(
    *,
    season: str,
    as_of_date,
    lookback_days: int = 30,
    min_games: int = 5,
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build every eligible player's requested 30-day similarity profile."""
    as_of_date = pd.to_datetime(as_of_date)
    if not refresh and season_store_exists(season, store_dir):
        return build_local_dashboard_profiles(
            season=season,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            min_games=min_games,
            store_dir=store_dir,
        )

    date_from = as_of_date - pd.Timedelta(days=lookback_days)
    request_args: dict[str, Any] = {
        "season": season,
        "date_from": date_from,
        "date_to": as_of_date,
        "season_type": "Regular Season",
        "refresh": refresh,
    }

    base = _one_row_per_player(fetch_league_player_stats(measure_type="Base", **request_args))
    advanced = _one_row_per_player(
        fetch_league_player_stats(measure_type="Advanced", **request_args)
    )
    rebounds = _one_row_per_player(
        fetch_league_tracking_stats(measure_type="Rebounding", **request_args)
    )
    possessions = _one_row_per_player(
        fetch_league_tracking_stats(measure_type="Possessions", **request_args)
    )
    starters = _one_row_per_player(
        fetch_league_player_stats(measure_type="Base", starter_bench="Starters", **request_args)
    )

    if base.empty:
        return pd.DataFrame()

    profiles = pd.DataFrame(index=base.index)
    profiles.index.name = "player_id"
    profiles["player_name"] = base.get("PLAYER_NAME")
    profiles["games_used"] = _numeric_column(base, "GP", profiles.index)
    base_minutes = _numeric_column(base, "MIN", profiles.index)
    profiles["minutes"] = _safe_divide(base_minutes, profiles["games_used"])
    profiles["fga_per_min"] = _safe_divide(
        _numeric_column(base, "FGA", profiles.index),
        base_minutes,
    )

    rebound_minutes = _numeric_column(rebounds, "MIN", profiles.index)
    profiles["reb_chances_per_min"] = _safe_divide(
        _numeric_column(rebounds, "REB_CHANCES", profiles.index),
        rebound_minutes,
    )
    profiles["usage_rate"] = _numeric_column(advanced, "USG_PCT", profiles.index)

    possession_minutes = _numeric_column(possessions, "MIN", profiles.index)
    profiles["touches_per_min"] = _safe_divide(
        _numeric_column(possessions, "TOUCHES", profiles.index),
        possession_minutes,
    )

    starter_games = _numeric_column(starters, "GP", profiles.index).fillna(0)
    profiles["starter_rate"] = _safe_divide(starter_games, profiles["games_used"]).fillna(0).clip(0, 1)
    profiles["starter_flag"] = (profiles["starter_rate"] >= 0.5).astype(int)

    memberships: dict[int, list[str]] = {}
    for position in ["G", "F", "C"]:
        position_rows = _one_row_per_player(
            fetch_league_player_stats(
                measure_type="Base",
                player_position=position,
                **request_args,
            )
        )
        for player_id in position_rows.index:
            memberships.setdefault(int(player_id), []).append(position)
    profiles["listed_position"] = [
        "-".join(memberships.get(int(player_id), [])) or pd.NA
        for player_id in profiles.index
    ]

    required = [
        "minutes",
        "fga_per_min",
        "reb_chances_per_min",
        "usage_rate",
        "touches_per_min",
    ]
    profiles = profiles[
        (profiles["games_used"] >= min_games)
        & (profiles["minutes"] > 0)
        & profiles[required].notna().all(axis=1)
    ].copy()
    profiles["as_of_date"] = as_of_date
    profiles["lookback_days"] = lookback_days
    return profiles


def build_opponent_rates(
    *,
    season: str,
    as_of_date,
    opponent_team_id: int,
    lookback_days: int = 30,
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build every player's FGA/min and rebound-chance/min split vs one team."""
    as_of_date = pd.to_datetime(as_of_date)
    if not refresh and season_store_exists(season, store_dir):
        return build_local_opponent_rates(
            season=season,
            as_of_date=as_of_date,
            opponent_team_id=opponent_team_id,
            lookback_days=lookback_days,
            store_dir=store_dir,
        )

    date_from = as_of_date - pd.Timedelta(days=lookback_days)
    request_args: dict[str, Any] = {
        "season": season,
        "date_from": date_from,
        "date_to": as_of_date,
        "opponent_team_id": int(opponent_team_id),
        "season_type": "Regular Season",
        "refresh": refresh,
    }

    base = _one_row_per_player(fetch_league_player_stats(measure_type="Base", **request_args))
    rebounds = _one_row_per_player(
        fetch_league_tracking_stats(measure_type="Rebounding", **request_args)
    )
    if base.empty and rebounds.empty:
        return pd.DataFrame()

    index = base.index.union(rebounds.index)
    rates = pd.DataFrame(index=index)
    rates.index.name = "player_id"
    base_minutes = _numeric_column(base, "MIN", index)
    rebound_minutes = _numeric_column(rebounds, "MIN", index)
    rates["fga_per_min"] = _safe_divide(_numeric_column(base, "FGA", index), base_minutes)
    rates["fga_games"] = _numeric_column(base, "GP", index)
    rates["reb_chances_per_min"] = _safe_divide(
        _numeric_column(rebounds, "REB_CHANCES", index),
        rebound_minutes,
    )
    rates["reb_chances_games"] = _numeric_column(rebounds, "GP", index)
    return rates.replace([np.inf, -np.inf], np.nan)

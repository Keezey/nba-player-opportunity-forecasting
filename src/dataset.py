"""Build the per-game dataset used by baselines and historical backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .fetch_nba import (
    fetch_advanced_for_games,
    fetch_players_gamelogs,
    fetch_tracking_for_games,
)


PLAYER_GAME_COLUMNS = [
    "game_id",
    "game_date",
    "player_id",
    "opp_abbr",
    "minutes",
    "fga",
    "reb",
    "reb_chances",
    "touches",
    "usage_rate",
    "listed_position",
    "starter_flag",
]


def _parse_minutes(min_value) -> float:
    """Convert NBA minute values into decimal minutes."""
    if pd.isna(min_value):
        return 0.0
    if isinstance(min_value, (int, float)):
        return float(min_value)

    try:
        parts = str(min_value).split(":")
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60.0
        return float(min_value)
    except (TypeError, ValueError):
        return 0.0


def clean_gamelog(raw_gamelog_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw PlayerGameLog output into the model's box-score fields."""
    if raw_gamelog_df.empty:
        return pd.DataFrame()

    df = raw_gamelog_df.copy()
    if "PLAYER_ID" in df.columns and "Player_ID" in df.columns:
        df.drop(columns=["PLAYER_ID"], inplace=True)

    df.rename(
        columns={
            "Game_ID": "game_id",
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "Player_ID": "player_id",
            "PLAYER_ID": "player_id",
            "MIN": "min_raw",
            "FGA": "fga",
            "REB": "reb",
            "MATCHUP": "matchup",
        },
        inplace=True,
    )
    df = df.loc[:, ~df.columns.duplicated()]

    required = ["game_id", "game_date", "player_id", "min_raw", "fga", "reb", "matchup"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Raw game log is missing required columns: {missing}")

    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["minutes"] = df["min_raw"].apply(_parse_minutes)
    df["opp_abbr"] = df["matchup"].astype(str).str.extract(
        r"(?:vs\.?|@)\s+([A-Z]{3})$",
        expand=False,
    )

    return df[["game_id", "game_date", "player_id", "opp_abbr", "minutes", "fga", "reb"]].copy()


def normalize_tracking(tracking_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the V3 tracking fields used by the current model."""
    if tracking_raw.empty:
        return pd.DataFrame(
            columns=["game_id", "player_id", "reb_chances", "touches", "tracking_position"]
        )

    tracking = tracking_raw.copy()
    tracking.rename(
        columns={
            "gameId": "game_id",
            "personId": "player_id",
            "reboundChancesTotal": "reb_chances",
            "touches": "touches",
            "position": "tracking_position",
        },
        inplace=True,
    )
    if "GAME_ID" in tracking.columns:
        tracking.drop(columns=["GAME_ID"], inplace=True)
    tracking = tracking.loc[:, ~tracking.columns.duplicated()]

    for column in ["game_id", "player_id", "reb_chances", "touches", "tracking_position"]:
        if column not in tracking.columns:
            tracking[column] = pd.NA

    tracking["game_id"] = tracking["game_id"].astype(str)
    return tracking[["game_id", "player_id", "reb_chances", "touches", "tracking_position"]].copy()


def normalize_advanced(advanced_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize per-game usage and backup position from BoxScoreAdvancedV3."""
    if advanced_raw.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "usage_rate", "advanced_position"])

    advanced = advanced_raw.copy()
    advanced.rename(
        columns={
            "gameId": "game_id",
            "personId": "player_id",
            "usagePercentage": "usage_rate",
            "position": "advanced_position",
        },
        inplace=True,
    )
    if "GAME_ID" in advanced.columns:
        advanced.drop(columns=["GAME_ID"], inplace=True)
    advanced = advanced.loc[:, ~advanced.columns.duplicated()]

    for column in ["game_id", "player_id", "usage_rate", "advanced_position"]:
        if column not in advanced.columns:
            advanced[column] = pd.NA

    advanced["game_id"] = advanced["game_id"].astype(str)
    return advanced[["game_id", "player_id", "usage_rate", "advanced_position"]].copy()


def _clean_position(value):
    if pd.isna(value):
        return pd.NA
    position = str(value).strip().upper()
    return position if position else pd.NA


def build_player_game_df(
    player_ids: Iterable[int],
    season: str = "2025-26",
    season_type: str = "Regular Season",
    date_from=None,
    date_to=None,
    include_advanced: bool = True,
    refresh: bool = False,
    verbose: bool = False,
    prefer_local: bool = True,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch and build a player-game dataset for selected players and dates."""
    if prefer_local and not refresh and season_type == "Regular Season":
        from .historical_store import load_season_player_games, season_store_exists

        if season_store_exists(season, store_dir):
            local = load_season_player_games(
                season,
                player_ids=player_ids,
                date_from=date_from,
                date_to=date_to,
                store_dir=store_dir,
            )
            if not include_advanced and not local.empty:
                local["usage_rate"] = pd.NA
            result = local.reindex(columns=PLAYER_GAME_COLUMNS).copy()
            result.sort_values(["game_date", "player_id"], inplace=True)
            result.reset_index(drop=True, inplace=True)
            if verbose:
                print(
                    f"Loaded {len(result)} local rows for "
                    f"{result['player_id'].nunique() if not result.empty else 0} players."
                )
            return result

    raw_logs = fetch_players_gamelogs(
        player_ids=player_ids,
        season=season,
        season_type=season_type,
        refresh=refresh,
    )
    if raw_logs.empty:
        return pd.DataFrame(columns=PLAYER_GAME_COLUMNS)

    logs = clean_gamelog(raw_logs)
    if date_from is not None:
        logs = logs[logs["game_date"] >= pd.to_datetime(date_from)].copy()
    if date_to is not None:
        logs = logs[logs["game_date"] <= pd.to_datetime(date_to)].copy()
    if logs.empty:
        return pd.DataFrame(columns=PLAYER_GAME_COLUMNS)

    game_ids = logs["game_id"].unique().tolist()
    tracking = normalize_tracking(fetch_tracking_for_games(game_ids, refresh=refresh))
    merged = logs.merge(tracking, on=["game_id", "player_id"], how="left")

    if include_advanced:
        advanced = normalize_advanced(fetch_advanced_for_games(game_ids, refresh=refresh))
        merged = merged.merge(advanced, on=["game_id", "player_id"], how="left")
    else:
        merged["usage_rate"] = pd.NA
        merged["advanced_position"] = pd.NA

    tracking_position = merged["tracking_position"].apply(_clean_position)
    advanced_position = merged["advanced_position"].apply(_clean_position)
    merged["listed_position"] = tracking_position.combine_first(advanced_position)
    merged["starter_flag"] = merged["listed_position"].notna().astype(int)

    for column in ["fga", "reb", "reb_chances", "touches", "usage_rate"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    player_game_df = merged[PLAYER_GAME_COLUMNS].copy()
    player_game_df.sort_values(["game_date", "player_id"], inplace=True)
    player_game_df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"Built {len(player_game_df)} rows for {player_game_df['player_id'].nunique()} players.")
    return player_game_df

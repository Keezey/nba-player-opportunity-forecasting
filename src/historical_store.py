"""Canonical, season-partitioned player-game storage and local feature queries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    HISTORICAL_STORE_DIR,
    MAX_SUPPORTED_SEASON_START,
    MIN_TRACKING_SEASON_START,
)
from .dataset import _parse_minutes, normalize_tracking
from .fetch_nba import fetch_season_player_game_logs, fetch_tracking_for_games


STORE_SCHEMA_VERSION = 1
HISTORICAL_PLAYER_GAME_COLUMNS = [
    "season",
    "game_id",
    "game_date",
    "player_id",
    "player_name",
    "team_id",
    "team_abbr",
    "opp_team_id",
    "opp_abbr",
    "home_away",
    "minutes",
    "fga",
    "reb",
    "reb_chances",
    "touches",
    "usage_rate",
    "listed_position",
    "starter_flag",
    "tracking_available",
    "advanced_available",
]


@dataclass(frozen=True)
class SeasonStoreBuildResult:
    season: str
    data_path: Path
    manifest_path: Path
    manifest: dict


def _store_root(store_dir: str | Path | None = None) -> Path:
    return Path(store_dir) if store_dir is not None else HISTORICAL_STORE_DIR


def season_store_dir(season: str, store_dir: str | Path | None = None) -> Path:
    return _store_root(store_dir) / f"season={season}"


def season_data_path(season: str, store_dir: str | Path | None = None) -> Path:
    return season_store_dir(season, store_dir) / "player_games.parquet"


def season_manifest_path(season: str, store_dir: str | Path | None = None) -> Path:
    return season_store_dir(season, store_dir) / "manifest.json"


def season_store_exists(season: str, store_dir: str | Path | None = None) -> bool:
    return season_data_path(season, store_dir).exists()


def validate_season(season: str) -> str:
    value = str(season)
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if not match:
        raise ValueError(f"Invalid NBA season string: {season!r}")
    start_year = int(match.group(1))
    expected_suffix = str(start_year + 1)[-2:]
    if match.group(2) != expected_suffix:
        raise ValueError(f"Invalid NBA season sequence: {season!r}")
    if not MIN_TRACKING_SEASON_START <= start_year <= MAX_SUPPORTED_SEASON_START:
        raise ValueError(
            f"Season {season} is outside the supported tracking range "
            f"{MIN_TRACKING_SEASON_START}-{str(MIN_TRACKING_SEASON_START + 1)[-2:]} "
            f"through {MAX_SUPPORTED_SEASON_START}-{str(MAX_SUPPORTED_SEASON_START + 1)[-2:]}."
        )
    return value


def _normalize_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _normalize_position(value):
    if pd.isna(value):
        return pd.NA
    position = str(value).strip().upper()
    return position if position else pd.NA


def normalize_season_base_logs(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize league-wide Base PlayerGameLogs into player appearances."""
    if raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    required = {
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "MIN",
        "FGA",
        "REB",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Season Base logs are missing columns: {missing}")

    frame.rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbr",
            "MATCHUP": "matchup",
            "MIN": "minutes_raw",
            "FGA": "fga",
            "REB": "reb",
        },
        inplace=True,
    )
    frame["season"] = season
    frame["game_id"] = frame["game_id"].astype(str)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["player_id"] = _normalize_id(frame["player_id"])
    frame["team_id"] = _normalize_id(frame["team_id"])
    frame["minutes"] = frame["minutes_raw"].apply(_parse_minutes)
    for column in ["fga", "reb"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["team_abbr"] = frame["team_abbr"].astype("string").str.upper()
    frame["opp_abbr"] = frame["matchup"].astype(str).str.extract(
        r"(?:vs\.?|@)\s+([A-Z]{3})$", expand=False
    )
    frame["home_away"] = np.where(
        frame["matchup"].astype(str).str.contains("@", regex=False), "A", "H"
    )

    teams = frame[["game_id", "team_id", "team_abbr"]].dropna().drop_duplicates()
    opponent_lookup: list[dict] = []
    for game_id, game_teams in teams.groupby("game_id", sort=False):
        game_teams = game_teams.drop_duplicates("team_id")
        if len(game_teams) != 2:
            continue
        first, second = game_teams.iloc[0], game_teams.iloc[1]
        opponent_lookup.extend(
            [
                {
                    "game_id": game_id,
                    "team_id": first["team_id"],
                    "opp_team_id": second["team_id"],
                    "derived_opp_abbr": second["team_abbr"],
                },
                {
                    "game_id": game_id,
                    "team_id": second["team_id"],
                    "opp_team_id": first["team_id"],
                    "derived_opp_abbr": first["team_abbr"],
                },
            ]
        )
    if opponent_lookup:
        frame = frame.merge(
            pd.DataFrame(opponent_lookup), on=["game_id", "team_id"], how="left"
        )
        frame["opp_abbr"] = frame["derived_opp_abbr"].combine_first(frame["opp_abbr"])
        frame.drop(columns=["derived_opp_abbr"], inplace=True)
    else:
        frame["opp_team_id"] = pd.NA

    frame["opp_team_id"] = _normalize_id(frame["opp_team_id"])
    frame = frame[(frame["minutes"] > 0) & frame["player_id"].notna()].copy()
    return frame[
        [
            "season",
            "game_id",
            "game_date",
            "player_id",
            "player_name",
            "team_id",
            "team_abbr",
            "opp_team_id",
            "opp_abbr",
            "home_away",
            "minutes",
            "fga",
            "reb",
        ]
    ]


def normalize_season_advanced_logs(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "usage_rate"])
    required = {"GAME_ID", "PLAYER_ID", "USG_PCT"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"Season Advanced logs are missing columns: {missing}")
    frame = raw.rename(
        columns={
            "GAME_ID": "game_id",
            "PLAYER_ID": "player_id",
            "USG_PCT": "usage_rate",
        }
    ).copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["player_id"] = _normalize_id(frame["player_id"])
    frame["usage_rate"] = pd.to_numeric(frame["usage_rate"], errors="coerce")
    return frame[["game_id", "player_id", "usage_rate"]].drop_duplicates(
        ["game_id", "player_id"], keep="first"
    )


def assemble_season_player_games(
    base_raw: pd.DataFrame,
    advanced_raw: pd.DataFrame,
    tracking_raw: pd.DataFrame,
    *,
    season: str,
) -> pd.DataFrame:
    """Merge season-wide box scores with game-level public tracking."""
    base = normalize_season_base_logs(base_raw, season)
    advanced = normalize_season_advanced_logs(advanced_raw)
    tracking = normalize_tracking(tracking_raw).drop_duplicates(
        ["game_id", "player_id"], keep="first"
    )
    tracking["_tracking_row_available"] = True

    frame = base.merge(advanced, on=["game_id", "player_id"], how="left")
    frame = frame.merge(tracking, on=["game_id", "player_id"], how="left")
    frame["listed_position"] = frame["tracking_position"].apply(_normalize_position)
    tracking_row_available = frame["_tracking_row_available"].notna()
    frame["starter_flag"] = (
        frame["listed_position"].notna().astype("Int64").where(tracking_row_available)
    )
    frame["tracking_available"] = frame[["reb_chances", "touches"]].notna().all(axis=1)
    frame["advanced_available"] = frame["usage_rate"].notna()
    frame.drop(columns=["tracking_position", "_tracking_row_available"], inplace=True)

    for column in ["fga", "reb", "reb_chances", "touches", "usage_rate"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["player_id"] = _normalize_id(frame["player_id"])
    frame["team_id"] = _normalize_id(frame["team_id"])
    frame["opp_team_id"] = _normalize_id(frame["opp_team_id"])
    frame = frame.reindex(columns=HISTORICAL_PLAYER_GAME_COLUMNS)
    frame.sort_values(["game_date", "game_id", "player_id"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    validate_player_game_store(frame, expected_season=season)
    return frame


def validate_player_game_store(
    frame: pd.DataFrame,
    *,
    expected_season: str | None = None,
) -> None:
    missing = [column for column in HISTORICAL_PLAYER_GAME_COLUMNS if column not in frame]
    if missing:
        raise KeyError(f"Historical player-game data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Historical player-game data is empty.")
    if frame[["game_id", "player_id"]].duplicated().any():
        raise ValueError("Historical player-game data contains duplicate game/player rows.")
    if pd.to_datetime(frame["game_date"], errors="coerce").isna().any():
        raise ValueError("Historical player-game data contains invalid game dates.")
    if expected_season is not None and set(frame["season"].dropna().astype(str)) != {
        str(expected_season)
    }:
        raise ValueError(f"Historical data does not contain only season {expected_season}.")
    for column in ["minutes", "fga", "reb", "reb_chances", "touches"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if (values.dropna() < 0).any():
            raise ValueError(f"Historical data contains negative {column} values.")


def _manifest_for_frame(
    frame: pd.DataFrame,
    *,
    season: str,
    requested_game_ids: Iterable[str],
    tracking_raw: pd.DataFrame,
) -> dict:
    if tracking_raw.empty:
        tracking_game_ids: set[str] = set()
    elif "gameId" in tracking_raw:
        tracking_game_ids = set(tracking_raw["gameId"].dropna().astype(str))
    elif "GAME_ID" in tracking_raw:
        tracking_game_ids = set(tracking_raw["GAME_ID"].dropna().astype(str))
    else:
        tracking_game_ids = set()
    requested = {str(game_id) for game_id in requested_game_ids}
    missing_tracking = sorted(requested.difference(tracking_game_ids))
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "season": season,
        "season_type": "Regular Season",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": int(len(frame)),
        "players": int(frame["player_id"].nunique()),
        "games": int(frame["game_id"].nunique()),
        "date_from": pd.to_datetime(frame["game_date"]).min().date().isoformat(),
        "date_to": pd.to_datetime(frame["game_date"]).max().date().isoformat(),
        "tracking_row_coverage": float(frame["tracking_available"].mean()),
        "advanced_row_coverage": float(frame["advanced_available"].mean()),
        "missing_tracking_games": missing_tracking,
        "complete_tracking": not missing_tracking,
    }


def write_season_store(
    frame: pd.DataFrame,
    manifest: dict,
    *,
    season: str,
    store_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    directory = season_store_dir(season, store_dir)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = season_data_path(season, store_dir)
    manifest_path = season_manifest_path(season, store_dir)

    temporary_data = data_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary_data, index=False)
    temporary_data.replace(data_path)

    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    _read_season_partition.cache_clear()
    return data_path, manifest_path


def build_season_store(
    season: str,
    *,
    store_dir: str | Path | None = None,
    refresh: bool = False,
    verbose: bool = True,
    progress_every: int = 100,
) -> SeasonStoreBuildResult:
    """Download, validate, and atomically write one regular-season partition."""
    season = validate_season(season)
    if verbose:
        print(f"{season}: fetching season-wide Base player-game rows")
    base_raw = fetch_season_player_game_logs(
        season, "Regular Season", "Base", refresh=refresh
    )
    if base_raw.empty:
        raise ValueError(f"NBA Stats returned no Base player-game rows for {season}.")

    if verbose:
        print(f"{season}: fetching season-wide Advanced player-game rows")
    advanced_raw = fetch_season_player_game_logs(
        season, "Regular Season", "Advanced", refresh=refresh
    )
    game_ids = sorted(base_raw["GAME_ID"].dropna().astype(str).unique())
    if verbose:
        print(f"{season}: fetching tracking for {len(game_ids)} games")
    tracking_raw = fetch_tracking_for_games(
        game_ids,
        refresh=refresh,
        verbose=verbose,
        progress_every=progress_every,
    )

    frame = assemble_season_player_games(
        base_raw,
        advanced_raw,
        tracking_raw,
        season=season,
    )
    manifest = _manifest_for_frame(
        frame,
        season=season,
        requested_game_ids=game_ids,
        tracking_raw=tracking_raw,
    )
    data_path, manifest_path = write_season_store(
        frame, manifest, season=season, store_dir=store_dir
    )
    return SeasonStoreBuildResult(
        season=season,
        data_path=data_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


@lru_cache(maxsize=16)
def _read_season_partition(path_text: str, modified_ns: int) -> pd.DataFrame:
    """Read one immutable season partition once per file version."""
    del modified_ns
    frame = pd.read_parquet(path_text)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame


def load_season_player_games(
    season: str,
    *,
    player_ids: Iterable[int] | None = None,
    date_from=None,
    date_to=None,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load and filter one local season partition."""
    path = season_data_path(season, store_dir)
    if not path.exists():
        return pd.DataFrame(columns=HISTORICAL_PLAYER_GAME_COLUMNS)
    frame = _read_season_partition(str(path.resolve()), path.stat().st_mtime_ns)
    mask = pd.Series(True, index=frame.index)
    if player_ids is not None:
        ids = {int(player_id) for player_id in player_ids}
        mask &= pd.to_numeric(frame["player_id"], errors="coerce").isin(ids)
    if date_from is not None:
        mask &= frame["game_date"] >= pd.to_datetime(date_from)
    if date_to is not None:
        mask &= frame["game_date"] <= pd.to_datetime(date_to)
    return (
        frame.loc[mask]
        .sort_values(["game_date", "player_id"])
        .reset_index(drop=True)
        .copy()
    )


def load_store_manifest(
    season: str,
    store_dir: str | Path | None = None,
) -> dict | None:
    path = season_manifest_path(season, store_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def available_store_seasons(store_dir: str | Path | None = None) -> list[str]:
    root = _store_root(store_dir)
    if not root.exists():
        return []
    seasons = [
        path.name.removeprefix("season=")
        for path in root.glob("season=*")
        if (path / "player_games.parquet").exists()
    ]
    return sorted(seasons)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.div(denominator.where(denominator > 0))


def _aggregate_positions(values: pd.Series):
    memberships: set[str] = set()
    for value in values.dropna():
        position = str(value).upper()
        if "G" in position:
            memberships.add("G")
        if "F" in position:
            memberships.add("F")
        if "C" in position:
            memberships.add("C")
    ordered = [position for position in ["G", "F", "C"] if position in memberships]
    return "-".join(ordered) if ordered else pd.NA


def build_local_dashboard_profiles(
    *,
    season: str,
    as_of_date,
    lookback_days: int = 30,
    min_games: int = 5,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Recreate rolling similarity profiles from local pregame rows."""
    as_of = pd.to_datetime(as_of_date).normalize()
    recent = load_season_player_games(
        season,
        date_from=as_of - pd.Timedelta(days=lookback_days),
        date_to=as_of,
        store_dir=store_dir,
    )
    if recent.empty:
        return pd.DataFrame()
    recent = recent[pd.to_numeric(recent["minutes"], errors="coerce") > 0].copy()
    recent.sort_values("game_date", inplace=True)
    recent["usage_minutes"] = recent["minutes"].where(recent["usage_rate"].notna(), 0)
    recent["usage_weighted"] = (
        pd.to_numeric(recent["usage_rate"], errors="coerce")
        * recent["usage_minutes"]
    ).fillna(0)
    recent["reb_tracking_minutes"] = recent["minutes"].where(
        recent["reb_chances"].notna(), 0
    )
    recent["touch_tracking_minutes"] = recent["minutes"].where(
        recent["touches"].notna(), 0
    )
    recent["starter_known"] = recent["starter_flag"].notna().astype(int)

    grouped = recent.groupby("player_id", sort=False)
    totals = grouped.agg(
        games_used=("game_id", "nunique"),
        total_minutes=("minutes", "sum"),
        total_fga=("fga", "sum"),
        total_reb_chances=("reb_chances", "sum"),
        reb_tracking_minutes=("reb_tracking_minutes", "sum"),
        total_touches=("touches", "sum"),
        touch_tracking_minutes=("touch_tracking_minutes", "sum"),
        usage_weighted=("usage_weighted", "sum"),
        usage_minutes=("usage_minutes", "sum"),
        starter_games=("starter_flag", "sum"),
        starter_known_games=("starter_known", "sum"),
    )
    profiles = pd.DataFrame(index=totals.index)
    profiles.index.name = "player_id"
    profiles["player_name"] = grouped["player_name"].last().reindex(profiles.index)
    profiles["games_used"] = totals["games_used"]
    profiles["minutes"] = _safe_ratio(totals["total_minutes"], totals["games_used"])
    profiles["fga_per_min"] = _safe_ratio(totals["total_fga"], totals["total_minutes"])
    profiles["reb_chances_per_min"] = _safe_ratio(
        totals["total_reb_chances"], totals["reb_tracking_minutes"]
    )
    profiles["usage_rate"] = _safe_ratio(
        totals["usage_weighted"], totals["usage_minutes"]
    )
    profiles["touches_per_min"] = _safe_ratio(
        totals["total_touches"], totals["touch_tracking_minutes"]
    )
    profiles["starter_rate"] = _safe_ratio(
        totals["starter_games"], totals["starter_known_games"]
    ).clip(0, 1)
    profiles["starter_flag"] = (profiles["starter_rate"] >= 0.5).astype("Int64")
    profiles["listed_position"] = grouped["listed_position"].agg(
        _aggregate_positions
    ).reindex(profiles.index)

    required = [
        "minutes",
        "fga_per_min",
        "reb_chances_per_min",
        "usage_rate",
        "touches_per_min",
        "starter_rate",
    ]
    profiles = profiles[
        (profiles["games_used"] >= min_games)
        & (profiles["minutes"] > 0)
        & profiles[required].notna().all(axis=1)
    ].copy()
    profiles["as_of_date"] = as_of
    profiles["lookback_days"] = lookback_days
    return profiles


def build_local_opponent_rates(
    *,
    season: str,
    as_of_date,
    opponent_team_id: int,
    lookback_days: int = 30,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Calculate each player's local per-minute split against one opponent."""
    as_of = pd.to_datetime(as_of_date).normalize()
    recent = load_season_player_games(
        season,
        date_from=as_of - pd.Timedelta(days=lookback_days),
        date_to=as_of,
        store_dir=store_dir,
    )
    if recent.empty:
        return pd.DataFrame()
    recent = recent[
        pd.to_numeric(recent["opp_team_id"], errors="coerce")
        == int(opponent_team_id)
    ].copy()
    if recent.empty:
        return pd.DataFrame()

    recent["fga_minutes"] = recent["minutes"].where(recent["fga"].notna(), 0)
    recent["reb_tracking_minutes"] = recent["minutes"].where(
        recent["reb_chances"].notna(), 0
    )
    recent["fga_valid"] = recent["fga"].notna().astype(int)
    recent["reb_valid"] = recent["reb_chances"].notna().astype(int)
    totals = recent.groupby("player_id").agg(
        total_fga=("fga", "sum"),
        fga_minutes=("fga_minutes", "sum"),
        fga_games=("fga_valid", "sum"),
        total_reb_chances=("reb_chances", "sum"),
        reb_minutes=("reb_tracking_minutes", "sum"),
        reb_chances_games=("reb_valid", "sum"),
    )
    rates = pd.DataFrame(index=totals.index)
    rates.index.name = "player_id"
    rates["fga_per_min"] = _safe_ratio(totals["total_fga"], totals["fga_minutes"])
    rates["fga_games"] = totals["fga_games"]
    rates["reb_chances_per_min"] = _safe_ratio(
        totals["total_reb_chances"], totals["reb_minutes"]
    )
    rates["reb_chances_games"] = totals["reb_chances_games"]
    return rates.replace([np.inf, -np.inf], np.nan)

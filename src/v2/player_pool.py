"""Reproducible target-player selection from the local historical warehouse."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..historical_store import (
    load_season_player_games,
    season_store_exists,
    validate_season,
)


POSITION_GROUPS = ("G", "F", "C")
USAGE_TIERS = ("low", "medium", "high")


@dataclass(frozen=True)
class PlayerPoolResult:
    """Selected players plus the profiles used to audit their selection."""

    selected: pd.DataFrame
    eligible_players: pd.DataFrame
    player_seasons: pd.DataFrame
    strata_summary: pd.DataFrame


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.div(denominator.where(denominator > 0))


def _position_memberships(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip().upper().replace("/", "-")
    if not text:
        return []
    memberships: list[str] = []
    for token in text.split("-"):
        token = token.strip()
        if token.endswith("G") and "G" not in memberships:
            memberships.append("G")
        if token.endswith("F") and "F" not in memberships:
            memberships.append("F")
        if token == "C" and "C" not in memberships:
            memberships.append("C")
    return memberships


def _dominant_position(values: pd.Series):
    observed: list[str] = []
    counts = {position: 0 for position in POSITION_GROUPS}
    for value in values:
        memberships = _position_memberships(value)
        observed.extend(memberships)
        for position in memberships:
            counts[position] += 1
    maximum = max(counts.values(), default=0)
    if maximum == 0:
        return pd.NA
    tied = {position for position, count in counts.items() if count == maximum}
    for position in reversed(observed):
        if position in tied:
            return position
    return pd.NA


def _eligibility_reason(
    row: pd.Series,
    *,
    min_games: int,
    min_average_minutes: float,
    min_starter_rate: float,
    min_data_coverage: float,
) -> str:
    reasons = []
    if row["games"] < min_games:
        reasons.append("games")
    if pd.isna(row["avg_minutes"]) or row["avg_minutes"] < min_average_minutes:
        reasons.append("minutes")
    if pd.isna(row["starter_rate"]) or row["starter_rate"] < min_starter_rate:
        reasons.append("starter_rate")
    if (
        pd.isna(row["required_data_coverage"])
        or row["required_data_coverage"] < min_data_coverage
    ):
        reasons.append("data_coverage")
    if pd.isna(row["usage_rate"]):
        reasons.append("usage_rate")
    if pd.isna(row["position_group"]):
        reasons.append("position")
    return "eligible" if not reasons else ";".join(reasons)


def build_player_season_profiles(
    player_games: pd.DataFrame,
    *,
    min_games: int = 30,
    min_average_minutes: float = 20.0,
    min_starter_rate: float = 0.50,
    min_data_coverage: float = 0.95,
) -> pd.DataFrame:
    """Aggregate player appearances and apply fixed player-season eligibility."""
    if min_games <= 0:
        raise ValueError("min_games must be positive.")
    if min_average_minutes < 0:
        raise ValueError("min_average_minutes cannot be negative.")
    if not 0 <= min_starter_rate <= 1:
        raise ValueError("min_starter_rate must be between 0 and 1.")
    if not 0 <= min_data_coverage <= 1:
        raise ValueError("min_data_coverage must be between 0 and 1.")
    if player_games.empty:
        return pd.DataFrame()

    required = {
        "season",
        "game_id",
        "game_date",
        "player_id",
        "player_name",
        "minutes",
        "fga",
        "reb_chances",
        "touches",
        "usage_rate",
        "listed_position",
        "starter_flag",
    }
    missing = sorted(required.difference(player_games.columns))
    if missing:
        raise KeyError(f"Player-game warehouse data is missing columns: {missing}")

    games = player_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games.sort_values(["season", "player_id", "game_date"], inplace=True)
    for column in [
        "minutes",
        "fga",
        "reb_chances",
        "touches",
        "usage_rate",
        "starter_flag",
    ]:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    games["usage_known"] = games["usage_rate"].notna().astype(int)
    games["usage_minutes"] = games["minutes"].where(games["usage_rate"].notna(), 0)
    games["usage_weighted"] = (
        games["usage_rate"] * games["usage_minutes"]
    ).fillna(0)
    games["tracking_known"] = games[["reb_chances", "touches"]].notna().all(axis=1).astype(int)
    games["reb_minutes"] = games["minutes"].where(games["reb_chances"].notna(), 0)
    games["touch_minutes"] = games["minutes"].where(games["touches"].notna(), 0)
    games["fga_minutes"] = games["minutes"].where(games["fga"].notna(), 0)
    games["starter_known"] = games["starter_flag"].notna().astype(int)
    games["starter_value"] = games["starter_flag"].fillna(0)

    grouped = games.groupby(["season", "player_id"], sort=True)
    totals = grouped.agg(
        player_name=("player_name", "last"),
        games=("game_id", "nunique"),
        total_minutes=("minutes", "sum"),
        total_fga=("fga", "sum"),
        fga_minutes=("fga_minutes", "sum"),
        total_reb_chances=("reb_chances", "sum"),
        reb_minutes=("reb_minutes", "sum"),
        total_touches=("touches", "sum"),
        touch_minutes=("touch_minutes", "sum"),
        usage_weighted=("usage_weighted", "sum"),
        usage_minutes=("usage_minutes", "sum"),
        usage_games=("usage_known", "sum"),
        tracking_games=("tracking_known", "sum"),
        starter_games=("starter_value", "sum"),
        starter_known_games=("starter_known", "sum"),
    ).reset_index()
    positions = grouped["listed_position"].agg(_dominant_position).reset_index(
        name="position_group"
    )
    profiles = totals.merge(positions, on=["season", "player_id"], how="left")

    profiles["avg_minutes"] = _safe_divide(
        profiles["total_minutes"], profiles["games"]
    )
    profiles["fga_per_min"] = _safe_divide(
        profiles["total_fga"], profiles["fga_minutes"]
    )
    profiles["reb_chances_per_min"] = _safe_divide(
        profiles["total_reb_chances"], profiles["reb_minutes"]
    )
    profiles["touches_per_min"] = _safe_divide(
        profiles["total_touches"], profiles["touch_minutes"]
    )
    profiles["usage_rate"] = _safe_divide(
        profiles["usage_weighted"], profiles["usage_minutes"]
    )
    profiles["starter_rate"] = _safe_divide(
        profiles["starter_games"], profiles["starter_known_games"]
    )
    profiles["usage_coverage"] = _safe_divide(
        profiles["usage_games"], profiles["games"]
    )
    profiles["tracking_coverage"] = _safe_divide(
        profiles["tracking_games"], profiles["games"]
    )
    profiles["starter_coverage"] = _safe_divide(
        profiles["starter_known_games"], profiles["games"]
    )
    profiles["required_data_coverage"] = profiles[
        ["usage_coverage", "tracking_coverage", "starter_coverage"]
    ].min(axis=1)
    profiles["eligibility_reason"] = profiles.apply(
        _eligibility_reason,
        axis=1,
        min_games=min_games,
        min_average_minutes=min_average_minutes,
        min_starter_rate=min_starter_rate,
        min_data_coverage=min_data_coverage,
    )
    profiles["eligible"] = profiles["eligibility_reason"].eq("eligible")

    output_columns = [
        "season",
        "player_id",
        "player_name",
        "games",
        "avg_minutes",
        "starter_rate",
        "usage_rate",
        "fga_per_min",
        "reb_chances_per_min",
        "touches_per_min",
        "position_group",
        "usage_coverage",
        "tracking_coverage",
        "starter_coverage",
        "required_data_coverage",
        "eligible",
        "eligibility_reason",
    ]
    return profiles[output_columns].sort_values(
        ["season", "player_id"]
    ).reset_index(drop=True)


def freeze_eligible_player_profiles(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Keep each player's latest eligible season and assign usage strata."""
    if player_seasons.empty:
        return pd.DataFrame()
    eligible = player_seasons[player_seasons["eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame()

    history = eligible.groupby("player_id").agg(
        eligible_season_count=("season", "nunique"),
        eligible_seasons=("season", lambda values: ",".join(sorted(set(values)))),
    )
    latest = (
        eligible.sort_values(["player_id", "season"])
        .drop_duplicates("player_id", keep="last")
        .rename(columns={"season": "profile_season"})
        .merge(history, left_on="player_id", right_index=True, how="left")
    )
    latest["player_id"] = pd.to_numeric(latest["player_id"], errors="raise").astype(int)
    latest["usage_tier"] = pd.Series(pd.NA, index=latest.index, dtype="string")

    for position in POSITION_GROUPS:
        index = latest.index[latest["position_group"] == position]
        if len(index) >= 3:
            ranks = latest.loc[index, "usage_rate"].rank(method="first")
            tiers = pd.qcut(ranks, q=3, labels=USAGE_TIERS)
            latest.loc[index, "usage_tier"] = tiers.astype("string").to_numpy()
        elif len(index) == 2:
            ordered = latest.loc[index].sort_values(["usage_rate", "player_id"]).index
            latest.loc[ordered[0], "usage_tier"] = "low"
            latest.loc[ordered[1], "usage_tier"] = "high"
        elif len(index) == 1:
            latest.loc[index[0], "usage_tier"] = "medium"

    latest = latest.dropna(subset=["position_group", "usage_tier"]).copy()
    latest["stratum"] = latest["position_group"] + "_" + latest["usage_tier"]
    return latest.sort_values(
        ["position_group", "usage_tier", "player_name", "player_id"]
    ).reset_index(drop=True)


def _strata_summary(
    eligible_players: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    required = {"position_group", "usage_tier", "usage_rate"}
    eligible = (
        eligible_players
        if required.issubset(eligible_players.columns)
        else pd.DataFrame(columns=sorted(required))
    )
    chosen_required = {"position_group", "usage_tier"}
    chosen_players = (
        selected
        if chosen_required.issubset(selected.columns)
        else pd.DataFrame(columns=sorted(chosen_required))
    )
    rows = []
    for position in POSITION_GROUPS:
        for usage_tier in USAGE_TIERS:
            candidates = eligible[
                (eligible["position_group"] == position)
                & (eligible["usage_tier"] == usage_tier)
            ]
            chosen = chosen_players[
                (chosen_players["position_group"] == position)
                & (chosen_players["usage_tier"] == usage_tier)
            ]
            rows.append(
                {
                    "position_group": position,
                    "usage_tier": usage_tier,
                    "stratum": f"{position}_{usage_tier}",
                    "eligible_players": len(candidates),
                    "selected_players": len(chosen),
                    "usage_rate_min": candidates["usage_rate"].min(),
                    "usage_rate_max": candidates["usage_rate"].max(),
                }
            )
    return pd.DataFrame(rows)


def select_stratified_players(
    eligible_players: pd.DataFrame,
    *,
    players_per_stratum: int = 10,
    random_state: int = 42,
    select_all: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample equally from position-by-usage strata with a fixed seed."""
    if players_per_stratum <= 0:
        raise ValueError("players_per_stratum must be positive.")
    if eligible_players.empty:
        return pd.DataFrame(), _strata_summary(eligible_players, pd.DataFrame())

    if select_all:
        selected = eligible_players.copy()
    else:
        rng = np.random.default_rng(random_state)
        selected_frames = []
        for position in POSITION_GROUPS:
            for usage_tier in USAGE_TIERS:
                candidates = eligible_players[
                    (eligible_players["position_group"] == position)
                    & (eligible_players["usage_tier"] == usage_tier)
                ].sort_values("player_id")
                if candidates.empty:
                    continue
                count = min(players_per_stratum, len(candidates))
                chosen_positions = np.sort(
                    rng.choice(len(candidates), size=count, replace=False)
                )
                selected_frames.append(candidates.iloc[chosen_positions])
        selected = (
            pd.concat(selected_frames, ignore_index=True)
            if selected_frames
            else pd.DataFrame(columns=eligible_players.columns)
        )

    selected = selected.copy()
    selected["selection_seed"] = int(random_state)
    selected["players_per_stratum"] = (
        pd.NA if select_all else int(players_per_stratum)
    )
    selected.sort_values(
        ["position_group", "usage_tier", "player_name", "player_id"],
        inplace=True,
    )
    selected.reset_index(drop=True, inplace=True)
    return selected, _strata_summary(eligible_players, selected)


def build_player_pool(
    seasons: Iterable[str],
    *,
    store_dir: str | Path | None = None,
    min_games: int = 30,
    min_average_minutes: float = 20.0,
    min_starter_rate: float = 0.50,
    min_data_coverage: float = 0.95,
    players_per_stratum: int = 10,
    random_state: int = 42,
    select_all: bool = False,
) -> PlayerPoolResult:
    """Load local seasons and create a frozen, auditable target-player pool."""
    season_list = list(dict.fromkeys(validate_season(season) for season in seasons))
    if not season_list:
        raise ValueError("At least one source season is required.")

    frames = []
    for season in season_list:
        if not season_store_exists(season, store_dir):
            raise FileNotFoundError(
                f"Local historical store is missing season {season}."
            )
        frames.append(load_season_player_games(season, store_dir=store_dir))
    games = pd.concat(frames, ignore_index=True)
    player_seasons = build_player_season_profiles(
        games,
        min_games=min_games,
        min_average_minutes=min_average_minutes,
        min_starter_rate=min_starter_rate,
        min_data_coverage=min_data_coverage,
    )
    eligible_players = freeze_eligible_player_profiles(player_seasons)
    selected, summary = select_stratified_players(
        eligible_players,
        players_per_stratum=players_per_stratum,
        random_state=random_state,
        select_all=select_all,
    )
    source_seasons = ",".join(season_list)
    audit_values = {
        "pool_source_seasons": source_seasons,
        "pool_min_games": int(min_games),
        "pool_min_average_minutes": float(min_average_minutes),
        "pool_min_starter_rate": float(min_starter_rate),
        "pool_min_data_coverage": float(min_data_coverage),
    }
    for frame in (eligible_players, selected):
        for column, value in audit_values.items():
            frame[column] = value
    return PlayerPoolResult(
        selected=selected,
        eligible_players=eligible_players,
        player_seasons=player_seasons,
        strata_summary=summary,
    )


def write_player_pool(
    result: PlayerPoolResult,
    output_path: str | Path,
    *,
    eligible_output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    eligible_output = (
        Path(eligible_output_path)
        if eligible_output_path is not None
        else output.with_name(f"{output.stem}_eligible{output.suffix or '.csv'}")
    )
    eligible_output.parent.mkdir(parents=True, exist_ok=True)
    result.selected.to_csv(output, index=False)
    result.eligible_players.to_csv(eligible_output, index=False)
    return output, eligible_output


def read_player_pool(player_pool_path: str | Path) -> list[int]:
    """Read unique target IDs from a pool CSV in its saved order."""
    path = Path(player_pool_path)
    frame = pd.read_csv(path)
    if "player_id" not in frame.columns:
        raise KeyError(f"Player-pool file {path} does not contain player_id.")
    ids = pd.to_numeric(frame["player_id"], errors="coerce")
    if ids.isna().any():
        raise ValueError(f"Player-pool file {path} contains invalid player IDs.")
    return list(dict.fromkeys(ids.astype(int).tolist()))


def _stable_player_order(
    frame: pd.DataFrame,
    *,
    random_state: int,
    phase: str,
) -> pd.DataFrame:
    """Order players reproducibly without relying on process-randomized hashes."""
    result = frame.copy()
    result["_nested_score"] = [
        hashlib.sha256(
            f"{random_state}:{phase}:{int(player_id)}".encode("utf-8")
        ).hexdigest()
        for player_id in result["player_id"]
    ]
    return result.sort_values(["_nested_score", "player_id"]).drop(
        columns="_nested_score"
    )


def _round_robin_strata(
    frame: pd.DataFrame,
    *,
    random_state: int,
    phase: str,
) -> list[int]:
    strata = [
        f"{position}_{usage_tier}"
        for position in POSITION_GROUPS
        for usage_tier in USAGE_TIERS
    ]
    if strata:
        offset = random_state % len(strata)
        strata = strata[offset:] + strata[:offset]

    queues: dict[str, list[int]] = {}
    for stratum in strata:
        candidates = frame[frame["stratum"] == stratum]
        ordered = _stable_player_order(
            candidates,
            random_state=random_state,
            phase=f"{phase}:{stratum}",
        )
        queues[stratum] = ordered.index.tolist()

    ordered_indices: list[int] = []
    level = 0
    while True:
        added = False
        for stratum in strata:
            queue = queues[stratum]
            if level < len(queue):
                ordered_indices.append(queue[level])
                added = True
        if not added:
            break
        level += 1
    return ordered_indices


def build_nested_player_pools(
    eligible_players: pd.DataFrame,
    *,
    pool_sizes: Iterable[int] = (30, 60, 90, 150, 250),
    anchor_players: pd.DataFrame | None = None,
    random_state: int = 42,
) -> dict[int, pd.DataFrame]:
    """Create nested, reproducible position-by-usage target pools.

    When an anchor is supplied, every smaller pool is selected from that anchor,
    the anchor itself appears as an exact prefix set, and larger pools add the
    remaining eligible players. This preserves the validated V2.0 90-player
    cohort while allowing controlled learning-curve experiments.
    """
    if eligible_players.empty:
        raise ValueError("eligible_players cannot be empty.")
    required = {"player_id", "stratum", "position_group", "usage_tier"}
    missing = sorted(required.difference(eligible_players.columns))
    if missing:
        raise KeyError(f"Eligible-player data is missing columns: {missing}")

    eligible = eligible_players.copy()
    eligible["player_id"] = pd.to_numeric(
        eligible["player_id"], errors="raise"
    ).astype(int)
    if eligible["player_id"].duplicated().any():
        raise ValueError("eligible_players contains duplicate player IDs.")

    sizes = sorted(set(int(size) for size in pool_sizes))
    if not sizes or sizes[0] <= 0:
        raise ValueError("pool_sizes must contain positive integers.")
    if sizes[-1] > len(eligible):
        raise ValueError(
            f"Largest requested pool has {sizes[-1]} players, but only "
            f"{len(eligible)} are eligible."
        )

    anchor_ids: set[int] = set()
    if anchor_players is not None and not anchor_players.empty:
        if "player_id" not in anchor_players:
            raise KeyError("anchor_players does not contain player_id.")
        anchor_ids = set(
            pd.to_numeric(anchor_players["player_id"], errors="raise").astype(int)
        )
        unknown = sorted(anchor_ids.difference(eligible["player_id"]))
        if unknown:
            raise ValueError(f"Anchor contains ineligible player IDs: {unknown}")

    anchored = eligible[eligible["player_id"].isin(anchor_ids)]
    remaining = eligible[~eligible["player_id"].isin(anchor_ids)]
    ordered_indices = _round_robin_strata(
        anchored,
        random_state=random_state,
        phase="anchor",
    )
    ordered_indices.extend(
        _round_robin_strata(
            remaining,
            random_state=random_state,
            phase="remaining",
        )
    )
    ordered = eligible.loc[ordered_indices].copy().reset_index(drop=True)
    ordered["nested_rank"] = np.arange(1, len(ordered) + 1)
    ordered["selection_seed"] = int(random_state)
    ordered["pool_design"] = "nested_round_robin_stratified"

    pools: dict[int, pd.DataFrame] = {}
    prior_ids: set[int] = set()
    for size in sizes:
        selected = ordered.head(size).copy()
        selected["pool_size"] = int(size)
        selected_ids = set(selected["player_id"])
        if not prior_ids.issubset(selected_ids):
            raise AssertionError("Nested player-pool construction lost prior members.")
        pools[size] = selected
        prior_ids = selected_ids

    if anchor_ids and len(anchor_ids) in pools:
        if set(pools[len(anchor_ids)]["player_id"]) != anchor_ids:
            raise AssertionError("The anchor-size pool does not preserve the anchor set.")
    return pools

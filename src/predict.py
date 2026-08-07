"""High-level saved-dataset and automatic live prediction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .baselines import compute_player_baseline
from .dataset import build_player_game_df
from .live_data import (
    build_dashboard_profiles,
    build_opponent_rates,
    lookup_player_game,
    resolve_player,
    season_from_game_date,
)
from .matchups import (
    apply_matchup_to_target_baseline,
    compute_profile_matchup_multipliers,
    project_target_matchup,
)
from .similarity import build_player_profiles, find_similar_players
from .trust import DEFAULT_MULTIPLIER_BOUNDS, MultiplierBounds


class PredictionUnavailableError(RuntimeError):
    """Raised when valid inputs exist but the model lacks enough prior data."""


def predict_target(
    player_game_df: pd.DataFrame,
    target_player_id: int,
    target_opp_abbr: str,
    as_of_date: Optional[pd.Timestamp] = None,
    candidate_player_ids: Optional[Iterable[int]] = None,
    manual_similar_player_ids: Optional[Sequence[int]] = None,
    top_n_similar: int = 10,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
    feature_weights: Optional[Mapping[str, float]] = None,
    position_weight: float = 0.35,
    require_same_starter_flag: bool = False,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = (
        DEFAULT_MULTIPLIER_BOUNDS
    ),
) -> dict[str, Any]:
    """Predict from an existing player-game dataset, primarily for backtests."""
    if player_game_df.empty:
        return {
            "projection": pd.Series(dtype="float64"),
            "similar_players": pd.DataFrame(),
            "profiles": pd.DataFrame(),
        }

    games = player_game_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    if as_of_date is None:
        as_of_date = games["game_date"].max()
    as_of_date = pd.to_datetime(as_of_date)

    if candidate_player_ids is not None:
        candidate_ids = {int(player_id) for player_id in candidate_player_ids}
        candidate_ids.add(int(target_player_id))
        profile_games = games[games["player_id"].isin(candidate_ids)].copy()
    else:
        profile_games = games

    profiles = build_player_profiles(
        profile_games,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        min_games=min_games,
        min_minutes_ratio=min_minutes_ratio,
    )
    if profiles.empty or target_player_id not in profiles.index:
        return {
            "projection": pd.Series(dtype="float64"),
            "similar_players": pd.DataFrame(),
            "profiles": profiles,
        }

    if manual_similar_player_ids is not None:
        similar_ids = [
            int(player_id)
            for player_id in manual_similar_player_ids
            if int(player_id) != target_player_id and int(player_id) in profiles.index
        ]
        similar_players = profiles.loc[similar_ids].copy() if similar_ids else pd.DataFrame()
    else:
        similar_players = find_similar_players(
            profiles,
            target_player_id=target_player_id,
            top_n=top_n_similar,
            feature_weights=feature_weights,
            position_weight=position_weight,
            require_same_starter_flag=require_same_starter_flag,
        )
        similar_ids = similar_players.index.tolist()

    projection = project_target_matchup(
        games,
        target_player_id=target_player_id,
        similar_player_ids=similar_ids,
        target_opp_abbr=target_opp_abbr,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        min_games=min_games,
        min_minutes_ratio=min_minutes_ratio,
        multiplier_bounds=multiplier_bounds,
    )
    if not projection.empty:
        projection = projection.copy()
        projection["similar_player_ids"] = similar_ids
        projection["n_similar_players"] = len(similar_ids)

    return {
        "projection": projection,
        "similar_players": similar_players,
        "profiles": profiles,
    }


def predict_player_game(
    player_query: str | int,
    game_date,
    *,
    season: str | None = None,
    candidate_player_ids: Optional[Iterable[int]] = None,
    manual_similar_player_ids: Optional[Sequence[int]] = None,
    top_n_similar: int = 10,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
    feature_weights: Optional[Mapping[str, float]] = None,
    position_weight: float = 0.35,
    require_same_starter_flag: bool = False,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = (
        DEFAULT_MULTIPLIER_BOUNDS
    ),
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Look up and predict any supported historical regular-season player-game."""
    target_player_id, target_player_name = resolve_player(player_query)
    game_date = pd.to_datetime(game_date).normalize()
    season = season or season_from_game_date(game_date)
    game = lookup_player_game(
        target_player_id,
        game_date,
        season=season,
        refresh=refresh,
        store_dir=store_dir,
    )
    as_of_date = game_date - pd.Timedelta(days=1)

    target_history = build_player_game_df(
        [target_player_id],
        season=season,
        season_type="Regular Season",
        date_from=as_of_date - pd.Timedelta(days=lookback_days),
        date_to=as_of_date,
        include_advanced=False,
        refresh=refresh,
        store_dir=store_dir,
    )
    target_baseline = compute_player_baseline(
        target_history,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        min_games=min_games,
        min_minutes_ratio=min_minutes_ratio,
    )
    if target_baseline.empty:
        raise PredictionUnavailableError(
            f"{target_player_name} does not have at least {min_games} qualifying prior games "
            f"with rebound tracking in the {lookback_days}-day window before {game_date.date()}."
        )

    profiles = build_dashboard_profiles(
        season=season,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        min_games=min_games,
        refresh=refresh,
        store_dir=store_dir,
    )
    if target_player_id not in profiles.index:
        raise PredictionUnavailableError(
            f"The NBA dashboard did not return a complete similarity profile for {target_player_name}."
        )

    opponent_rates = build_opponent_rates(
        season=season,
        as_of_date=as_of_date,
        opponent_team_id=int(game["opp_team_id"]),
        lookback_days=lookback_days,
        refresh=refresh,
        store_dir=store_dir,
    )
    eligible_opponent_ids = opponent_rates.dropna(
        subset=["fga_per_min", "reb_chances_per_min"]
    ).index
    eligible_ids = profiles.index.intersection(eligible_opponent_ids)

    if candidate_player_ids is not None:
        eligible_ids = eligible_ids.intersection(
            [int(player_id) for player_id in candidate_player_ids]
        )
    scoring_ids = eligible_ids.union(pd.Index([target_player_id]))
    scoring_profiles = profiles.loc[profiles.index.intersection(scoring_ids)].copy()

    if manual_similar_player_ids is not None:
        requested = [
            int(player_id)
            for player_id in manual_similar_player_ids
            if int(player_id) != target_player_id
        ]
        similar_ids = [player_id for player_id in requested if player_id in eligible_ids]
        if not similar_ids:
            raise PredictionUnavailableError(
                "None of the manually selected players had both a valid profile and a game "
                f"against {game['opp_abbr']} in the lookback window."
            )
        all_scores = find_similar_players(
            scoring_profiles,
            target_player_id=target_player_id,
            top_n=len(scoring_profiles),
            feature_weights=feature_weights,
            position_weight=position_weight,
            require_same_starter_flag=require_same_starter_flag,
        )
        similar_players = all_scores.loc[all_scores.index.intersection(similar_ids)].copy()
        similar_ids = similar_players.index.tolist()
    else:
        similar_players = find_similar_players(
            scoring_profiles,
            target_player_id=target_player_id,
            top_n=top_n_similar,
            feature_weights=feature_weights,
            position_weight=position_weight,
            require_same_starter_flag=require_same_starter_flag,
        )
        similar_ids = similar_players.index.tolist()

    if not similar_ids:
        raise PredictionUnavailableError(
            f"No eligible similar players faced {game['opp_abbr']} in the {lookback_days}-day window."
        )

    multipliers = compute_profile_matchup_multipliers(
        profiles,
        opponent_rates,
        similar_player_ids=similar_ids,
        target_opp_abbr=str(game["opp_abbr"]),
        multiplier_bounds=multiplier_bounds,
    )
    projection = apply_matchup_to_target_baseline(target_baseline, multipliers)
    projection["game_date"] = game_date
    projection["game_id"] = game["game_id"]
    projection["season"] = season
    projection["similar_player_ids"] = similar_ids
    projection["n_similar_players"] = len(similar_ids)

    return {
        "target_player_id": target_player_id,
        "target_player_name": target_player_name,
        "game": game,
        "projection": projection,
        "target_baseline": target_baseline,
        "similar_players": similar_players,
        "profiles": profiles,
        "opponent_rates": opponent_rates,
        "matchup_multipliers": multipliers,
    }

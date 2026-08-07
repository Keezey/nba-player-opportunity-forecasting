"""Player role profiles and similarity scoring.

This module turns the manual scouting idea of "who plays like this target?" into
a repeatable profile comparison. It intentionally excludes assist features for
now and focuses on minutes, shot volume, rebound chances, usage, touches,
listed position, and starter/team-role signal.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


PROFILE_NUMERIC_FEATURES = [
    "minutes",
    "fga_per_min",
    "reb_chances_per_min",
    "usage_rate",
    "touches_per_min",
    "starter_rate",
]

DEFAULT_FEATURE_WEIGHTS = {
    "minutes": 1.0,
    "fga_per_min": 1.25,
    "reb_chances_per_min": 1.5,
    "usage_rate": 1.25,
    "touches_per_min": 1.25,
    "starter_rate": 0.75,
}


def _mode_or_na(values: pd.Series):
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.mode().iloc[0]


def build_player_profile(
    player_games: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
) -> pd.Series:
    """Build one player's recent role profile.

    The input should contain games for exactly one player. Returns an empty
    Series when there are not enough qualifying games.
    """
    if player_games.empty:
        return pd.Series(dtype="float64")

    games = player_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    unique_ids = games["player_id"].unique()
    if len(unique_ids) > 1:
        raise ValueError("build_player_profile expects data for one player_id.")

    if as_of_date is None:
        as_of_date = games["game_date"].max()
    as_of_date = pd.to_datetime(as_of_date)

    window_start = as_of_date - pd.Timedelta(days=lookback_days)
    recent = games[(games["game_date"] >= window_start) & (games["game_date"] <= as_of_date)].copy()
    if recent.empty:
        return pd.Series(dtype="float64")

    usual_minutes = recent["minutes"].mean()
    min_minutes = usual_minutes * min_minutes_ratio
    qual = recent[recent["minutes"] >= min_minutes].copy()

    if len(qual) < min_games:
        return pd.Series(dtype="float64")

    minutes = qual["minutes"].mean()
    fga_per_min = (qual["fga"] / qual["minutes"]).mean()
    reb_chances_per_min = (qual["reb_chances"] / qual["minutes"]).mean()

    if "touches" in qual.columns:
        touches_per_min = (pd.to_numeric(qual["touches"], errors="coerce") / qual["minutes"]).mean()
    else:
        touches_per_min = np.nan

    if "usage_rate" in qual.columns:
        usage_rate = pd.to_numeric(qual["usage_rate"], errors="coerce").mean()
    else:
        usage_rate = np.nan

    if "starter_flag" in qual.columns:
        starter_values = pd.to_numeric(qual["starter_flag"], errors="coerce").fillna(0)
    else:
        starter_values = pd.Series(0, index=qual.index)
    starter_rate = starter_values.mean()
    starter_flag = int(starter_rate >= 0.5)

    listed_position = _mode_or_na(qual["listed_position"]) if "listed_position" in qual.columns else pd.NA

    return pd.Series(
        {
            "player_id": unique_ids[0],
            "as_of_date": as_of_date,
            "games_used": len(qual),
            "lookback_days": lookback_days,
            "minutes": minutes,
            "fga_per_min": fga_per_min,
            "reb_chances_per_min": reb_chances_per_min,
            "usage_rate": usage_rate,
            "touches_per_min": touches_per_min,
            "listed_position": listed_position,
            "starter_rate": starter_rate,
            "starter_flag": starter_flag,
        }
    )


def build_player_profiles(
    player_game_df: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
) -> pd.DataFrame:
    """Build recent role profiles for every player in a player-game dataset."""
    if player_game_df.empty:
        return pd.DataFrame()

    games = player_game_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    if as_of_date is None:
        as_of_date = games["game_date"].max()
    as_of_date = pd.to_datetime(as_of_date)

    profiles = []
    for _, group in games[games["game_date"] <= as_of_date].groupby("player_id"):
        profile = build_player_profile(
            group,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            min_games=min_games,
            min_minutes_ratio=min_minutes_ratio,
        )
        if not profile.empty:
            profiles.append(profile)

    if not profiles:
        return pd.DataFrame()

    result = pd.DataFrame(profiles)
    result.set_index("player_id", inplace=True)
    return result


def compute_similarity_scores(
    profiles: pd.DataFrame,
    target_player_id: int,
    feature_weights: Optional[Mapping[str, float]] = None,
    position_weight: float = 0.35,
    require_same_starter_flag: bool = False,
) -> pd.DataFrame:
    """Score every candidate profile against one target profile.

    Lower `similarity_score` means more similar. Listed position is a soft
    penalty rather than a hard filter because the project intentionally allows
    cross-position matches.
    """
    if profiles.empty:
        return pd.DataFrame()
    if target_player_id not in profiles.index:
        raise KeyError(f"target_player_id {target_player_id} is not in profiles.")

    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if feature_weights is not None:
        weights.update(feature_weights)

    features = [feature for feature in PROFILE_NUMERIC_FEATURES if feature in profiles.columns]
    if not features:
        raise ValueError("No numeric profile features are available for similarity scoring.")

    numeric = profiles[features].apply(pd.to_numeric, errors="coerce")
    features = [
        feature
        for feature in features
        if numeric[feature].notna().sum() >= 2 and pd.notna(numeric.loc[target_player_id, feature])
    ]
    if not features:
        raise ValueError("The target and candidate pool do not share usable profile features.")
    numeric = numeric[features]
    medians = numeric.median()
    numeric = numeric.fillna(medians)

    std = numeric.std(ddof=0).replace(0, 1).fillna(1)
    z = (numeric - numeric.mean()) / std

    target = z.loc[target_player_id]
    weighted_sq_diff = pd.DataFrame(index=z.index)
    total_weight = 0.0
    for feature in features:
        weight = float(weights.get(feature, 1.0))
        weighted_sq_diff[feature] = weight * (z[feature] - target[feature]) ** 2
        total_weight += weight

    numeric_distance = np.sqrt(weighted_sq_diff.sum(axis=1) / total_weight)

    def position_tokens(value) -> set[str]:
        if pd.isna(value):
            return set()
        return {token for token in str(value).upper().split("-") if token}

    target_position = profiles.loc[target_player_id].get("listed_position", pd.NA)
    target_position_tokens = position_tokens(target_position)
    if "listed_position" in profiles.columns and pd.notna(target_position):
        position_penalty = profiles["listed_position"].apply(
            lambda position: (
                0.0
                if not position_tokens(position)
                or target_position_tokens.intersection(position_tokens(position))
                else position_weight
            )
        )
    else:
        position_penalty = pd.Series(0.0, index=profiles.index)

    result = profiles.copy()
    result["numeric_distance"] = numeric_distance
    result["position_penalty"] = position_penalty
    result["similarity_score"] = result["numeric_distance"] + result["position_penalty"]

    result = result[result.index != target_player_id].copy()

    if require_same_starter_flag and "starter_flag" in result.columns:
        target_starter_flag = profiles.loc[target_player_id].get("starter_flag")
        result = result[result["starter_flag"] == target_starter_flag].copy()

    return result.sort_values("similarity_score")


def find_similar_players(
    profiles: pd.DataFrame,
    target_player_id: int,
    top_n: int = 10,
    feature_weights: Optional[Mapping[str, float]] = None,
    position_weight: float = 0.35,
    require_same_starter_flag: bool = False,
    exclude_player_ids: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Return the top-N most similar players to the target profile."""
    scores = compute_similarity_scores(
        profiles=profiles,
        target_player_id=target_player_id,
        feature_weights=feature_weights,
        position_weight=position_weight,
        require_same_starter_flag=require_same_starter_flag,
    )

    if exclude_player_ids:
        scores = scores[~scores.index.isin(exclude_player_ids)].copy()

    return scores.head(top_n)

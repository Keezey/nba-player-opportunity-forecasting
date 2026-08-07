"""Leakage-safe feature rows for version-two machine-learning models.

V1 remains responsible for basketball logic and the initial projection. This
module only translates a completed pregame V1 result into a stable tabular
schema, then attaches actual outcomes in a separate step for model training.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..matchups import compute_profile_matchup_multipliers


IDENTIFIER_COLUMNS = [
    "game_id",
    "game_date",
    "as_of_date",
    "season",
    "player_id",
    "opp_abbr",
]

NUMERIC_FEATURE_COLUMNS = [
    "baseline_games_used",
    "tracking_games_used",
    "usual_minutes",
    "min_minutes_threshold",
    "baseline_minutes",
    "baseline_fga",
    "baseline_reb",
    "baseline_reb_chances",
    "baseline_fga_per_min",
    "baseline_reb_chances_per_min",
    "baseline_reb_conversion",
    "profile_games_used",
    "profile_minutes",
    "profile_fga_per_min",
    "profile_reb_chances_per_min",
    "profile_usage_rate",
    "profile_touches_per_min",
    "profile_starter_rate",
    "profile_starter_flag",
    "n_similar_players",
    "similarity_score_min",
    "similarity_score_mean",
    "similarity_score_max",
    "similarity_score_std",
    "numeric_distance_mean",
    "position_penalty_mean",
    "raw_fga_multiplier",
    "fga_multiplier",
    "fga_n_samples",
    "fga_trust_weight",
    "raw_reb_chances_multiplier",
    "reb_chances_multiplier",
    "reb_chances_n_samples",
    "reb_chances_trust_weight",
    "v1_pred_minutes",
    "v1_pred_fga",
    "v1_pred_reb_chances",
    "v1_pred_reb",
]

CATEGORICAL_FEATURE_COLUMNS = ["profile_listed_position"]
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

TARGET_COLUMNS = [
    "actual_minutes",
    "actual_fga",
    "actual_reb_chances",
    "actual_reb",
    "fga_residual",
    "reb_chances_residual",
    "reb_residual",
]

TRAINING_ROW_COLUMNS = IDENTIFIER_COLUMNS + MODEL_FEATURE_COLUMNS + TARGET_COLUMNS


class FeatureLeakageError(ValueError):
    """Raised when a feature row is not strictly earlier than its target game."""


def _as_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, Mapping):
        return pd.Series(dict(value))
    raise TypeError(f"{name} must be a pandas Series or mapping.")


def _numeric(value: Any) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else np.nan


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    numerator_value = _numeric(numerator)
    denominator_value = _numeric(denominator)
    if not np.isfinite(numerator_value) or not np.isfinite(denominator_value):
        return np.nan
    if denominator_value == 0:
        return np.nan
    return numerator_value / denominator_value


def _summary(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if frame.empty or column not in frame.columns:
        return {"min": np.nan, "mean": np.nan, "max": np.nan, "std": np.nan}

    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"min": np.nan, "mean": np.nan, "max": np.nan, "std": np.nan}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def _target_profile(prediction_result: Mapping[str, Any], player_id: int) -> pd.Series:
    profiles = prediction_result.get("profiles")
    if not isinstance(profiles, pd.DataFrame) or profiles.empty:
        return pd.Series(dtype="object")
    if player_id not in profiles.index:
        return pd.Series(dtype="object")
    profile = profiles.loc[player_id]
    if isinstance(profile, pd.DataFrame):
        profile = profile.iloc[0]
    return profile


def _matchup_multipliers(
    prediction_result: Mapping[str, Any],
    projection: pd.Series,
    baseline: pd.Series,
) -> pd.Series:
    supplied = prediction_result.get("matchup_multipliers")
    if isinstance(supplied, pd.Series) and not supplied.empty:
        return supplied
    if isinstance(supplied, Mapping) and supplied:
        return pd.Series(dict(supplied))

    profiles = prediction_result.get("profiles")
    opponent_rates = prediction_result.get("opponent_rates")
    similar_ids = projection.get("similar_player_ids", [])
    opponent = projection.get("target_opp_abbr", pd.NA)
    if (
        isinstance(profiles, pd.DataFrame)
        and isinstance(opponent_rates, pd.DataFrame)
        and not profiles.empty
        and not opponent_rates.empty
        and similar_ids
        and pd.notna(opponent)
    ):
        return compute_profile_matchup_multipliers(
            profiles,
            opponent_rates,
            similar_player_ids=similar_ids,
            target_opp_abbr=str(opponent),
        )

    return pd.Series(
        {
            "raw_fga_multiplier": np.nan,
            "fga_multiplier": _safe_ratio(
                projection.get("pred_fga"), baseline.get("baseline_fga")
            ),
            "fga_n_samples": projection.get("fga_n_samples", 0),
            "fga_trust_weight": projection.get("fga_trust_weight", 0.0),
            "raw_reb_chances_multiplier": np.nan,
            "reb_chances_multiplier": _safe_ratio(
                projection.get("pred_reb_chances"),
                baseline.get("baseline_reb_chances"),
            ),
            "reb_chances_n_samples": projection.get("reb_chances_n_samples", 0),
            "reb_chances_trust_weight": projection.get(
                "reb_chances_trust_weight", 0.0
            ),
        }
    )


def build_pregame_feature_row(prediction_result: Mapping[str, Any]) -> pd.Series:
    """Convert one V1 result into features known before the target game.

    The function intentionally accepts no actual game statistics. It also
    verifies that the baseline cutoff is strictly before the game date.
    """
    projection = _as_series(prediction_result.get("projection"), "projection")
    baseline = _as_series(prediction_result.get("target_baseline"), "target_baseline")
    game = _as_series(prediction_result.get("game"), "game")
    if projection.empty or baseline.empty or game.empty:
        raise ValueError("prediction_result must contain non-empty projection, baseline, and game data.")

    player_id = int(prediction_result.get("target_player_id", baseline.get("player_id")))
    game_date = pd.to_datetime(projection.get("game_date", game.get("game_date"))).normalize()
    as_of_date = pd.to_datetime(baseline.get("as_of_date", projection.get("as_of_date"))).normalize()
    if pd.isna(game_date) or pd.isna(as_of_date):
        raise ValueError("Both game_date and as_of_date are required for leakage checks.")
    if as_of_date >= game_date:
        raise FeatureLeakageError(
            f"as_of_date {as_of_date.date()} must be before game_date {game_date.date()}."
        )

    profile = _target_profile(prediction_result, player_id)
    similar_players = prediction_result.get("similar_players")
    if not isinstance(similar_players, pd.DataFrame):
        similar_players = pd.DataFrame()
    similarity = _summary(similar_players, "similarity_score")
    numeric_distance = _summary(similar_players, "numeric_distance")
    position_penalty = _summary(similar_players, "position_penalty")
    multipliers = _matchup_multipliers(prediction_result, projection, baseline)

    row = {
        "game_id": str(projection.get("game_id", game.get("game_id"))),
        "game_date": game_date,
        "as_of_date": as_of_date,
        "season": projection.get("season", game.get("season", pd.NA)),
        "player_id": player_id,
        "opp_abbr": str(
            projection.get("target_opp_abbr", game.get("opp_abbr", pd.NA))
        ).upper(),
        "baseline_games_used": baseline.get("games_used", np.nan),
        "tracking_games_used": baseline.get("tracking_games_used", np.nan),
        "usual_minutes": baseline.get("usual_minutes", np.nan),
        "min_minutes_threshold": baseline.get("min_minutes_threshold", np.nan),
        "baseline_minutes": baseline.get("baseline_minutes", np.nan),
        "baseline_fga": baseline.get("baseline_fga", np.nan),
        "baseline_reb": baseline.get("baseline_reb", np.nan),
        "baseline_reb_chances": baseline.get("baseline_reb_chances", np.nan),
        "baseline_fga_per_min": baseline.get("fga_per_min", np.nan),
        "baseline_reb_chances_per_min": baseline.get(
            "reb_chances_per_min", np.nan
        ),
        "baseline_reb_conversion": baseline.get("reb_conversion", np.nan),
        "profile_games_used": profile.get("games_used", np.nan),
        "profile_minutes": profile.get("minutes", np.nan),
        "profile_fga_per_min": profile.get("fga_per_min", np.nan),
        "profile_reb_chances_per_min": profile.get(
            "reb_chances_per_min", np.nan
        ),
        "profile_usage_rate": profile.get("usage_rate", np.nan),
        "profile_touches_per_min": profile.get("touches_per_min", np.nan),
        "profile_starter_rate": profile.get("starter_rate", np.nan),
        "profile_starter_flag": profile.get("starter_flag", np.nan),
        "profile_listed_position": profile.get("listed_position", pd.NA),
        "n_similar_players": projection.get(
            "n_similar_players", len(similar_players)
        ),
        "similarity_score_min": similarity["min"],
        "similarity_score_mean": similarity["mean"],
        "similarity_score_max": similarity["max"],
        "similarity_score_std": similarity["std"],
        "numeric_distance_mean": numeric_distance["mean"],
        "position_penalty_mean": position_penalty["mean"],
        "raw_fga_multiplier": multipliers.get("raw_fga_multiplier", np.nan),
        "fga_multiplier": multipliers.get("fga_multiplier", np.nan),
        "fga_n_samples": multipliers.get("fga_n_samples", 0),
        "fga_trust_weight": multipliers.get("fga_trust_weight", 0.0),
        "raw_reb_chances_multiplier": multipliers.get(
            "raw_reb_chances_multiplier", np.nan
        ),
        "reb_chances_multiplier": multipliers.get(
            "reb_chances_multiplier", np.nan
        ),
        "reb_chances_n_samples": multipliers.get("reb_chances_n_samples", 0),
        "reb_chances_trust_weight": multipliers.get(
            "reb_chances_trust_weight", 0.0
        ),
        "v1_pred_minutes": projection.get("minutes_pred", np.nan),
        "v1_pred_fga": projection.get("pred_fga", np.nan),
        "v1_pred_reb_chances": projection.get("pred_reb_chances", np.nan),
        "v1_pred_reb": projection.get("pred_reb", np.nan),
    }
    return pd.Series(row).reindex(IDENTIFIER_COLUMNS + MODEL_FEATURE_COLUMNS)


def attach_training_targets(
    pregame_features: pd.Series | Mapping[str, Any],
    actual_game: pd.Series | Mapping[str, Any],
) -> pd.Series:
    """Append actual outcomes and V1 residual labels to pregame features."""
    features = _as_series(pregame_features, "pregame_features").copy()
    actual = _as_series(actual_game, "actual_game")

    feature_game_id = features.get("game_id")
    actual_game_id = actual.get("game_id")
    if pd.notna(feature_game_id) and pd.notna(actual_game_id):
        if str(feature_game_id) != str(actual_game_id):
            raise ValueError(
                f"Actual game_id {actual_game_id} does not match feature game_id {feature_game_id}."
            )

    feature_date = pd.to_datetime(features.get("game_date"), errors="coerce")
    actual_date = pd.to_datetime(actual.get("game_date"), errors="coerce")
    if pd.notna(feature_date) and pd.notna(actual_date):
        if feature_date.normalize() != actual_date.normalize():
            raise ValueError(
                f"Actual game_date {actual_date.date()} does not match feature game_date "
                f"{feature_date.date()}."
            )

    targets = {
        "actual_minutes": _numeric(actual.get("minutes")),
        "actual_fga": _numeric(actual.get("fga")),
        "actual_reb_chances": _numeric(actual.get("reb_chances")),
        "actual_reb": _numeric(actual.get("reb")),
    }
    targets["fga_residual"] = targets["actual_fga"] - _numeric(
        features.get("v1_pred_fga")
    )
    targets["reb_chances_residual"] = targets[
        "actual_reb_chances"
    ] - _numeric(features.get("v1_pred_reb_chances"))
    targets["reb_residual"] = targets["actual_reb"] - _numeric(
        features.get("v1_pred_reb")
    )

    return pd.concat([features, pd.Series(targets)]).reindex(TRAINING_ROW_COLUMNS)


def build_training_row(
    prediction_result: Mapping[str, Any],
    actual_game: pd.Series | Mapping[str, Any],
) -> pd.Series:
    """Build one complete V2 training row from a V1 result and its actuals."""
    features = build_pregame_feature_row(prediction_result)
    return attach_training_targets(features, actual_game)


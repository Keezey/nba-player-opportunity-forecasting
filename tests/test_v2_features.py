import numpy as np
import pandas as pd
import pytest

from src.v2.features import (
    MODEL_FEATURE_COLUMNS,
    FeatureLeakageError,
    attach_training_targets,
    build_pregame_feature_row,
    build_training_row,
)


def _prediction_result(as_of_date="2026-04-06"):
    baseline = pd.Series(
        {
            "player_id": 1,
            "as_of_date": pd.Timestamp(as_of_date),
            "games_used": 8,
            "tracking_games_used": 7,
            "usual_minutes": 31,
            "min_minutes_threshold": 23.25,
            "baseline_minutes": 32,
            "baseline_fga": 12,
            "baseline_reb": 7,
            "baseline_reb_chances": 14,
            "fga_per_min": 0.375,
            "reb_chances_per_min": 0.4375,
            "reb_conversion": 0.5,
        }
    )
    profiles = pd.DataFrame(
        {
            "games_used": [8, 7, 6],
            "minutes": [32, 31, 30],
            "fga_per_min": [0.375, 0.4, 0.5],
            "reb_chances_per_min": [0.4375, 0.4, 0.5],
            "usage_rate": [0.24, 0.25, 0.27],
            "touches_per_min": [1.7, 1.6, 1.8],
            "starter_rate": [1.0, 1.0, 0.8],
            "starter_flag": [1, 1, 1],
            "listed_position": ["G-F", "G", "F"],
        },
        index=[1, 2, 3],
    )
    opponent_rates = pd.DataFrame(
        {
            "fga_per_min": [0.48, 0.45],
            "fga_games": [2, 1],
            "reb_chances_per_min": [0.48, 0.45],
            "reb_chances_games": [2, 1],
        },
        index=[2, 3],
    )
    similar = profiles.loc[[2, 3]].copy()
    similar["numeric_distance"] = [0.1, 0.3]
    similar["position_penalty"] = [0.0, 0.35]
    similar["similarity_score"] = [0.1, 0.65]
    projection = pd.Series(
        {
            "player_id": 1,
            "game_id": "g1",
            "game_date": pd.Timestamp("2026-04-07"),
            "season": "2025-26",
            "target_opp_abbr": "LAL",
            "minutes_pred": 32,
            "pred_fga": 12.5,
            "pred_reb_chances": 14.5,
            "pred_reb": 7.25,
            "similar_player_ids": [2, 3],
            "n_similar_players": 2,
        }
    )
    return {
        "target_player_id": 1,
        "game": pd.Series(
            {
                "game_id": "g1",
                "game_date": pd.Timestamp("2026-04-07"),
                "opp_abbr": "LAL",
                "season": "2025-26",
            }
        ),
        "projection": projection,
        "target_baseline": baseline,
        "profiles": profiles,
        "opponent_rates": opponent_rates,
        "similar_players": similar,
    }


def test_pregame_features_exclude_actuals_and_capture_v1_inputs():
    row = build_pregame_feature_row(_prediction_result())

    assert row["game_id"] == "g1"
    assert row["as_of_date"] == pd.Timestamp("2026-04-06")
    assert row["profile_usage_rate"] == 0.24
    assert row["similarity_score_mean"] == 0.375
    assert row["similarity_score_std"] == 0.275
    assert row["raw_fga_multiplier"] == pytest.approx((1.2 * 2 + 0.9) / 3)
    assert all(not column.startswith("actual_") for column in row.index)
    assert all("residual" not in column for column in MODEL_FEATURE_COLUMNS)


def test_training_targets_are_attached_after_feature_creation():
    features = build_pregame_feature_row(_prediction_result())
    row = attach_training_targets(
        features,
        {
            "game_id": "g1",
            "game_date": "2026-04-07",
            "minutes": 34,
            "fga": 15,
            "reb_chances": 18,
            "reb": 10,
        },
    )

    assert row["actual_minutes"] == 34
    assert row["fga_residual"] == 2.5
    assert row["reb_chances_residual"] == 3.5
    assert row["reb_residual"] == 2.75


def test_build_training_row_allows_a_missing_tracking_target():
    row = build_training_row(
        _prediction_result(),
        {
            "game_id": "g1",
            "game_date": "2026-04-07",
            "minutes": 34,
            "fga": 15,
            "reb_chances": np.nan,
            "reb": 10,
        },
    )

    assert np.isnan(row["actual_reb_chances"])
    assert np.isnan(row["reb_chances_residual"])
    assert row["fga_residual"] == 2.5


def test_pregame_features_reject_same_day_cutoff():
    with pytest.raises(FeatureLeakageError):
        build_pregame_feature_row(_prediction_result(as_of_date="2026-04-07"))


def test_training_targets_reject_a_different_game():
    features = build_pregame_feature_row(_prediction_result())
    with pytest.raises(ValueError, match="does not match"):
        attach_training_targets(
            features,
            {
                "game_id": "g2",
                "game_date": "2026-04-08",
                "fga": 15,
                "reb_chances": 18,
                "reb": 10,
            },
        )

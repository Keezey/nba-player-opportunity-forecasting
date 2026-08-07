import os

import numpy as np
import pandas as pd
import pytest

from src.v2.features import MODEL_FEATURE_COLUMNS


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


@pytest.fixture
def v2_training_frame():
    rng = np.random.default_rng(7)
    row_count = 120
    index = np.arange(row_count)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(index // 2, unit="D")

    frame = pd.DataFrame(index=index)
    for column in MODEL_FEATURE_COLUMNS:
        frame[column] = 1.0

    frame["game_id"] = [f"g{i}" for i in index]
    frame["game_date"] = dates
    frame["as_of_date"] = dates - pd.Timedelta(days=1)
    frame["season"] = "2023-24"
    frame["player_id"] = 100 + index % 12
    frame["opp_abbr"] = np.where(index % 2 == 0, "BOS", "LAL")
    frame["profile_listed_position"] = np.where(index % 3 == 0, "G-F", "F")

    frame["baseline_games_used"] = 8 + index % 6
    frame["tracking_games_used"] = 7 + index % 5
    frame["usual_minutes"] = 28 + index % 8
    frame["min_minutes_threshold"] = frame["usual_minutes"] * 0.75
    frame["baseline_minutes"] = frame["usual_minutes"] + rng.normal(0, 0.5, row_count)
    frame["baseline_fga"] = 10 + 0.08 * index + rng.normal(0, 0.4, row_count)
    frame["baseline_reb_chances"] = 12 + 0.04 * index + rng.normal(0, 0.5, row_count)
    frame["baseline_reb_conversion"] = 0.52 + 0.03 * np.sin(index / 8)
    frame["baseline_reb"] = (
        frame["baseline_reb_chances"] * frame["baseline_reb_conversion"]
    )
    frame["baseline_fga_per_min"] = frame["baseline_fga"] / frame["baseline_minutes"]
    frame["baseline_reb_chances_per_min"] = (
        frame["baseline_reb_chances"] / frame["baseline_minutes"]
    )

    frame["profile_games_used"] = frame["baseline_games_used"]
    frame["profile_minutes"] = frame["baseline_minutes"]
    frame["profile_fga_per_min"] = frame["baseline_fga_per_min"] + rng.normal(
        0, 0.01, row_count
    )
    frame["profile_reb_chances_per_min"] = frame[
        "baseline_reb_chances_per_min"
    ] + rng.normal(0, 0.01, row_count)
    frame["profile_usage_rate"] = 0.18 + 0.0012 * index
    frame["profile_touches_per_min"] = 1.2 + 0.004 * index
    frame["profile_starter_rate"] = np.where(index % 5 == 0, 0.6, 1.0)
    frame["profile_starter_flag"] = 1

    frame["n_similar_players"] = 10
    frame["similarity_score_min"] = 0.12 + 0.001 * (index % 7)
    frame["similarity_score_mean"] = 0.35 + 0.002 * (index % 9)
    frame["similarity_score_max"] = 0.7 + 0.002 * (index % 11)
    frame["similarity_score_std"] = 0.15
    frame["numeric_distance_mean"] = frame["similarity_score_mean"] - 0.05
    frame["position_penalty_mean"] = np.where(index % 3 == 0, 0.05, 0.2)

    frame["raw_fga_multiplier"] = 0.9 + 0.0025 * index
    frame["fga_multiplier"] = 1 + 0.8 * (frame["raw_fga_multiplier"] - 1)
    frame["fga_n_samples"] = 8 + index % 8
    frame["fga_trust_weight"] = 0.7 + 0.02 * (index % 6)
    frame["raw_reb_chances_multiplier"] = 0.92 + 0.002 * index
    frame["reb_chances_multiplier"] = 1 + 0.8 * (
        frame["raw_reb_chances_multiplier"] - 1
    )
    frame["reb_chances_n_samples"] = 7 + index % 9
    frame["reb_chances_trust_weight"] = 0.68 + 0.02 * (index % 7)

    frame["v1_pred_minutes"] = frame["baseline_minutes"]
    frame["v1_pred_fga"] = frame["baseline_fga"] * frame["fga_multiplier"]
    frame["v1_pred_reb_chances"] = (
        frame["baseline_reb_chances"] * frame["reb_chances_multiplier"]
    )
    frame["v1_pred_reb"] = (
        frame["v1_pred_reb_chances"] * frame["baseline_reb_conversion"]
    )

    frame["fga_residual"] = (
        12 * (frame["profile_usage_rate"] - 0.24)
        + 1.5 * (frame["raw_fga_multiplier"] - 1)
        + rng.normal(0, 0.15, row_count)
    )
    frame["reb_chances_residual"] = (
        5 * (frame["profile_touches_per_min"] - 1.4)
        + 1.2 * (frame["raw_reb_chances_multiplier"] - 1)
        + rng.normal(0, 0.2, row_count)
    )
    frame["actual_fga"] = frame["v1_pred_fga"] + frame["fga_residual"]
    frame["actual_reb_chances"] = (
        frame["v1_pred_reb_chances"] + frame["reb_chances_residual"]
    )
    frame["actual_reb"] = (
        frame["actual_reb_chances"] * frame["baseline_reb_conversion"]
    )
    frame["actual_minutes"] = frame["baseline_minutes"] + rng.normal(
        0, 1, row_count
    )
    frame["reb_residual"] = frame["actual_reb"] - frame["v1_pred_reb"]
    return frame

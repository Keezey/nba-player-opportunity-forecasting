import numpy as np
import pandas as pd

from src.v2.conversion import (
    ConversionPrior,
    predict_shrunken_conversion,
    prepare_conversion_frame,
)
from src.v2.conversion_experiment import (
    run_conversion_experiment,
)


def _four_season_frame(v2_training_frame):
    frame = v2_training_frame.copy()
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    starts = ["2021-10-19", "2022-10-18", "2023-10-24", "2024-10-22"]
    rows_per_season = len(frame) // len(seasons)
    for index, (season, start) in enumerate(zip(seasons, starts)):
        positions = frame.index[
            index * rows_per_season : (index + 1) * rows_per_season
        ]
        dates = pd.Timestamp(start) + pd.to_timedelta(
            range(len(positions)), unit="D"
        )
        frame.loc[positions, "season"] = season
        frame.loc[positions, "game_date"] = dates
        frame.loc[positions, "as_of_date"] = dates - pd.Timedelta(days=1)
    frame["actual_minutes"] = frame["v1_pred_minutes"]
    conversion_adjustment = 0.5 * (frame["profile_usage_rate"] - 0.24)
    frame["actual_reb"] = frame["actual_reb_chances"] * (
        frame["baseline_reb_conversion"] + conversion_adjustment
    ).clip(0, 1)
    frame["reb_residual"] = frame["actual_reb"] - frame["v1_pred_reb"]
    return frame


def _base_predictions(frame, test_season):
    test = frame[frame["season"] == test_season]
    return pd.DataFrame(
        {
            "test_season": test_season,
            "training_cohort": "all",
            "model_type": "elastic_net",
            "bound_strategy": "fixed_0.50_1.50",
            "game_id": test["game_id"].to_numpy(),
            "player_id": test["player_id"].to_numpy(),
            "v2_pred_reb_chances": test["v1_pred_reb_chances"].to_numpy(),
        }
    )


def test_prepare_conversion_frame_handles_exposure_and_zero_chances():
    frame = pd.DataFrame(
        {
            "baseline_reb_conversion": [0.5, 0.6],
            "baseline_reb_chances": [10.0, 8.0],
            "tracking_games_used": [5, 4],
            "actual_reb": [6.0, 2.0],
            "actual_reb_chances": [12.0, 0.0],
        }
    )
    result = prepare_conversion_frame(frame)

    assert result["conversion_exposure"].tolist() == [50.0, 32.0]
    assert result.loc[0, "actual_reb_conversion"] == 0.5
    assert np.isnan(result.loc[1, "actual_reb_conversion"])


def test_shrinkage_moves_low_exposure_farther_toward_prior():
    frame = pd.DataFrame(
        {
            "baseline_reb_conversion": [0.8, 0.8],
            "baseline_reb_chances": [2.0, 20.0],
            "tracking_games_used": [1, 10],
            "profile_listed_position": ["F", "F"],
        }
    )
    prior = ConversionPrior(global_rate=0.5, position_rates={"F": 0.5})
    result = predict_shrunken_conversion(frame, prior, prior_strength=20)

    assert result.iloc[0] < result.iloc[1] < 0.8
    assert abs(result.iloc[0] - 0.5) < abs(result.iloc[1] - 0.5)


def test_conversion_experiment_uses_frozen_chances_and_scores_methods(
    v2_training_frame,
):
    frame = _four_season_frame(v2_training_frame)
    base = _base_predictions(frame, "2023-24")
    result = run_conversion_experiment(
        frame,
        base,
        test_seasons=["2023-24"],
        training_cohorts=["all"],
        shrinkage_strengths=[0, 25],
        elastic_n_iter=1,
        n_splits=2,
        min_rows=10,
        bootstrap_samples=50,
    )

    expected_rows = int((frame["season"] == "2023-24").sum())
    assert len(result.predictions) == expected_rows
    assert set(result.summary["method"]) == {
        "current",
        "shrunk",
        "elastic_net",
    }
    assert set(result.summary["evaluation_cohort"]) == {"all", "regular"}
    assert result.predictions["v2_pred_reb_chances"].notna().all()
    assert result.predictions[
        ["current_pred_reb", "shrunk_pred_reb", "elastic_net_pred_reb"]
    ].notna().all().all()
    parameters = result.parameters["2023-24"]["training_cohorts"]["all"]
    assert parameters["conversion_model_rows"] > 0
    assert 0 <= parameters["conversion_lower_bound"] < parameters[
        "conversion_upper_bound"
    ] <= 1
    assert set(result.bootstrap["method"]) == {"shrunk", "elastic_net"}
    assert result.bootstrap["bootstrap_samples"].eq(50).all()

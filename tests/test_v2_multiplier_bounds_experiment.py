import pandas as pd

from src.v2.multiplier_bounds_experiment import (
    derive_multiplier_bounds,
    run_multiplier_bounds_experiment,
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
    test_index = frame.index[frame["season"] == "2023-24"][0]
    frame.loc[test_index, "raw_fga_multiplier"] = 10.0
    frame.loc[test_index, "raw_reb_chances_multiplier"] = 10.0
    frame.loc[test_index, ["fga_n_samples", "reb_chances_n_samples"]] = 15
    return frame


def test_training_percentile_bounds_use_only_supplied_rows(v2_training_frame):
    frame = v2_training_frame.copy()
    frame.loc[frame.index[-1], "raw_reb_chances_multiplier"] = 100.0

    bounds = derive_multiplier_bounds(frame.iloc[:-1], "training_p01_p99")

    assert bounds.reb_chances_upper < 100
    assert bounds.fga_lower <= 1 <= bounds.fga_upper


def test_multiplier_bounds_experiment_compares_strategies(v2_training_frame):
    frame = _four_season_frame(v2_training_frame)
    result = run_multiplier_bounds_experiment(
        frame,
        test_seasons=["2023-24"],
        bound_strategies=["unbounded", "fixed_0.50_1.50"],
        training_cohorts=["all"],
        model_types=["elastic_net"],
        trust_n_iter=1,
        model_n_iter={"elastic_net": 1},
        n_splits=2,
        min_rows=10,
    )

    test_rows = int((frame["season"] == "2023-24").sum())
    assert len(result.predictions) == 2 * test_rows
    assert set(result.predictions["bound_strategy"]) == {
        "unbounded",
        "fixed_0.50_1.50",
    }
    bounded = result.predictions[
        result.predictions["bound_strategy"] == "fixed_0.50_1.50"
    ]
    assert bounded["fga_multiplier_was_bounded"].any()
    assert set(result.summary["evaluation_cohort"]) == {
        "all",
        "regular",
    }

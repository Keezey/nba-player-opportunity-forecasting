import pandas as pd

import src.v2.rolling_experiment as rolling


def _four_season_frame(v2_training_frame):
    frame = v2_training_frame.copy()
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    starts = ["2021-10-19", "2022-10-18", "2023-10-24", "2024-10-22"]
    rows_per_season = len(frame) // len(seasons)
    for index, (season, start) in enumerate(zip(seasons, starts)):
        row_start = index * rows_per_season
        row_end = (index + 1) * rows_per_season
        positions = frame.index[row_start:row_end]
        frame.loc[positions, "season"] = season
        dates = pd.Timestamp(start) + pd.to_timedelta(
            range(len(positions)), unit="D"
        )
        frame.loc[positions, "game_date"] = dates
        frame.loc[positions, "as_of_date"] = dates - pd.Timedelta(days=1)
    return frame


def test_rolling_folds_use_only_complete_prior_seasons(v2_training_frame):
    frame = _four_season_frame(v2_training_frame)
    folds = rolling.build_rolling_season_folds(
        frame,
        test_seasons=["2023-24", "2024-25"],
        min_training_seasons=2,
    )

    assert [fold.test_season for fold in folds] == ["2023-24", "2024-25"]
    assert folds[0].training_seasons == ("2021-22", "2022-23")
    assert folds[1].training_seasons == (
        "2021-22",
        "2022-23",
        "2023-24",
    )
    for fold in folds:
        assert fold.training["game_date"].max() < fold.test["game_date"].min()
        assert fold.test["season"].eq(fold.test_season).all()
        assert fold.test_season not in set(fold.training["season"])


def test_rolling_experiment_retunes_inside_each_outer_fold(
    monkeypatch,
    v2_training_frame,
):
    frame = _four_season_frame(v2_training_frame)
    test_starts = {
        season: frame.loc[frame["season"] == season, "game_date"].min()
        for season in ["2023-24", "2024-25"]
    }
    trust_training_ends = []
    model_training_ends = []
    original_trust = rolling.tune_trust_parameters
    original_models = rolling.tune_model_parameters

    def tracked_trust(training, **kwargs):
        trust_training_ends.append(training["game_date"].max())
        return original_trust(training, **kwargs)

    def tracked_models(training, **kwargs):
        model_training_ends.append(training["game_date"].max())
        return original_models(training, **kwargs)

    monkeypatch.setattr(rolling, "tune_trust_parameters", tracked_trust)
    monkeypatch.setattr(rolling, "tune_model_parameters", tracked_models)

    result = rolling.run_rolling_season_experiment(
        frame,
        test_seasons=["2023-24", "2024-25"],
        model_types=["elastic_net"],
        trust_n_iter=1,
        model_n_iter={"elastic_net": 1},
        n_splits=2,
        min_rows=10,
    )

    assert len(trust_training_ends) == 2
    assert len(model_training_ends) == 2
    for season, trust_end, model_end in zip(
        ["2023-24", "2024-25"], trust_training_ends, model_training_ends
    ):
        assert trust_end < test_starts[season]
        assert model_end < test_starts[season]

    expected_prediction_rows = len(
        frame[frame["season"].isin(["2023-24", "2024-25"])]
    )
    assert len(result.predictions) == expected_prediction_rows
    assert set(result.parameters) == {"2023-24", "2024-25"}
    assert set(result.predictions["test_season"]) == {"2023-24", "2024-25"}
    all_game_weighted = result.summary[
        (result.summary["scope"] == "all")
        & (result.summary["aggregation"] == "game_weighted")
    ]
    assert len(all_game_weighted) == 2 * 3
    assert set(all_game_weighted["stat"]) == {"fga", "reb_chances", "reb"}

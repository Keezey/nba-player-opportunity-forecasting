import numpy as np
import pandas as pd

from src.v2.regular_minutes_experiment import (
    annotate_regular_minutes,
    run_regular_minutes_experiment,
)


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
    irregular = frame.index % 4 == 0
    frame.loc[~irregular, "actual_minutes"] = frame.loc[
        ~irregular, "v1_pred_minutes"
    ] + 1.0
    frame.loc[irregular, "actual_minutes"] = frame.loc[
        irregular, "v1_pred_minutes"
    ] - 12.0
    return frame


def test_regular_minutes_annotation_is_symmetric_and_hybrid():
    frame = pd.DataFrame(
        {
            "actual_minutes": [34.5, 34.6, 13.0, 7.0, 45.0],
            "v1_pred_minutes": [30.0, 30.0, 10.0, 10.0, 30.0],
        }
    )
    result = annotate_regular_minutes(
        frame,
        absolute_tolerance=3.0,
        relative_tolerance=0.15,
    )

    assert result["regular_minutes_tolerance"].tolist() == [4.5, 4.5, 3, 3, 4.5]
    assert result["regular_minutes_flag"].tolist() == [True, False, True, True, False]
    assert np.isclose(result.loc[0, "absolute_minute_error"], 4.5)


def test_regular_minutes_experiment_compares_training_cohorts(v2_training_frame):
    frame = _four_season_frame(v2_training_frame)
    result = run_regular_minutes_experiment(
        frame,
        test_seasons=["2023-24"],
        model_types=["elastic_net"],
        training_cohorts=["all", "regular"],
        trust_n_iter=1,
        model_n_iter={"elastic_net": 1},
        n_splits=2,
        min_rows=10,
    )

    test_rows = int((frame["season"] == "2023-24").sum())
    regular_test_rows = int(
        annotate_regular_minutes(frame[frame["season"] == "2023-24"])[
            "regular_minutes_flag"
        ].sum()
    )
    assert len(result.predictions) == 2 * test_rows
    assert set(result.predictions["training_cohort"]) == {"all", "regular"}
    assert result.parameters["2023-24"]["regular_test_rows"] == regular_test_rows

    primary = result.summary[
        (result.summary["evaluation_cohort"] == "regular")
        & (result.summary["aggregation"] == "game_weighted")
    ]
    assert len(primary) == 2 * 3
    assert primary["test_rows"].eq(regular_test_rows).all()
    assert set(primary["stat"]) == {"fga", "reb_chances", "reb"}

    cohorts = result.parameters["2023-24"]["training_cohorts"]
    assert cohorts["regular"]["training_rows"] < cohorts["all"]["training_rows"]
    assert cohorts["regular"]["training_coverage_pct"] == 75.0

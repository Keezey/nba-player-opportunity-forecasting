import pandas as pd
import pytest

import src.v2.training_data as training_data
from src.predict import PredictionUnavailableError
from src.v2.features import TRAINING_ROW_COLUMNS


def _raw_games():
    return pd.DataFrame(
        {
            "Game_ID": ["g1", "g2"],
            "GAME_DATE": ["Mar 1, 2026", "Mar 3, 2026"],
            "Player_ID": [1, 1],
            "MIN": [30, 31],
            "FGA": [10, 12],
            "REB": [6, 7],
            "MATCHUP": ["AAA @ BOS", "AAA vs. LAL"],
        }
    )


def _actual_games():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "game_date": pd.to_datetime(["2026-03-01", "2026-03-03"]),
            "player_id": [1, 1],
            "opp_abbr": ["BOS", "LAL"],
            "minutes": [30, 31],
            "fga": [10, 12],
            "reb": [6, 7],
            "reb_chances": [12, 14],
            "touches": [50, 52],
            "usage_rate": [pd.NA, pd.NA],
            "listed_position": ["G", "G"],
            "starter_flag": [1, 1],
        }
    )


def _prediction_result(game_date):
    game_date = pd.to_datetime(game_date)
    baseline = pd.Series(
        {
            "player_id": 1,
            "as_of_date": game_date - pd.Timedelta(days=1),
            "games_used": 5,
            "tracking_games_used": 5,
            "usual_minutes": 30,
            "min_minutes_threshold": 22.5,
            "baseline_minutes": 30,
            "baseline_fga": 9,
            "baseline_reb": 6,
            "baseline_reb_chances": 12,
            "fga_per_min": 0.3,
            "reb_chances_per_min": 0.4,
            "reb_conversion": 0.5,
        }
    )
    profiles = pd.DataFrame(
        {
            "games_used": [5, 5],
            "minutes": [30, 31],
            "fga_per_min": [0.3, 0.32],
            "reb_chances_per_min": [0.4, 0.42],
            "usage_rate": [0.24, 0.25],
            "touches_per_min": [1.6, 1.65],
            "starter_rate": [1.0, 1.0],
            "starter_flag": [1, 1],
            "listed_position": ["G", "G-F"],
        },
        index=[1, 2],
    )
    similar = profiles.loc[[2]].copy()
    similar["numeric_distance"] = 0.1
    similar["position_penalty"] = 0.0
    similar["similarity_score"] = 0.1
    game_id = "g1" if game_date.day == 1 else "g2"
    opponent = "BOS" if game_id == "g1" else "LAL"
    projection = pd.Series(
        {
            "player_id": 1,
            "game_id": game_id,
            "game_date": game_date,
            "season": "2025-26",
            "target_opp_abbr": opponent,
            "minutes_pred": 30,
            "pred_fga": 9.5,
            "pred_reb_chances": 12.5,
            "pred_reb": 6.25,
            "similar_player_ids": [2],
            "n_similar_players": 1,
        }
    )
    return {
        "target_player_id": 1,
        "target_player_name": "Target",
        "game": pd.Series(
            {
                "game_id": game_id,
                "game_date": game_date,
                "season": "2025-26",
                "opp_abbr": opponent,
            }
        ),
        "projection": projection,
        "target_baseline": baseline,
        "profiles": profiles,
        "similar_players": similar,
        "opponent_rates": pd.DataFrame(),
        "matchup_multipliers": {
            "raw_fga_multiplier": 1.1,
            "fga_multiplier": 1.05,
            "fga_n_samples": 8,
            "fga_trust_weight": 0.6,
            "raw_reb_chances_multiplier": 1.1,
            "reb_chances_multiplier": 1.05,
            "reb_chances_n_samples": 8,
            "reb_chances_trust_weight": 0.6,
        },
    }


def test_seasons_for_date_range_can_span_multiple_seasons():
    assert training_data.seasons_for_date_range("2024-10-01", "2026-04-01") == [
        "2024-25",
        "2025-26",
    ]


def test_player_training_dataset_runs_v1_and_records_skips(monkeypatch):
    monkeypatch.setattr(training_data, "resolve_player", lambda query: (1, "Target"))
    monkeypatch.setattr(training_data, "fetch_player_gamelog", lambda *args, **kwargs: _raw_games())
    monkeypatch.setattr(training_data, "build_player_game_df", lambda *args, **kwargs: _actual_games())

    calls = []

    def fake_predict(player_query, game_date, **kwargs):
        date = pd.to_datetime(game_date)
        calls.append(date)
        if date.day == 3:
            raise PredictionUnavailableError("not enough prior games")
        return _prediction_result(date)

    monkeypatch.setattr(training_data, "predict_player_game", fake_predict)

    result = training_data.build_player_training_dataset(
        "Target",
        "2026-03-01",
        "2026-03-03",
        refresh=True,
    )

    assert calls == [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-03")]
    assert result.games_found == 2
    assert result.predictions_built == 1
    assert result.rows.columns.tolist() == TRAINING_ROW_COLUMNS
    assert result.rows.loc[0, "fga_residual"] == 0.5
    assert result.skipped.loc[0, "error_type"] == "PredictionUnavailableError"


def test_multiple_player_results_are_combined_chronologically(monkeypatch):
    first = training_data.TrainingDatasetResult(
        rows=pd.DataFrame(
            [{"game_date": pd.Timestamp("2026-03-03"), "player_id": 2}]
        ).reindex(columns=TRAINING_ROW_COLUMNS),
        skipped=pd.DataFrame(columns=training_data.SKIPPED_ROW_COLUMNS),
    )
    second = training_data.TrainingDatasetResult(
        rows=pd.DataFrame(
            [{"game_date": pd.Timestamp("2026-03-01"), "player_id": 1}]
        ).reindex(columns=TRAINING_ROW_COLUMNS),
        skipped=pd.DataFrame(columns=training_data.SKIPPED_ROW_COLUMNS),
    )
    results = iter([first, second])
    monkeypatch.setattr(
        training_data,
        "build_player_training_dataset",
        lambda *args, **kwargs: next(results),
    )

    progress_events = []
    combined = training_data.build_training_dataset(
        ["Player 2", "Player 1"],
        "2026-03-01",
        "2026-03-03",
        progress=lambda *values: progress_events.append(values),
    )

    assert combined.rows["player_id"].tolist() == [1, 2]
    assert progress_events[-1][:2] == (2, 2)
    assert progress_events[-1][3:] == (2, 0)


def test_training_dataset_rejects_nonpositive_worker_count():
    with pytest.raises(ValueError, match="workers must be positive"):
        training_data.build_training_dataset(
            [],
            "2026-03-01",
            "2026-03-03",
            workers=0,
        )


def test_training_dataset_can_be_written_and_read(tmp_path):
    rows = pd.DataFrame(
        [{"game_id": "g1", "game_date": pd.Timestamp("2026-03-01"), "player_id": 1}]
    ).reindex(columns=TRAINING_ROW_COLUMNS)
    skipped = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Target",
                "game_id": "g2",
                "game_date": pd.Timestamp("2026-03-03"),
                "season": "2025-26",
                "opp_abbr": "LAL",
                "error_type": "PredictionUnavailableError",
                "message": "not enough prior games",
            }
        ]
    ).reindex(columns=training_data.SKIPPED_ROW_COLUMNS)
    result = training_data.TrainingDatasetResult(rows=rows, skipped=skipped)

    output, skipped_output = training_data.write_training_dataset(
        result,
        tmp_path / "training.parquet",
        skipped_path=tmp_path / "skipped.csv",
    )

    assert pd.read_parquet(output).loc[0, "game_id"] == "g1"
    assert pd.read_csv(skipped_output).loc[0, "game_id"] == "g2"

import pandas as pd
import pytest

import src.dataset as dataset
import src.historical_store as historical_store
import src.live_data as live_data
import src.predict as prediction


SEASON = "2025-26"


def _base_logs():
    rows = [
        ("g1", "2025-10-22", 1, "Target", 10, "AAA", "AAA vs. BBB", "30:00", 12, 6),
        ("g1", "2025-10-22", 2, "Bench", 10, "AAA", "AAA vs. BBB", "10:00", 3, 2),
        ("g1", "2025-10-22", 3, "Opponent", 20, "BBB", "BBB @ AAA", "32:00", 15, 7),
        ("g1", "2025-10-22", 6, "DNP", 20, "BBB", "BBB @ AAA", "0:00", 0, 0),
        ("g2", "2025-10-25", 1, "Target", 10, "AAA", "AAA @ CCC", "40:00", 20, 8),
        ("g2", "2025-10-25", 4, "Opponent Two", 30, "CCC", "CCC vs. AAA", "30:00", 10, 5),
        ("g3", "2025-10-28", 1, "Target", 10, "AAA", "AAA vs. DDD", "35:00", 14, 7),
        ("g3", "2025-10-28", 5, "Opponent Three", 40, "DDD", "DDD @ AAA", "31:00", 11, 6),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "MATCHUP",
            "MIN",
            "FGA",
            "REB",
        ],
    )


def _advanced_logs():
    return pd.DataFrame(
        [
            ("g1", 1, 0.20),
            ("g1", 2, 0.10),
            ("g1", 3, 0.24),
            ("g2", 1, 0.30),
            ("g2", 4, 0.18),
            ("g3", 1, 0.25),
            ("g3", 5, 0.19),
        ],
        columns=["GAME_ID", "PLAYER_ID", "USG_PCT"],
    )


def _tracking_logs():
    return pd.DataFrame(
        [
            ("g1", 1, "G", 12, 60),
            ("g1", 2, "", 4, 10),
            ("g1", 3, "C", 14, 55),
            ("g2", 1, "G", 16, 80),
            ("g2", 4, "F", 10, 45),
        ],
        columns=[
            "gameId",
            "personId",
            "position",
            "reboundChancesTotal",
            "touches",
        ],
    )


def _assembled_store():
    return historical_store.assemble_season_player_games(
        _base_logs(),
        _advanced_logs(),
        _tracking_logs(),
        season=SEASON,
    )


def _write_test_store(tmp_path):
    frame = _assembled_store()
    manifest = {
        "schema_version": 1,
        "season": SEASON,
        "rows": len(frame),
    }
    historical_store.write_season_store(
        frame,
        manifest,
        season=SEASON,
        store_dir=tmp_path,
    )
    return frame


def test_assemble_season_store_preserves_rows_and_availability():
    frame = _assembled_store()

    assert len(frame) == 7
    assert 6 not in frame["player_id"].tolist()

    target_first = frame[(frame["game_id"] == "g1") & (frame["player_id"] == 1)].iloc[0]
    assert target_first["opp_team_id"] == 20
    assert target_first["opp_abbr"] == "BBB"
    assert target_first["reb_chances"] == 12
    assert target_first["touches"] == 60
    assert target_first["usage_rate"] == pytest.approx(0.20)
    assert target_first["starter_flag"] == 1

    bench = frame[(frame["game_id"] == "g1") & (frame["player_id"] == 2)].iloc[0]
    assert bench["starter_flag"] == 0
    assert pd.isna(bench["listed_position"])

    missing_tracking = frame[
        (frame["game_id"] == "g3") & (frame["player_id"] == 1)
    ].iloc[0]
    assert pd.isna(missing_tracking["starter_flag"])
    assert not missing_tracking["tracking_available"]
    assert missing_tracking["advanced_available"]


def test_build_season_store_writes_resumable_partition(monkeypatch, tmp_path):
    def fake_season_logs(season, season_type, measure_type, *, refresh=False):
        assert season == SEASON
        assert season_type == "Regular Season"
        return _base_logs() if measure_type == "Base" else _advanced_logs()

    monkeypatch.setattr(
        historical_store,
        "fetch_season_player_game_logs",
        fake_season_logs,
    )
    monkeypatch.setattr(
        historical_store,
        "fetch_tracking_for_games",
        lambda game_ids, **kwargs: _tracking_logs(),
    )

    result = historical_store.build_season_store(
        SEASON,
        store_dir=tmp_path,
        verbose=False,
    )

    assert result.data_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest["rows"] == 7
    assert result.manifest["games"] == 3
    assert result.manifest["missing_tracking_games"] == ["g3"]
    assert not result.manifest["complete_tracking"]

    loaded = historical_store.load_season_player_games(
        SEASON,
        player_ids=[1],
        date_from="2025-10-25",
        store_dir=tmp_path,
    )
    assert loaded["game_id"].tolist() == ["g2", "g3"]


def test_local_season_partition_is_cached_without_sharing_mutations(
    monkeypatch, tmp_path
):
    _write_test_store(tmp_path)
    historical_store._read_season_partition.cache_clear()
    real_read_parquet = pd.read_parquet
    calls = []

    def counted_read(path, *args, **kwargs):
        calls.append(path)
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted_read)
    first = historical_store.load_season_player_games(
        SEASON,
        player_ids=[1],
        store_dir=tmp_path,
    )
    first.loc[:, "fga"] = -1
    second = historical_store.load_season_player_games(
        SEASON,
        player_ids=[1],
        store_dir=tmp_path,
    )

    assert len(calls) == 1
    assert (second["fga"] >= 0).all()


def test_local_profiles_and_opponent_rates_use_correct_denominators(tmp_path):
    _write_test_store(tmp_path)

    profiles = historical_store.build_local_dashboard_profiles(
        season=SEASON,
        as_of_date="2025-10-28",
        lookback_days=30,
        min_games=3,
        store_dir=tmp_path,
    )
    target = profiles.loc[1]
    assert target["games_used"] == 3
    assert target["minutes"] == pytest.approx(35.0)
    assert target["fga_per_min"] == pytest.approx(46 / 105)
    assert target["reb_chances_per_min"] == pytest.approx(28 / 70)
    assert target["touches_per_min"] == pytest.approx(140 / 70)
    assert target["usage_rate"] == pytest.approx((0.20 * 30 + 0.30 * 40 + 0.25 * 35) / 105)
    assert target["starter_rate"] == pytest.approx(1.0)
    assert target["listed_position"] == "G"

    rates = historical_store.build_local_opponent_rates(
        season=SEASON,
        as_of_date="2025-10-28",
        opponent_team_id=20,
        lookback_days=30,
        store_dir=tmp_path,
    )
    assert rates.loc[1, "fga_per_min"] == pytest.approx(12 / 30)
    assert rates.loc[1, "fga_games"] == 1
    assert rates.loc[1, "reb_chances_per_min"] == pytest.approx(12 / 30)
    assert rates.loc[1, "reb_chances_games"] == 1


def test_existing_dataset_and_game_lookup_are_local_first(monkeypatch, tmp_path):
    _write_test_store(tmp_path)
    monkeypatch.setattr(
        dataset,
        "fetch_players_gamelogs",
        lambda *args, **kwargs: pytest.fail("NBA API should not be called"),
    )

    rows = dataset.build_player_game_df(
        [1],
        season=SEASON,
        date_from="2025-10-22",
        date_to="2025-10-25",
        store_dir=tmp_path,
    )
    assert rows["game_id"].tolist() == ["g1", "g2"]
    assert rows["reb_chances"].tolist() == [12, 16]

    game = live_data.lookup_player_game(
        1,
        "2025-10-22",
        season=SEASON,
        store_dir=tmp_path,
    )
    assert game["game_id"] == "g1"
    assert game["opp_team_id"] == 20
    assert game["opp_abbr"] == "BBB"


def _write_prediction_store(tmp_path):
    rows = []
    history_dates = pd.to_datetime(
        ["2025-10-01", "2025-10-05", "2025-10-09", "2025-10-13", "2025-10-17"]
    )
    for index, game_date in enumerate(history_dates, start=1):
        rows.append(
            {
                "season": SEASON,
                "game_id": f"target-{index}",
                "game_date": game_date,
                "player_id": 1,
                "player_name": "Target",
                "team_id": 10,
                "team_abbr": "AAA",
                "opp_team_id": 30,
                "opp_abbr": "CCC",
                "home_away": "H",
                "minutes": 30,
                "fga": 10 + index,
                "reb": 6,
                "reb_chances": 12,
                "touches": 50,
                "usage_rate": 0.22,
                "listed_position": "G",
                "starter_flag": 1,
                "tracking_available": True,
                "advanced_available": True,
            }
        )
        rows.append(
            {
                "season": SEASON,
                "game_id": f"similar-{index}",
                "game_date": game_date,
                "player_id": 2,
                "player_name": "Similar",
                "team_id": 11,
                "team_abbr": "DDD",
                "opp_team_id": 20,
                "opp_abbr": "BBB",
                "home_away": "A",
                "minutes": 31,
                "fga": 13,
                "reb": 6,
                "reb_chances": 13,
                "touches": 52,
                "usage_rate": 0.23,
                "listed_position": "G-F",
                "starter_flag": 1,
                "tracking_available": True,
                "advanced_available": True,
            }
        )

    rows.append(
        {
            "season": SEASON,
            "game_id": "prediction-game",
            "game_date": pd.Timestamp("2025-11-01"),
            "player_id": 1,
            "player_name": "Target",
            "team_id": 10,
            "team_abbr": "AAA",
            "opp_team_id": 20,
            "opp_abbr": "BBB",
            "home_away": "H",
            "minutes": 32,
            "fga": 15,
            "reb": 7,
            "reb_chances": 14,
            "touches": 55,
            "usage_rate": 0.24,
            "listed_position": "G",
            "starter_flag": 1,
            "tracking_available": True,
            "advanced_available": True,
        }
    )
    frame = pd.DataFrame(rows).reindex(
        columns=historical_store.HISTORICAL_PLAYER_GAME_COLUMNS
    )
    historical_store.validate_player_game_store(frame, expected_season=SEASON)
    historical_store.write_season_store(
        frame,
        {"schema_version": 1, "season": SEASON, "rows": len(frame)},
        season=SEASON,
        store_dir=tmp_path,
    )


def test_full_single_game_prediction_runs_without_api(monkeypatch, tmp_path):
    _write_prediction_store(tmp_path)
    monkeypatch.setattr(prediction, "resolve_player", lambda query: (1, "Target"))

    def fail_network(*args, **kwargs):
        pytest.fail("NBA API should not be called when the local season exists")

    monkeypatch.setattr(dataset, "fetch_players_gamelogs", fail_network)
    monkeypatch.setattr(live_data, "fetch_player_gamelog", fail_network)
    monkeypatch.setattr(live_data, "fetch_league_player_stats", fail_network)
    monkeypatch.setattr(live_data, "fetch_league_tracking_stats", fail_network)

    result = prediction.predict_player_game(
        "Target",
        "2025-11-01",
        season=SEASON,
        candidate_player_ids=[2],
        top_n_similar=1,
        store_dir=tmp_path,
    )

    projection = result["projection"]
    assert projection["game_id"] == "prediction-game"
    assert projection["baseline_games_used"] == 5
    assert projection["similar_player_ids"] == [2]
    assert projection["fga_n_samples"] == 5
    assert projection["reb_chances_n_samples"] == 5
    assert projection["pred_fga"] > 0
    assert projection["pred_reb_chances"] > 0

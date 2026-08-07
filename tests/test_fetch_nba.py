import pandas as pd
import pytest

import src.fetch_nba as fetch_nba


def test_game_fetches_resume_from_cache_without_sleeping(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_nba, "CACHE_DIR", tmp_path)
    sleeps = []
    monkeypatch.setattr(fetch_nba.time, "sleep", lambda seconds: sleeps.append(seconds))
    calls = []

    def getter(game_id):
        calls.append(game_id)
        return pd.DataFrame({"gameId": [game_id], "value": [1]})

    first = fetch_nba._fetch_games(
        ["g1", "g2"],
        namespace="test_tracking",
        getter=getter,
        refresh=False,
    )
    second = fetch_nba._fetch_games(
        ["g1", "g2"],
        namespace="test_tracking",
        getter=getter,
        refresh=False,
    )

    assert len(first) == 2
    assert len(second) == 2
    assert calls == ["g1", "g2"]
    assert sleeps == [fetch_nba.REQUEST_SLEEP_SECONDS]


def test_empty_game_response_is_not_kept_in_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_nba, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetch_nba.time, "sleep", lambda seconds: None)
    calls = []

    def getter(game_id):
        calls.append(game_id)
        return pd.DataFrame()

    for _ in range(2):
        result = fetch_nba._fetch_games(
            ["g1"],
            namespace="empty_tracking",
            getter=getter,
            refresh=False,
        )
        assert result.empty

    assert calls == ["g1", "g1"]
    assert not list(tmp_path.glob("empty_tracking_*.parquet"))


def test_empty_season_response_is_not_kept_in_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_nba, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        fetch_nba,
        "get_season_player_game_logs",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    with pytest.raises(fetch_nba.NBADataFetchError, match="returned no Base"):
        fetch_nba.fetch_season_player_game_logs("2025-26")

    assert not list(tmp_path.glob("season_player_game_logs_*.parquet"))

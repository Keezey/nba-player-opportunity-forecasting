import pandas as pd

import src.live_evaluate as live_evaluate
import src.live_data as live_data
import src.predict as predict


def _base_dashboard(player_ids=(1, 2, 3), starter=False):
    return pd.DataFrame(
        {
            "PLAYER_ID": list(player_ids),
            "PLAYER_NAME": [f"Player {player_id}" for player_id in player_ids],
            "GP": [4 if starter else 5 for _ in player_ids],
            "MIN": [150, 155, 120][: len(player_ids)],
            "FGA": [50, 55, 30][: len(player_ids)],
            "REB": [30, 32, 20][: len(player_ids)],
            "TEAM_COUNT": [1] * len(player_ids),
        }
    )


def _tracking_dashboard(measure_type, player_ids=(1, 2, 3)):
    data = {
        "PLAYER_ID": list(player_ids),
        "GP": [5] * len(player_ids),
        "MIN": [150, 155, 120][: len(player_ids)],
    }
    if measure_type == "Rebounding":
        data["REB_CHANCES"] = [60, 65, 30][: len(player_ids)]
    else:
        data["TOUCHES"] = [250, 260, 120][: len(player_ids)]
    return pd.DataFrame(data)


def test_season_from_game_date_supports_tracking_era_through_2025_26():
    assert live_data.season_from_game_date("2025-11-01") == "2025-26"
    assert live_data.season_from_game_date("2026-04-07") == "2025-26"


def test_build_dashboard_profiles_populates_all_requested_factors(monkeypatch):
    def fake_player_stats(*, measure_type, starter_bench="", player_position="", **kwargs):
        if player_position:
            members = {"G": (1, 2), "F": (1,), "C": (3,)}[player_position]
            return _base_dashboard(members)
        if measure_type == "Advanced":
            return pd.DataFrame(
                {
                    "PLAYER_ID": [1, 2, 3],
                    "USG_PCT": [0.25, 0.26, 0.15],
                    "MIN": [30, 31, 24],
                }
            )
        return _base_dashboard(starter=starter_bench == "Starters")

    monkeypatch.setattr(live_data, "fetch_league_player_stats", fake_player_stats)
    monkeypatch.setattr(
        live_data,
        "fetch_league_tracking_stats",
        lambda *, measure_type, **kwargs: _tracking_dashboard(measure_type),
    )

    profiles = live_data.build_dashboard_profiles(
        season="2025-26",
        as_of_date="2026-04-06",
        min_games=5,
        refresh=True,
    )
    assert profiles.loc[1, "listed_position"] == "G-F"
    assert profiles.loc[1, "starter_rate"] == 0.8
    assert profiles.loc[1, "usage_rate"] == 0.25
    assert profiles.loc[1, "touches_per_min"] == 250 / 150


def test_build_opponent_rates_normalizes_totals(monkeypatch):
    monkeypatch.setattr(
        live_data,
        "fetch_league_player_stats",
        lambda **kwargs: _base_dashboard((2, 3)),
    )
    monkeypatch.setattr(
        live_data,
        "fetch_league_tracking_stats",
        lambda **kwargs: _tracking_dashboard("Rebounding", (2, 3)),
    )
    rates = live_data.build_opponent_rates(
        season="2025-26",
        as_of_date="2026-04-06",
        opponent_team_id=100,
        refresh=True,
    )
    assert rates.loc[2, "fga_games"] == 5
    assert rates.loc[2, "reb_chances_per_min"] == 60 / 150


def test_predict_player_game_looks_up_every_input(monkeypatch):
    target_history = pd.DataFrame(
        {
            "player_id": [1] * 5,
            "game_id": [f"1-{day}" for day in range(5)],
            "game_date": pd.date_range("2026-03-25", periods=5),
            "opp_abbr": ["BOS"] * 5,
            "minutes": [30] * 5,
            "fga": [10] * 5,
            "reb": [6] * 5,
            "reb_chances": [12] * 5,
            "touches": [50] * 5,
            "usage_rate": [0.25] * 5,
            "listed_position": ["G"] * 5,
            "starter_flag": [1] * 5,
        }
    )
    profiles = pd.DataFrame(
        {
            "player_name": ["Target", "Similar", "Different"],
            "games_used": [5, 5, 5],
            "minutes": [30, 31, 20],
            "fga_per_min": [0.33, 0.35, 0.15],
            "reb_chances_per_min": [0.4, 0.42, 0.15],
            "usage_rate": [0.25, 0.26, 0.12],
            "touches_per_min": [1.67, 1.7, 0.8],
            "listed_position": ["G", "G-F", "C"],
            "starter_rate": [1.0, 1.0, 0.0],
            "starter_flag": [1, 1, 0],
        },
        index=[1, 2, 3],
    )
    opponent_rates = pd.DataFrame(
        {
            "fga_per_min": [0.40, 0.20],
            "fga_games": [2, 1],
            "reb_chances_per_min": [0.50, 0.20],
            "reb_chances_games": [2, 1],
        },
        index=[2, 3],
    )

    monkeypatch.setattr(predict, "resolve_player", lambda query: (1, "Target"))
    monkeypatch.setattr(
        predict,
        "lookup_player_game",
        lambda *args, **kwargs: pd.Series(
            {
                "game_id": "target-game",
                "opp_abbr": "LAL",
                "opp_team_id": 100,
                "season": "2025-26",
            }
        ),
    )
    monkeypatch.setattr(predict, "build_player_game_df", lambda *args, **kwargs: target_history)
    monkeypatch.setattr(predict, "build_dashboard_profiles", lambda **kwargs: profiles)
    monkeypatch.setattr(predict, "build_opponent_rates", lambda **kwargs: opponent_rates)

    result = predict.predict_player_game("Target", "2026-04-07", top_n_similar=1)
    assert result["projection"]["game_id"] == "target-game"
    assert result["projection"]["similar_player_ids"] == [2]
    assert result["projection"]["pred_fga"] > 10


def test_live_period_backtest_uses_single_game_workflow(monkeypatch):
    target_games = pd.DataFrame(
        {
            "player_id": [1, 1],
            "game_id": ["g1", "g2"],
            "game_date": pd.to_datetime(["2026-03-01", "2026-03-03"]),
            "opp_abbr": ["BOS", "LAL"],
            "minutes": [30, 31],
            "fga": [10, 12],
            "reb": [6, 7],
        }
    )
    actuals = target_games.assign(
        reb_chances=[12, 14],
        touches=[50, 52],
        usage_rate=[pd.NA, pd.NA],
        listed_position=["G", "G"],
        starter_flag=[1, 1],
    )
    calls = []

    def fake_predict_player_game(player_query, game_date, **kwargs):
        calls.append((player_query, pd.to_datetime(game_date).date()))
        projection = pd.Series(
            {
                "game_id": f"g{len(calls)}",
                "game_date": pd.to_datetime(game_date),
                "season": "2025-26",
                "target_opp_abbr": "BOS",
                "pred_fga": 11,
                "pred_reb": 6.5,
                "pred_reb_chances": 13,
                "baseline_games_used": 5,
                "tracking_games_used": 5,
                "n_similar_players": 1,
                "n_matchup_samples": 8,
                "trust_weight": 0.6,
                "trust_label": "medium",
                "fga_n_samples": 8,
                "reb_chances_n_samples": 8,
                "similar_player_ids": [2],
            }
        )
        similar = pd.DataFrame({"player_name": ["Similar"]}, index=[2])
        return {"projection": projection, "similar_players": similar}

    monkeypatch.setattr(live_evaluate, "resolve_player", lambda query: (1, "Target"))
    monkeypatch.setattr(live_evaluate, "_target_games_in_range", lambda *args, **kwargs: target_games)
    monkeypatch.setattr(live_evaluate, "_target_actuals_in_range", lambda *args, **kwargs: actuals.set_index("game_id"))
    monkeypatch.setattr(live_evaluate, "predict_player_game", fake_predict_player_game)

    backtest = live_evaluate.run_live_period_backtest(
        "Target",
        "2026-03-01",
        "2026-03-05",
        top_n_similar=1,
    )

    assert calls == [(1, pd.Timestamp("2026-03-01").date()), (1, pd.Timestamp("2026-03-03").date())]
    assert len(backtest) == 2
    assert backtest["status"].tolist() == ["predicted", "predicted"]
    assert backtest.loc[0, "error_fga"] == 1
    assert backtest.loc[0, "similar_player_names"] == ["Similar"]

import numpy as np
import pandas as pd
import pytest

from src.baselines import compute_player_baseline
from src.dataset import _parse_minutes, clean_gamelog, normalize_advanced, normalize_tracking
from src.evaluate import backtest_predictions, summarize_backtest
from src.matchups import (
    apply_matchup_to_target_baseline,
    compute_matchup_multipliers,
    compute_profile_matchup_multipliers,
    project_target_matchup,
)
from src.predict import predict_target
from src.similarity import build_player_profiles, find_similar_players
from src.trust import (
    DEFAULT_MULTIPLIER_BOUNDS,
    MultiplierBounds,
    effective_multiplier,
    sample_trust_weight,
)


def _game_rows(days=5):
    rows = []
    for player_id, fga, chances, touches, usage, position in [
        (1, 10, 12, 50, 0.25, "G-F"),
        (2, 11, 13, 52, 0.26, "G"),
        (3, 4, 5, 20, 0.12, "C"),
    ]:
        for day in range(days):
            opponent = "LAL" if player_id == 2 and day >= max(days - 2, 0) else "BOS"
            rows.append(
                {
                    "player_id": player_id,
                    "game_id": f"{player_id}-{day}",
                    "game_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "opp_abbr": opponent,
                    "minutes": 30,
                    "fga": fga + (2 if opponent == "LAL" else 0),
                    "reb": 6,
                    "reb_chances": chances + (3 if opponent == "LAL" else 0),
                    "touches": touches,
                    "usage_rate": usage,
                    "listed_position": position,
                    "starter_flag": 1,
                }
            )
    return pd.DataFrame(rows)


def test_parse_minutes():
    assert _parse_minutes("32:30") == 32.5
    assert _parse_minutes(28) == 28.0
    assert _parse_minutes(None) == 0.0


def test_dataset_normalizers_keep_only_current_fields():
    raw_log = pd.DataFrame(
        {
            "Game_ID": ["001"],
            "GAME_DATE": ["Jan 1, 2026"],
            "Player_ID": [1],
            "PLAYER_ID": [1],
            "MIN": [30],
            "FGA": [10],
            "REB": [7],
            "MATCHUP": ["AAA @ BBB"],
        }
    )
    clean = clean_gamelog(raw_log)
    assert clean.columns.tolist().count("player_id") == 1
    assert clean.loc[0, "opp_abbr"] == "BBB"

    tracking = normalize_tracking(
        pd.DataFrame(
            {
                "gameId": ["001"],
                "GAME_ID": ["001"],
                "personId": [1],
                "reboundChancesTotal": [12],
                "touches": [50],
                "position": ["G"],
            }
        )
    )
    assert tracking.loc[0, "reb_chances"] == 12
    assert tracking.loc[0, "tracking_position"] == "G"

    advanced = normalize_advanced(
        pd.DataFrame(
            {
                "gameId": ["001"],
                "GAME_ID": ["001"],
                "personId": [1],
                "usagePercentage": [0.27],
                "position": ["G"],
            }
        )
    )
    assert advanced.loc[0, "usage_rate"] == 0.27


def test_compute_player_baseline_uses_paired_tracking_conversion():
    games = pd.DataFrame(
        {
            "player_id": [1] * 5,
            "game_date": pd.to_datetime(
                ["2026-01-01", "2026-01-03", "2026-01-05", "2026-01-07", "2026-01-09"]
            ),
            "minutes": [30, 31, 32, 33, 34],
            "fga": [10, 12, 11, 13, 14],
            "reb": [7, 8, 9, 8, 10],
            "reb_chances": [12, 13, 14, 13, 15],
        }
    )
    baseline = compute_player_baseline(games, as_of_date="2026-01-09", min_games=5)
    assert baseline["games_used"] == 5
    assert baseline["tracking_games_used"] == 5
    assert baseline["baseline_reb"] == 8.4
    assert baseline["reb_conversion"] == 42 / 67


def test_compute_player_baseline_rejects_too_few_tracking_games():
    games = _game_rows().query("player_id == 1").copy()
    games.loc[games.index[:2], "reb_chances"] = np.nan
    assert compute_player_baseline(games, min_games=5).empty


def test_similarity_uses_requested_features_and_overlapping_positions():
    games = _game_rows()
    profiles = build_player_profiles(games, as_of_date="2026-01-05", min_games=5)
    similar = find_similar_players(profiles, target_player_id=1, top_n=1)
    assert similar.index.tolist() == [2]
    assert similar.loc[2, "position_penalty"] == 0


def test_similarity_drops_an_entirely_missing_feature():
    games = _game_rows()
    games["usage_rate"] = np.nan
    profiles = build_player_profiles(games, as_of_date="2026-01-05", min_games=5)
    similar = find_similar_players(profiles, target_player_id=1, top_n=1)
    assert similar.index.tolist() == [2]


def test_sample_trust_curve_matches_project_thresholds():
    assert sample_trust_weight(0) == 0.0
    assert sample_trust_weight(2) < sample_trust_weight(5)
    assert sample_trust_weight(5) < sample_trust_weight(10)
    assert sample_trust_weight(10) >= 0.80
    assert sample_trust_weight(15) == 0.95
    assert effective_multiplier(1.20, 2) < effective_multiplier(1.20, 10)


def test_extreme_multiplier_is_bounded_before_shrinkage():
    bounded = effective_multiplier(10.0, 15)
    unbounded = effective_multiplier(
        10.0,
        15,
        lower_bound=None,
        upper_bound=None,
    )

    assert bounded == pytest.approx(1 + 0.95 * (1.5 - 1))
    assert unbounded == pytest.approx(1 + 0.95 * (10 - 1))
    assert DEFAULT_MULTIPLIER_BOUNDS == MultiplierBounds.fixed(0.5, 1.5)


def test_matchup_multipliers_use_metric_specific_samples():
    games = _game_rows()
    multipliers = compute_matchup_multipliers(
        games,
        similar_player_ids=[2],
        target_opp_abbr="LAL",
        as_of_date="2026-01-05",
        min_games=5,
    )
    assert multipliers["fga_n_samples"] == 2
    assert multipliers["reb_chances_n_samples"] == 2
    assert multipliers["reb_chances_multiplier"] > 1.0
    assert multipliers["reb_chances_multiplier"] < multipliers["raw_reb_chances_multiplier"]


def test_dashboard_matchup_multipliers_weight_by_games():
    profiles = pd.DataFrame(
        {
            "fga_per_min": [0.4, 0.5],
            "reb_chances_per_min": [0.4, 0.5],
        },
        index=[2, 3],
    )
    opponent = pd.DataFrame(
        {
            "fga_per_min": [0.48, 0.45],
            "fga_games": [2, 1],
            "reb_chances_per_min": [0.48, 0.45],
            "reb_chances_games": [2, 1],
        },
        index=[2, 3],
    )
    multipliers = compute_profile_matchup_multipliers(profiles, opponent, [2, 3], "LAL")
    assert multipliers["n_samples"] == 3
    assert multipliers["raw_fga_multiplier"] == (1.2 * 2 + 0.9) / 3


def test_dashboard_matchup_multipliers_preserve_and_bound_extreme_ratio():
    profiles = pd.DataFrame(
        {"fga_per_min": [0.1], "reb_chances_per_min": [0.1]},
        index=[2],
    )
    opponent = pd.DataFrame(
        {
            "fga_per_min": [0.4],
            "fga_games": [15],
            "reb_chances_per_min": [0.4],
            "reb_chances_games": [15],
        },
        index=[2],
    )

    multipliers = compute_profile_matchup_multipliers(
        profiles,
        opponent,
        [2],
        "LAL",
    )

    assert multipliers["raw_fga_multiplier"] == 4.0
    assert multipliers["bounded_fga_multiplier"] == 1.5
    assert bool(multipliers["fga_multiplier_was_bounded"])
    assert multipliers["fga_multiplier"] == pytest.approx(1 + 0.95 * 0.5)


def test_apply_matchup_uses_rebound_chances_and_conversion():
    baseline = pd.Series(
        {
            "player_id": 1,
            "as_of_date": pd.Timestamp("2026-01-05"),
            "games_used": 5,
            "tracking_games_used": 5,
            "baseline_minutes": 30,
            "baseline_fga": 10,
            "baseline_reb": 6,
            "baseline_reb_chances": 12,
            "reb_conversion": 0.5,
        }
    )
    multipliers = pd.Series(
        {
            "target_opp_abbr": "LAL",
            "fga_multiplier": 1.1,
            "reb_chances_multiplier": 1.2,
            "n_samples": 10,
            "trust_weight": 0.8,
            "trust_label": "high",
        }
    )
    prediction = apply_matchup_to_target_baseline(baseline, multipliers)
    assert prediction["pred_fga"] == 11
    assert prediction["pred_reb"] == 7.199999999999999


def test_saved_dataset_prediction_and_backtest_still_run():
    games = _game_rows(days=8)
    games.loc[(games["player_id"] == 1) & (games["game_date"] == "2026-01-08"), "opp_abbr"] = "LAL"

    result = predict_target(
        games,
        target_player_id=1,
        target_opp_abbr="LAL",
        as_of_date="2026-01-07",
        top_n_similar=1,
        min_games=5,
    )
    assert result["projection"]["player_id"] == 1

    backtest = backtest_predictions(
        games,
        target_player_ids=[1],
        candidate_player_ids=[1, 2, 3],
        start_date="2026-01-08",
        end_date="2026-01-08",
        top_n_similar=1,
        min_games=5,
    )
    summary = summarize_backtest(backtest)
    assert len(backtest) == 1
    assert summary["n_predictions"] == 1


def test_project_target_matchup_returns_projection():
    games = _game_rows()
    projection = project_target_matchup(
        games,
        target_player_id=1,
        similar_player_ids=[2],
        target_opp_abbr="LAL",
        as_of_date="2026-01-05",
        min_games=5,
    )
    assert projection["player_id"] == 1
    assert projection["n_matchup_samples"] == 2

import pandas as pd

import src.v2.predict as v2_predict
from src.v2.evaluate import chronological_split
from src.v2.models import fit_residual_models


def _fake_v1_result(row):
    baseline = pd.Series(
        {
            "player_id": int(row["player_id"]),
            "as_of_date": row["as_of_date"],
            "games_used": row["baseline_games_used"],
            "tracking_games_used": row["tracking_games_used"],
            "usual_minutes": row["usual_minutes"],
            "min_minutes_threshold": row["min_minutes_threshold"],
            "baseline_minutes": row["baseline_minutes"],
            "baseline_fga": row["baseline_fga"],
            "baseline_reb": row["baseline_reb"],
            "baseline_reb_chances": row["baseline_reb_chances"],
            "fga_per_min": row["baseline_fga_per_min"],
            "reb_chances_per_min": row["baseline_reb_chances_per_min"],
            "reb_conversion": row["baseline_reb_conversion"],
        }
    )
    profiles = pd.DataFrame(
        {
            "games_used": [row["profile_games_used"]],
            "minutes": [row["profile_minutes"]],
            "fga_per_min": [row["profile_fga_per_min"]],
            "reb_chances_per_min": [row["profile_reb_chances_per_min"]],
            "usage_rate": [row["profile_usage_rate"]],
            "touches_per_min": [row["profile_touches_per_min"]],
            "starter_rate": [row["profile_starter_rate"]],
            "starter_flag": [row["profile_starter_flag"]],
            "listed_position": [row["profile_listed_position"]],
        },
        index=[int(row["player_id"])],
    )
    projection = pd.Series(
        {
            "player_id": int(row["player_id"]),
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "season": row["season"],
            "target_opp_abbr": row["opp_abbr"],
            "minutes_pred": row["v1_pred_minutes"],
            "pred_fga": row["v1_pred_fga"],
            "pred_reb_chances": row["v1_pred_reb_chances"],
            "pred_reb": row["v1_pred_reb"],
            "similar_player_ids": [],
            "n_similar_players": 0,
            "fga_n_samples": row["fga_n_samples"],
            "fga_trust_weight": row["fga_trust_weight"],
            "reb_chances_n_samples": row["reb_chances_n_samples"],
            "reb_chances_trust_weight": row["reb_chances_trust_weight"],
        }
    )
    return {
        "target_player_id": int(row["player_id"]),
        "target_player_name": "Target",
        "game": pd.Series(
            {
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "season": row["season"],
                "opp_abbr": row["opp_abbr"],
            }
        ),
        "projection": projection,
        "target_baseline": baseline,
        "profiles": profiles,
        "opponent_rates": pd.DataFrame(),
        "similar_players": pd.DataFrame(),
    }


def test_v2_single_game_prediction_wraps_v1(monkeypatch, v2_training_frame):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    bundle = fit_residual_models(train, model_type="elastic_net", min_rows=20)
    row = test.iloc[0]
    monkeypatch.setattr(
        v2_predict,
        "predict_player_game",
        lambda *args, **kwargs: _fake_v1_result(row),
    )

    result = v2_predict.predict_player_game_v2(
        bundle,
        "Target",
        row["game_date"],
    )

    assert result["target_player_name"] == "Target"
    assert "v2_pred_fga" in result["v2_projection"]
    assert "v2_pred_reb_chances" in result["v2_projection"]

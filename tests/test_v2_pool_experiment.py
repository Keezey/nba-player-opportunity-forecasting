import pandas as pd

from src.v2.pool_experiment import run_pool_size_experiment
from src.v2.tuning import TrustParameters


def test_pool_size_experiment_uses_one_test_set_and_reports_seen_unseen(
    v2_training_frame,
):
    development = v2_training_frame.iloc[:80].copy()
    test = v2_training_frame.iloc[80:].copy()
    development["game_date"] = pd.date_range(
        "2022-01-01", periods=len(development)
    )
    development["as_of_date"] = development["game_date"] - pd.Timedelta(days=1)
    development["game_id"] = [f"dev-{index}" for index in range(len(development))]
    test["game_date"] = pd.date_range("2023-01-01", periods=len(test))
    test["as_of_date"] = test["game_date"] - pd.Timedelta(days=1)
    test["game_id"] = [f"test-{index}" for index in range(len(test))]

    all_ids = sorted(development["player_id"].unique())
    pools = {
        4: all_ids[:4],
        8: all_ids[:8],
        12: all_ids,
    }
    model_parameters = {
        "elastic_net": {
            "fga": {"alpha": 0.01, "l1_ratio": 0.1},
            "reb_chances": {"alpha": 0.005, "l1_ratio": 0.5},
        }
    }

    result = run_pool_size_experiment(
        development,
        test,
        pools=pools,
        model_parameters=model_parameters,
        trust_parameters=TrustParameters(),
        min_rows=20,
    )

    all_game_weighted = result.summary[
        (result.summary["scope"] == "all")
        & (result.summary["aggregation"] == "game_weighted")
    ]
    assert len(all_game_weighted) == 3 * 3
    assert all_game_weighted["test_rows"].nunique() == 1
    assert all_game_weighted["test_rows"].iloc[0] == len(test)
    assert set(result.summary["aggregation"]) == {
        "game_weighted",
        "player_weighted",
    }
    assert {"seen", "unseen"}.issubset(result.summary["scope"])

    largest_predictions = result.predictions[result.predictions["pool_size"] == 12]
    assert largest_predictions["target_seen_in_training"].all()
    assert not result.predictions[result.predictions["pool_size"] == 4][
        "target_seen_in_training"
    ].all()

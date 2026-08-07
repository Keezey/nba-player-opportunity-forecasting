import json
import sys

import pandas as pd

import scripts.build_v2_training_dataset as build_training_script
from scripts.run_v2_backtest import external_holdout_split, main as backtest_main
from scripts.train_v2_models import main as train_main
from scripts.tune_v2_parameters import main as tune_main
from src.v2.training_data import TrainingDatasetResult


def test_v2_offline_cli_pipeline(monkeypatch, tmp_path, v2_training_frame):
    dataset = tmp_path / "training.parquet"
    parameters = tmp_path / "parameters.json"
    report = tmp_path / "backtest.md"
    model = tmp_path / "model.joblib"
    v2_training_frame.to_parquet(dataset, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_v2_parameters",
            str(dataset),
            str(parameters),
            "--model-type",
            "elastic_net",
            "--n-iter",
            "2",
            "--n-splits",
            "2",
            "--min-rows",
            "20",
        ],
    )
    assert tune_main() == 0
    assert parameters.exists()
    with parameters.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["multiplier_bounds"] == {
        "fga_lower": 0.5,
        "fga_upper": 1.5,
        "reb_chances_lower": 0.5,
        "reb_chances_upper": 1.5,
    }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v2_backtest",
            str(dataset),
            str(report),
            "--parameters-json",
            str(parameters),
            "--apply-tuned-trust",
            "--min-rows",
            "20",
            "--skip-importance",
            "--skip-ablation",
        ],
    )
    assert backtest_main() == 0
    assert report.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_v2_models",
            str(dataset),
            str(model),
            "--parameters-json",
            str(parameters),
            "--apply-tuned-trust",
            "--min-rows",
            "20",
        ],
    )
    assert train_main() == 0
    assert model.exists()


def test_training_dataset_cli_accepts_saved_player_pool(monkeypatch, tmp_path):
    pool_path = tmp_path / "pool.csv"
    output_path = tmp_path / "training.parquet"
    pd.DataFrame({"player_id": [101, 202, 101]}).to_csv(pool_path, index=False)
    captured = {}

    def fake_build(player_queries, start_date, end_date, **kwargs):
        captured["player_queries"] = list(player_queries)
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return TrainingDatasetResult(pd.DataFrame(), pd.DataFrame())

    def fake_write(result, output, *, skipped_path=None):
        return output, skipped_path

    monkeypatch.setattr(build_training_script, "build_training_dataset", fake_build)
    monkeypatch.setattr(build_training_script, "write_training_dataset", fake_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_v2_training_dataset",
            str(output_path),
            "2022-10-01",
            "2023-04-01",
            "--player-pool",
            str(pool_path),
        ],
    )

    assert build_training_script.main() == 0
    assert captured["player_queries"] == [101, 202]
    assert captured["start_date"] == "2022-10-01"
    assert captured["end_date"] == "2023-04-01"


def test_external_holdout_requires_strictly_later_dates(v2_training_frame):
    training = v2_training_frame.iloc[:80].copy()
    holdout = v2_training_frame.iloc[80:].copy()
    training["game_date"] = pd.date_range("2022-01-01", periods=len(training))
    holdout["game_date"] = pd.date_range("2023-01-01", periods=len(holdout))

    returned_training, returned_holdout = external_holdout_split(training, holdout)
    assert len(returned_training) == len(training)
    assert len(returned_holdout) == len(holdout)

    holdout.loc[holdout.index[0], "game_date"] = training["game_date"].max()
    try:
        external_holdout_split(training, holdout)
    except ValueError as exc:
        assert "must begin after" in str(exc)
    else:
        raise AssertionError("Expected overlapping dates to be rejected.")

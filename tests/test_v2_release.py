import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.finalize_v2_release import main as finalize_main
from scripts.prepare_v2_production_parameters import extract_production_parameters
from src.v2.cohorts import select_training_cohort
from src.v2.models import load_model_bundle
from src.v2.release import (
    combine_release_datasets,
    merge_training_datasets,
    validate_training_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "artifact_name",
    ["v2_3_250_all_games", "v2_3_250_normal_minutes"],
)
def test_packaged_v2_artifacts_match_public_manifests(artifact_name):
    artifact_dir = PROJECT_ROOT / "artifacts" / "v2.3.0"
    model_path = artifact_dir / f"{artifact_name}.joblib"
    manifest_path = artifact_dir / f"{artifact_name}_manifest.json"

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    bundle = load_model_bundle(model_path)

    assert "/Users/" not in manifest_text
    assert model_hash == manifest["artifacts"]["model"]["sha256"]
    assert bundle.model_type == "elastic_net"
    assert bundle.release_metadata["release_version"] == "2.3.0"


def _chronological_parts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = frame.iloc[:80].copy()
    holdout = frame.iloc[80:].copy()
    development["game_date"] = pd.date_range("2022-01-01", periods=len(development))
    development["as_of_date"] = development["game_date"] - pd.Timedelta(days=1)
    holdout["game_date"] = pd.date_range("2023-01-01", periods=len(holdout))
    holdout["as_of_date"] = holdout["game_date"] - pd.Timedelta(days=1)
    development["game_id"] = [f"development-{i}" for i in range(len(development))]
    holdout["game_id"] = [f"00{i:08d}" for i in range(len(holdout))]
    return development, holdout


def test_combine_release_datasets_enforces_chronology(v2_training_frame):
    development, holdout = _chronological_parts(v2_training_frame)
    combined = combine_release_datasets(development, holdout)

    assert len(combined) == len(v2_training_frame)
    assert combined["game_date"].is_monotonic_increasing
    assert combined.duplicated(["game_id", "player_id"]).sum() == 0

    holdout["game_date"] = pd.Timestamp("2022-01-15")
    holdout["as_of_date"] = holdout["game_date"] - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="strictly after"):
        combine_release_datasets(development, holdout)


def test_release_validation_rejects_duplicate_keys_and_leakage(v2_training_frame):
    development, _ = _chronological_parts(v2_training_frame)
    duplicated = pd.concat([development, development.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_training_frame(duplicated, label="duplicate test")

    leaking = development.copy()
    leaking.loc[leaking.index[0], "as_of_date"] = leaking.loc[
        leaking.index[0], "game_date"
    ]
    with pytest.raises(ValueError, match="feature cutoff"):
        validate_training_frame(leaking, label="leakage test")


def test_merge_training_datasets_accepts_parallel_player_cohorts(
    v2_training_frame,
):
    left = v2_training_frame[v2_training_frame["player_id"] < 106].copy()
    right = v2_training_frame[v2_training_frame["player_id"] >= 106].copy()

    combined = merge_training_datasets({"left": left, "right": right})

    assert len(combined) == len(v2_training_frame)
    assert combined["player_id"].nunique() == v2_training_frame["player_id"].nunique()

    with pytest.raises(ValueError, match="duplicate"):
        merge_training_datasets({"first": left, "overlap": left.iloc[:2].copy()})


def test_finalize_release_cli_writes_audited_artifacts(
    monkeypatch, tmp_path, v2_training_frame
):
    development, holdout = _chronological_parts(v2_training_frame)
    development_path = tmp_path / "development.parquet"
    holdout_path = tmp_path / "holdout.parquet"
    output_dataset = tmp_path / "all.parquet"
    output_model = tmp_path / "production.joblib"
    manifest_path = tmp_path / "manifest.json"
    parameters_path = tmp_path / "parameters.json"
    evaluation_path = tmp_path / "evaluation.csv"
    development.to_parquet(development_path, index=False)
    holdout.to_parquet(holdout_path, index=False)
    evaluation = holdout[
        [
            "game_id",
            "game_date",
            "player_id",
            "actual_fga",
            "actual_reb_chances",
            "actual_reb",
            "v1_pred_fga",
            "v1_pred_reb_chances",
            "v1_pred_reb",
        ]
    ].copy()
    for stat in ["fga", "reb_chances", "reb"]:
        evaluation[f"v2_pred_{stat}"] = evaluation[f"v1_pred_{stat}"]
    evaluation.to_csv(evaluation_path, index=False)
    parameters_path.write_text(
        json.dumps(
            {
                "model_type": "elastic_net",
                "training_cohort": "all",
                "model_parameters": {
                    "fga": {"alpha": 0.01, "l1_ratio": 0.1},
                    "reb_chances": {"alpha": 0.005, "l1_ratio": 0.5},
                },
                "trust_parameters": {
                    "low_sample_threshold": 5,
                    "high_sample_threshold": 10,
                    "max_sample_threshold": 15,
                    "low_trust_weight": 0.15,
                    "high_trust_weight": 0.65,
                    "max_trust_weight": 0.85,
                },
                "conversion_parameters": {
                    "alpha": 0.003,
                    "l1_ratio": 0.25,
                },
                "regular_minutes": {
                    "absolute_tolerance": 3.0,
                    "relative_tolerance": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_v2_release",
            str(development_path),
            str(holdout_path),
            str(output_dataset),
            str(output_model),
            "--parameters-json",
            str(parameters_path),
            "--manifest",
            str(manifest_path),
            "--evaluation-predictions",
            str(evaluation_path),
            "--min-rows",
            "20",
        ],
    )
    assert finalize_main() == 0

    assert len(pd.read_parquet(output_dataset)) == len(v2_training_frame)
    bundle = load_model_bundle(output_model)
    assert bundle.model_type == "elastic_net"
    assert bundle.conversion_model is not None
    assert bundle.release_metadata["release_version"] == "2.3.0"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["training_dataset"]["rows"] == len(v2_training_frame)
    assert manifest["artifacts"]["model"]["sha256"]
    assert manifest["artifacts"]["model"]["has_conversion_model"] is True
    assert manifest["training_cohort"] == "all"
    assert manifest["feature_schema"]["conversion_features"]
    assert manifest["final_evaluation"]["rows"] == len(holdout)


def test_regular_training_cohort_filters_out_abnormal_minutes(v2_training_frame):
    frame = v2_training_frame.copy()
    frame.loc[frame.index[:10], "actual_minutes"] += 20

    regular = select_training_cohort(frame, cohort="regular")

    assert len(regular) == len(frame) - 10
    assert regular["regular_minutes_flag"].all()


def test_production_parameter_extraction_combines_experiment_outputs():
    target_scores = {
        "fga": {
            "parameters": {"alpha": 0.005, "l1_ratio": 1.0},
            "v1_mae": 3.1,
            "v2_mae": 3.0,
        },
        "reb_chances": {
            "parameters": {"alpha": 0.005, "l1_ratio": 0.25},
            "v1_mae": 3.2,
            "v2_mae": 3.1,
        },
    }
    opportunity = {
        "2025-26": {
            "training_seasons": ["2021-22", "2022-23", "2023-24", "2024-25"],
            "training_cohorts": {
                "all": {
                    "fixed_0.50_1.50": {
                        "training_rows": 100,
                        "multiplier_bounds": {
                            "fga_lower": 0.5,
                            "fga_upper": 1.5,
                            "reb_chances_lower": 0.5,
                            "reb_chances_upper": 1.5,
                        },
                        "trust": {
                            "parameters": {"low_sample_threshold": 5},
                            "objective": 0.99,
                        },
                        "models": {
                            "elastic_net": {"targets": target_scores}
                        },
                    }
                }
            },
        }
    }
    conversion = {
        "2025-26": {
            "training_seasons": ["2021-22", "2022-23", "2023-24", "2024-25"],
            "training_cohorts": {
                "all": {
                    "training_rows": 100,
                    "conversion_model_rows": 95,
                    "conversion_lower_bound": 0.4,
                    "conversion_upper_bound": 0.75,
                    "elastic_net": {"alpha": 0.003, "l1_ratio": 0.1},
                }
            },
        }
    }

    result = extract_production_parameters(
        opportunity,
        conversion,
        selection_season="2025-26",
        training_cohort="all",
        bound_strategy="fixed_0.50_1.50",
        model_type="elastic_net",
    )

    assert result["training_cohort"] == "all"
    assert result["model_parameters"]["fga"]["l1_ratio"] == 1.0
    assert result["conversion_parameters"] == {
        "alpha": 0.003,
        "l1_ratio": 0.1,
    }

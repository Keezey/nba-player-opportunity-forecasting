"""Build the audited all-history dataset and finalized V2 model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.v2.cohorts import select_training_cohort
from src.v2.conversion import CONVERSION_FEATURES
from src.v2.features import MODEL_FEATURE_COLUMNS
from src.v2.models import fit_model_bundle, save_model_bundle
from src.v2.release import (
    RELEASE_SCHEMA_VERSION,
    combine_release_datasets,
    evaluation_summary,
    file_sha256,
    frame_summary,
    runtime_summary,
    utc_now_iso,
    write_json_atomic,
)
from src.v2.tuning import TrustParameters, rebase_training_frame_with_trust


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate development and final-holdout rows, combine them, and train "
            "a production V2 artifact with frozen parameters."
        )
    )
    parser.add_argument("development_dataset")
    parser.add_argument("final_holdout_dataset")
    parser.add_argument("output_dataset")
    parser.add_argument("output_model")
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--release-version", default="2.3.0")
    parser.add_argument(
        "--training-cohort",
        choices=["all", "regular"],
        help=(
            "Rows used to fit the artifact. Defaults to training_cohort in the "
            "parameter JSON, then to all."
        ),
    )
    parser.add_argument("--regular-minutes-absolute-tolerance", type=float)
    parser.add_argument("--regular-minutes-relative-tolerance", type=float)
    parser.add_argument("--evaluation-report")
    parser.add_argument("--evaluation-predictions")
    parser.add_argument("--skipped-csv")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _absolute(path: str | Path) -> str:
    return str(Path(path).resolve())


def _source_record(path: str | Path, frame: pd.DataFrame) -> dict:
    return {
        "path": _absolute(path),
        "sha256": file_sha256(path),
        **frame_summary(frame),
    }


def _staging_path(path: str | Path) -> Path:
    output = Path(path)
    return output.with_name(f".{output.name}.staging")


def main() -> int:
    args = build_parser().parse_args()
    development_path = Path(args.development_dataset)
    holdout_path = Path(args.final_holdout_dataset)
    parameters_path = Path(args.parameters_json)

    development_raw = pd.read_parquet(development_path)
    holdout_raw = pd.read_parquet(holdout_path)
    full_combined = combine_release_datasets(development_raw, holdout_raw)
    development = full_combined[
        full_combined["game_date"]
        <= pd.to_datetime(development_raw["game_date"]).max()
    ].copy()
    holdout = full_combined[
        full_combined["game_date"]
        >= pd.to_datetime(holdout_raw["game_date"]).min()
    ].copy()

    with parameters_path.open(encoding="utf-8") as handle:
        parameters = json.load(handle)
    model_type = parameters.get("model_type")
    if not model_type:
        raise ValueError("The parameters JSON must contain model_type.")
    if "trust_parameters" not in parameters:
        raise ValueError("The parameters JSON must contain frozen trust_parameters.")
    if "model_parameters" not in parameters:
        raise ValueError("The parameters JSON must contain frozen model_parameters.")

    parameter_cohort = parameters.get("training_cohort")
    if (
        args.training_cohort is not None
        and parameter_cohort is not None
        and args.training_cohort != parameter_cohort
    ):
        raise ValueError(
            "--training-cohort does not match the frozen parameter JSON: "
            f"{args.training_cohort!r} != {parameter_cohort!r}."
        )
    training_cohort = args.training_cohort or parameter_cohort or "all"
    regular_config = parameters.get("regular_minutes", {})
    absolute_tolerance = (
        args.regular_minutes_absolute_tolerance
        if args.regular_minutes_absolute_tolerance is not None
        else float(regular_config.get("absolute_tolerance", 3.0))
    )
    relative_tolerance = (
        args.regular_minutes_relative_tolerance
        if args.regular_minutes_relative_tolerance is not None
        else float(regular_config.get("relative_tolerance", 0.15))
    )
    training_frame = select_training_cohort(
        full_combined,
        cohort=training_cohort,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    holdout_cohort = select_training_cohort(
        holdout,
        cohort=training_cohort,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    evaluation = None
    evaluation_path = None
    if args.evaluation_predictions:
        evaluation_path = Path(args.evaluation_predictions)
        evaluation = pd.read_csv(evaluation_path, dtype={"game_id": "string"})
        expected_keys = set(
            zip(
                holdout_cohort["game_id"].astype(str),
                holdout_cohort["player_id"].astype(int),
            )
        )
        evaluation_keys = set(
            zip(evaluation["game_id"].astype(str), evaluation["player_id"].astype(int))
        )
        if evaluation_keys != expected_keys:
            missing_keys = sorted(expected_keys - evaluation_keys)[:5]
            extra_keys = sorted(evaluation_keys - expected_keys)[:5]
            raise ValueError(
                "Evaluation predictions do not match the final-holdout player-game "
                f"keys. Missing examples: {missing_keys}; extra examples: {extra_keys}."
            )

    report_path = Path(args.evaluation_report) if args.evaluation_report else None
    if report_path is not None and not report_path.is_file():
        raise FileNotFoundError(f"Evaluation report does not exist: {report_path}")

    skipped_path = Path(args.skipped_csv) if args.skipped_csv else None
    skipped = None
    if skipped_path is not None:
        skipped = pd.read_csv(skipped_path)
        required_skip_columns = ["player_id", "error_type"]
        missing_skip_columns = [
            column for column in required_skip_columns if column not in skipped
        ]
        if missing_skip_columns:
            raise ValueError(
                f"Skipped-game CSV is missing columns: {missing_skip_columns}"
            )

    output_dataset = Path(args.output_dataset)
    model_path = Path(args.output_model)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else model_path.with_name(f"{model_path.stem}_manifest.json")
    )
    for output in [output_dataset, model_path, manifest_path]:
        output.parent.mkdir(parents=True, exist_ok=True)

    staged_dataset = _staging_path(output_dataset)
    staged_model = _staging_path(model_path)
    staged_manifest = _staging_path(manifest_path)
    staged_paths = [staged_dataset, staged_model, staged_manifest]
    for staged in staged_paths:
        staged.unlink(missing_ok=True)

    training_frame.to_parquet(staged_dataset, index=False)
    dataset_hash = file_sha256(staged_dataset)
    created_at = utc_now_iso()

    fitted_frame = rebase_training_frame_with_trust(
        training_frame,
        TrustParameters(**parameters["trust_parameters"]),
        parameters.get("multiplier_bounds"),
    )
    bundle = fit_model_bundle(
        fitted_frame,
        model_type=model_type,
        parameters_by_target=parameters["model_parameters"],
        conversion_parameters=parameters.get("conversion_parameters"),
        min_rows=args.min_rows,
        random_state=args.random_state,
    )
    bundle.trust_parameters = parameters["trust_parameters"]
    bundle.multiplier_bounds = parameters.get("multiplier_bounds")
    bundle.workflow_parameters = parameters.get("workflow_parameters")
    bundle.release_metadata = {
        "release_version": args.release_version,
        "created_at_utc": created_at,
        "training_dataset_sha256": dataset_hash,
        "parameters_sha256": file_sha256(parameters_path),
        "training_cohort": training_cohort,
        "training_rows": int(len(training_frame)),
        "full_history_rows": int(len(full_combined)),
        "train_start": training_frame["game_date"].min().date().isoformat(),
        "train_end": training_frame["game_date"].max().date().isoformat(),
        "opportunity_features": list(MODEL_FEATURE_COLUMNS),
        "conversion_features": list(CONVERSION_FEATURES),
    }

    save_model_bundle(bundle, staged_model)

    manifest: dict = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_version": args.release_version,
        "created_at_utc": created_at,
        "model_type": model_type,
        "training_cohort": training_cohort,
        "random_state": args.random_state,
        "parameters": parameters,
        "cohort_definition": {
            "name": training_cohort,
            "full_history_rows": int(len(full_combined)),
            "selected_rows": int(len(training_frame)),
            "excluded_rows": int(len(full_combined) - len(training_frame)),
            "regular_minutes": {
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
                "formula": (
                    "abs(actual_minutes - v1_pred_minutes) <= "
                    "max(absolute_tolerance, relative_tolerance * v1_pred_minutes)"
                ),
                "pregame_feature": False,
            },
        },
        "feature_schema": {
            "opportunity_features": list(MODEL_FEATURE_COLUMNS),
            "conversion_features": list(CONVERSION_FEATURES),
        },
        "runtime": runtime_summary(),
        "sources": {
            "development": _source_record(development_path, development),
            "final_holdout": _source_record(holdout_path, holdout),
            "parameters": {
                "path": _absolute(parameters_path),
                "sha256": file_sha256(parameters_path),
            },
        },
        "artifacts": {
            "training_dataset": {
                "path": _absolute(output_dataset),
                "sha256": dataset_hash,
                **frame_summary(training_frame),
            },
            "model": {
                "path": _absolute(model_path),
                "sha256": file_sha256(staged_model),
                "has_conversion_model": bundle.conversion_model is not None,
            },
        },
    }

    if bundle.conversion_model is not None:
        manifest["artifacts"]["model"]["conversion_model"] = {
            "parameters": bundle.conversion_model.parameters,
            "training_rows": bundle.conversion_model.training_rows,
            "lower_bound": bundle.conversion_model.lower_bound,
            "upper_bound": bundle.conversion_model.upper_bound,
        }

    if evaluation is not None and evaluation_path is not None:
        manifest["final_evaluation"] = evaluation_summary(evaluation)
        manifest["final_evaluation"]["predictions_path"] = _absolute(evaluation_path)
        manifest["final_evaluation"]["predictions_sha256"] = file_sha256(evaluation_path)

    if report_path is not None:
        evidence_key = (
            "final_evaluation" if evaluation is not None else "validation_evidence"
        )
        manifest.setdefault(evidence_key, {})["report_path"] = _absolute(report_path)
        manifest[evidence_key]["report_sha256"] = file_sha256(report_path)

    if skipped_path is not None and skipped is not None:
        manifest["final_holdout_skips"] = {
            "path": _absolute(skipped_path),
            "sha256": file_sha256(skipped_path),
            "rows": int(len(skipped)),
            "players": int(skipped["player_id"].nunique()),
            "error_types": {
                str(key): int(value)
                for key, value in skipped["error_type"].value_counts().items()
            },
        }

    write_json_atomic(manifest, staged_manifest)
    staged_dataset.replace(output_dataset)
    staged_model.replace(model_path)
    staged_manifest.replace(manifest_path)
    print(f"Validated development rows: {len(development)}")
    print(f"Validated final-holdout rows: {len(holdout)}")
    print(f"Full-history rows: {len(full_combined)}")
    print(f"Training cohort: {training_cohort} ({len(training_frame)} rows)")
    print(f"Model type: {model_type}")
    print(f"Wrote production dataset to {output_dataset}")
    print(f"Wrote production model to {model_path}")
    print(f"Wrote release manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

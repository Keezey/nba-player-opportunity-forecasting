"""Fit final V2 residual models and write a reusable model artifact."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.v2.cohorts import select_training_cohort
from src.v2.models import SUPPORTED_MODEL_TYPES, fit_model_bundle, save_model_bundle
from src.v2.tuning import TrustParameters, rebase_training_frame_with_trust


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train final V2 correction models.")
    parser.add_argument("dataset_path")
    parser.add_argument("output_model")
    parser.add_argument("--parameters-json")
    parser.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
    )
    parser.add_argument("--train-through")
    parser.add_argument("--training-cohort", choices=["all", "regular"])
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--apply-tuned-trust", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frame = pd.read_parquet(args.dataset_path)
    tuning = {}
    if args.parameters_json:
        with open(args.parameters_json, encoding="utf-8") as handle:
            tuning = json.load(handle)
    model_type = args.model_type or tuning.get(
        "model_type", "hist_gradient_boosting"
    )
    parameters = (
        tuning.get("model_parameters", {})
        if not args.model_type or args.model_type == tuning.get("model_type")
        else {}
    )
    parameter_cohort = tuning.get("training_cohort")
    if (
        args.training_cohort is not None
        and parameter_cohort is not None
        and args.training_cohort != parameter_cohort
    ):
        raise ValueError("--training-cohort conflicts with the parameter JSON.")
    training_cohort = args.training_cohort or parameter_cohort or "all"
    regular_config = tuning.get("regular_minutes", {})
    frame = select_training_cohort(
        frame,
        cohort=training_cohort,
        absolute_tolerance=float(regular_config.get("absolute_tolerance", 3.0)),
        relative_tolerance=float(regular_config.get("relative_tolerance", 0.15)),
    )
    if args.apply_tuned_trust:
        if "trust_parameters" not in tuning:
            raise ValueError("The parameters JSON does not contain trust_parameters.")
        frame = rebase_training_frame_with_trust(
            frame,
            TrustParameters(**tuning["trust_parameters"]),
            tuning.get("multiplier_bounds"),
        )
    if args.train_through:
        cutoff = pd.to_datetime(args.train_through).normalize()
        frame = frame[pd.to_datetime(frame["game_date"]).dt.normalize() <= cutoff].copy()

    bundle = fit_model_bundle(
        frame,
        model_type=model_type,
        parameters_by_target=parameters,
        conversion_parameters=tuning.get("conversion_parameters"),
        min_rows=args.min_rows,
        random_state=args.random_state,
    )
    bundle.workflow_parameters = tuning.get("workflow_parameters")
    if args.apply_tuned_trust:
        bundle.trust_parameters = tuning.get("trust_parameters")
        bundle.multiplier_bounds = tuning.get("multiplier_bounds")
    output = save_model_bundle(bundle, args.output_model)
    print(f"Model type: {model_type}")
    print(f"Training cohort: {training_cohort} ({len(frame)} rows)")
    if bundle.conversion_model is not None:
        print(
            "rebound conversion: "
            f"{bundle.conversion_model.training_rows} training rows"
        )
    if bundle.multitask_model is not None:
        model = bundle.multitask_model
        print(
            f"multi-task: {model.training_rows} training rows, "
            f"{len(model.networks)} seeds, {model.selected_epochs} epochs"
        )
    else:
        for target, model in bundle.models.items():
            print(f"{target}: {model.training_rows} training rows")
    print(f"Wrote model bundle to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

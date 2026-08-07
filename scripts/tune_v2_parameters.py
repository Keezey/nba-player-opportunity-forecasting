"""Tune V2 residual models and the sample-trust curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.trust import DEFAULT_MULTIPLIER_BOUNDS, MultiplierBounds
from src.v2.models import SUPPORTED_MODEL_TYPES, TARGET_SPECS
from src.v2.tuning import (
    rebase_training_frame_with_trust,
    tune_multitask_residual_models,
    tune_residual_model,
    tune_trust_parameters,
    write_tuning_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune V2 parameters using chronological walk-forward validation."
    )
    parser.add_argument("dataset_path")
    parser.add_argument("output_json")
    parser.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
        default="hist_gradient_boosting",
    )
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-trust", action="store_true")
    parser.add_argument("--trials-dir")
    parser.add_argument("--workflow-parameters-json")
    parser.add_argument(
        "--unbounded-multipliers",
        action="store_true",
        help="Reproduce the legacy workflow without the robust 0.50-1.50 bounds.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frame = pd.read_parquet(args.dataset_path)
    multiplier_bounds = (
        MultiplierBounds()
        if args.unbounded_multipliers
        else DEFAULT_MULTIPLIER_BOUNDS
    )
    trust = None
    model_frame = frame
    if not args.skip_trust:
        trust = tune_trust_parameters(
            frame,
            n_iter=args.n_iter,
            n_splits=args.n_splits,
            random_state=args.random_state,
            multiplier_bounds=multiplier_bounds,
        )
        model_frame = rebase_training_frame_with_trust(
            frame,
            trust.best_parameters,
            multiplier_bounds,
        )
    if args.model_type == "pytorch_multitask":
        results = tune_multitask_residual_models(
            model_frame,
            n_iter=args.n_iter,
            n_splits=args.n_splits,
            min_rows=args.min_rows,
            random_state=args.random_state,
        )
    else:
        results = {
            target: tune_residual_model(
                model_frame,
                target,
                model_type=args.model_type,
                n_iter=args.n_iter,
                n_splits=args.n_splits,
                min_rows=args.min_rows,
                random_state=args.random_state,
            )
            for target in TARGET_SPECS
        }

    workflow_parameters = None
    if args.workflow_parameters_json:
        with open(args.workflow_parameters_json, encoding="utf-8") as handle:
            workflow_parameters = json.load(handle).get("workflow_parameters")
    output = write_tuning_summary(
        results,
        args.output_json,
        trust_result=trust,
        workflow_parameters=workflow_parameters,
        multiplier_bounds=multiplier_bounds,
    )
    trials_dir = Path(args.trials_dir) if args.trials_dir else output.parent / f"{output.stem}_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    for target, result in results.items():
        result.trials.to_csv(trials_dir / f"{target}_model_trials.csv", index=False)
        print(
            f"{target}: V1 MAE {result.mean_v1_mae:.3f}, "
            f"best V2 MAE {result.best_v2_mae:.3f}"
        )
    if trust is not None:
        trust.trials.to_csv(trials_dir / "trust_trials.csv", index=False)
        print(f"Trust objective: {trust.best_objective:.3f}")
    print(f"Wrote best parameters to {output}")
    print(f"Wrote trial details to {trials_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

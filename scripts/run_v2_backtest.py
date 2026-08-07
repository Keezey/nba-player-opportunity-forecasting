"""Train and evaluate V2 on a strictly later chronological holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.v2.cohorts import select_training_cohort
from src.v2.evaluate import (
    chronological_split,
    evaluate_model_bundle,
    write_evaluation_report,
)
from src.v2.importance import (
    group_ablation_table,
    permutation_importance_table,
    write_importance_report,
)
from src.v2.models import SUPPORTED_MODEL_TYPES, fit_model_bundle, save_model_bundle
from src.v2.tuning import TrustParameters, rebase_training_frame_with_trust


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronologically backtest V2 against the frozen V1 predictions."
    )
    parser.add_argument("dataset_path")
    parser.add_argument("report_path")
    parser.add_argument("--parameters-json")
    parser.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--test-dataset",
        help=(
            "Optional untouched Parquet dataset to use as the entire holdout. "
            "Its first game date must be later than every training date."
        ),
    )
    parser.add_argument("--gap-days", type=int, default=0)
    parser.add_argument("--training-cohort", choices=["all", "regular"])
    parser.add_argument(
        "--evaluation-cohort", choices=["all", "regular"], default="all"
    )
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--predictions-csv")
    parser.add_argument("--importance-report")
    parser.add_argument("--importance-csv")
    parser.add_argument("--ablation-csv")
    parser.add_argument("--model-output")
    parser.add_argument("--skip-importance", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--apply-tuned-trust", action="store_true")
    return parser


def external_holdout_split(
    training_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    date_column: str = "game_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and return strictly chronological external train/test data."""
    if training_frame.empty:
        raise ValueError("The training dataset is empty.")
    if test_frame.empty:
        raise ValueError("The external test dataset is empty.")
    if date_column not in training_frame or date_column not in test_frame:
        raise KeyError(f"Both datasets must contain {date_column}.")

    training_dates = pd.to_datetime(
        training_frame[date_column], errors="coerce"
    ).dt.normalize()
    test_dates = pd.to_datetime(test_frame[date_column], errors="coerce").dt.normalize()
    if training_dates.isna().any() or test_dates.isna().any():
        raise ValueError(f"{date_column} contains missing or invalid dates.")
    if test_dates.min() <= training_dates.max():
        raise ValueError(
            "The external holdout must begin after the final training date: "
            f"training ends {training_dates.max().date()}, "
            f"test begins {test_dates.min().date()}."
        )
    return training_frame.copy(), test_frame.copy()


def main() -> int:
    args = build_parser().parse_args()
    training_frame = pd.read_parquet(args.dataset_path)
    external_test_frame = (
        pd.read_parquet(args.test_dataset) if args.test_dataset else None
    )
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
    absolute_tolerance = float(regular_config.get("absolute_tolerance", 3.0))
    relative_tolerance = float(regular_config.get("relative_tolerance", 0.15))

    if external_test_frame is not None:
        train, test = external_holdout_split(
            training_frame,
            external_test_frame,
        )
    else:
        train, test = chronological_split(
            training_frame,
            test_fraction=args.test_fraction,
            gap_days=args.gap_days,
        )
    train = select_training_cohort(
        train,
        cohort=training_cohort,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    test = select_training_cohort(
        test,
        cohort=args.evaluation_cohort,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if args.apply_tuned_trust:
        if "trust_parameters" not in tuning:
            raise ValueError("The parameters JSON does not contain trust_parameters.")
        trust_parameters = TrustParameters(**tuning["trust_parameters"])
        train = rebase_training_frame_with_trust(
            train,
            trust_parameters,
            tuning.get("multiplier_bounds"),
        )
        test = rebase_training_frame_with_trust(
            test,
            trust_parameters,
            tuning.get("multiplier_bounds"),
        )
    bundle = fit_model_bundle(
        train,
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
    evaluation = evaluate_model_bundle(bundle, test)
    report = write_evaluation_report(evaluation, args.report_path)

    predictions_path = (
        Path(args.predictions_csv)
        if args.predictions_csv
        else report.with_name(f"{report.stem}_predictions.csv")
    )
    evaluation.predictions.to_csv(predictions_path, index=False)

    permutation = pd.DataFrame()
    ablation = pd.DataFrame()
    if not args.skip_importance:
        permutation = permutation_importance_table(
            bundle,
            test,
            random_state=args.random_state,
        )
        importance_csv = (
            Path(args.importance_csv)
            if args.importance_csv
            else report.with_name(f"{report.stem}_importance.csv")
        )
        permutation.to_csv(importance_csv, index=False)
    if not args.skip_ablation:
        ablation = group_ablation_table(
            train,
            test,
            model_type=model_type,
            parameters_by_target=parameters,
            min_rows=args.min_rows,
            random_state=args.random_state,
        )
        ablation_csv = (
            Path(args.ablation_csv)
            if args.ablation_csv
            else report.with_name(f"{report.stem}_ablation.csv")
        )
        ablation.to_csv(ablation_csv, index=False)
    if not permutation.empty or not ablation.empty:
        importance_report = (
            Path(args.importance_report)
            if args.importance_report
            else report.with_name(f"{report.stem}_feature_importance.md")
        )
        write_importance_report(permutation, ablation, importance_report)

    if args.model_output:
        save_model_bundle(bundle, args.model_output)

    print(f"Training rows: {len(train)}")
    print(f"Held-out rows: {len(test)}")
    print(f"Training cohort: {training_cohort}")
    print(f"Evaluation cohort: {args.evaluation_cohort}")
    print(evaluation.summary.to_string(index=False))
    print(f"Wrote evaluation report to {report}")
    print(f"Wrote held-out predictions to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

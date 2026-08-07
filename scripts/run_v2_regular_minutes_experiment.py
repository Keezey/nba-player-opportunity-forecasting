"""Compare all-game and regular-minute training on rolling future seasons."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.v2.regular_minutes_experiment import run_regular_minutes_experiment
from src.v2.release import file_sha256, frame_summary, utc_now_iso, write_json_atomic
from src.v2.rolling_experiment import ROLLING_MODEL_TYPES


DEFAULT_TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare all-game versus regular-minute-only training inside "
            "leakage-safe rolling outer-season folds."
        )
    )
    parser.add_argument("dataset_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--test-seasons", nargs="+", default=DEFAULT_TEST_SEASONS
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ROLLING_MODEL_TYPES,
        default=["elastic_net"],
    )
    parser.add_argument(
        "--training-cohorts",
        nargs="+",
        choices=["all", "regular"],
        default=["all", "regular"],
    )
    parser.add_argument("--absolute-tolerance", type=float, default=3.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.15)
    parser.add_argument("--trust-n-iter", type=int, default=20)
    parser.add_argument("--elastic-n-iter", type=int, default=20)
    parser.add_argument("--neural-n-iter", type=int, default=6)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-training-seasons", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compute-threads", type=int, default=2)
    parser.add_argument("--neural-tuning-max-epochs", type=int, default=80)
    parser.add_argument("--neural-tuning-patience", type=int, default=10)
    parser.add_argument("--neural-ensemble-size", type=int, default=5)
    return parser


def _display_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "n/a" if pd.isna(value) else f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows available."
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = " | ".join(_display_value(row[column]) for column in columns)
        lines.append(f"| {values} |")
    return "\n".join(lines)


def _report_tables(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary = summary[
        (summary["evaluation_cohort"] == "regular")
        & (summary["aggregation"] == "game_weighted")
    ][
        [
            "test_season",
            "training_cohort",
            "model_type",
            "stat",
            "training_rows",
            "training_coverage_pct",
            "test_rows",
            "evaluation_coverage_pct",
            "v1_mae",
            "v2_mae",
            "mae_improvement_pct",
            "v2_rmse",
            "v2_bias",
        ]
    ].copy()
    pivot = primary.pivot_table(
        index=["test_season", "model_type", "stat"],
        columns="training_cohort",
        values="v2_mae",
        aggfunc="first",
    ).reset_index()
    if {"all", "regular"}.issubset(pivot.columns):
        pivot["regular_training_minus_all_training_mae"] = (
            pivot["regular"] - pivot["all"]
        )
    delta_columns = [
        "test_season",
        "model_type",
        "stat",
        "all",
        "regular",
        "regular_training_minus_all_training_mae",
    ]
    deltas = pivot.reindex(columns=delta_columns)
    equal_season = (
        primary.groupby(["training_cohort", "model_type", "stat"], as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            mean_training_coverage_pct=("training_coverage_pct", "mean"),
            mean_test_coverage_pct=("evaluation_coverage_pct", "mean"),
            mean_v1_mae=("v1_mae", "mean"),
            mean_v2_mae=("v2_mae", "mean"),
            mean_improvement_pct=("mae_improvement_pct", "mean"),
        )
    )
    all_test = summary[
        (summary["evaluation_cohort"] == "all")
        & (summary["aggregation"] == "game_weighted")
    ][
        [
            "test_season",
            "training_cohort",
            "model_type",
            "stat",
            "v1_mae",
            "v2_mae",
            "mae_improvement_pct",
        ]
    ].copy()
    player_weighted = summary[
        (summary["evaluation_cohort"] == "regular")
        & (summary["aggregation"] == "player_weighted")
    ][
        [
            "test_season",
            "training_cohort",
            "model_type",
            "stat",
            "test_players",
            "v1_mae",
            "v2_mae",
            "mae_improvement_pct",
        ]
    ].copy()
    return {
        "primary": primary,
        "deltas": deltas,
        "equal_season": equal_season,
        "all_test": all_test,
        "player_weighted": player_weighted,
    }


def _write_report(
    summary: pd.DataFrame,
    path: Path,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    tables = _report_tables(summary)
    lines = [
        "# V2.1 Conditional Regular-Minutes Experiment",
        "",
        "A game is regular when `abs(actual_minutes - pregame_expected_minutes)`",
        f"is at most `max({absolute_tolerance:g}, {relative_tolerance:.0%} of expected)`.",
        "The flag uses the outcome only to define a research cohort; it is never a",
        "model input. Every trust and model search remains inside earlier seasons.",
        "",
        "## Regular-Minutes Test Results",
        "",
        _markdown_table(tables["primary"]),
        "",
        "## Effect Of Filtering Training",
        "",
        "Negative deltas mean regular-minute-only training improved conditional MAE.",
        "",
        _markdown_table(tables["deltas"]),
        "",
        "## Equal-Season Summary",
        "",
        _markdown_table(tables["equal_season"]),
        "",
        "## Equal-Player Conditional Results",
        "",
        _markdown_table(tables["player_weighted"]),
        "",
        "## Full-Season Sensitivity",
        "",
        _markdown_table(tables["all_test"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_trials(
    trials: dict[str, pd.DataFrame], output_dir: Path
) -> dict[str, str]:
    written: dict[str, str] = {}
    for name, frame in trials.items():
        path = output_dir.joinpath(*name.split("/")).with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        written[name] = str(path.resolve())
    return written


def main() -> int:
    args = build_parser().parse_args()
    if args.compute_threads < 1:
        raise ValueError("--compute-threads must be positive.")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.compute_threads))
    dataset_path = Path(args.dataset_path)
    frame = pd.read_parquet(dataset_path)
    if "pytorch_multitask" in args.models:
        import torch

        torch.set_num_threads(args.compute_threads)
        torch.set_num_interop_threads(args.compute_threads)

    model_n_iter = {
        "elastic_net": args.elastic_n_iter,
        "pytorch_multitask": args.neural_n_iter,
    }
    with threadpool_limits(limits=args.compute_threads):
        result = run_regular_minutes_experiment(
            frame,
            test_seasons=args.test_seasons,
            model_types=args.models,
            training_cohorts=args.training_cohorts,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            trust_n_iter=args.trust_n_iter,
            model_n_iter=model_n_iter,
            n_splits=args.n_splits,
            min_training_seasons=args.min_training_seasons,
            min_rows=args.min_rows,
            random_state=args.random_state,
            neural_tuning_max_epochs=args.neural_tuning_max_epochs,
            neural_tuning_patience=args.neural_tuning_patience,
            neural_final_ensemble_size=args.neural_ensemble_size,
            progress=print,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    predictions_path = output_dir / "predictions.parquet"
    parameters_path = output_dir / "parameters.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    trials_dir = output_dir / "trials"
    result.summary.to_csv(summary_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)
    write_json_atomic(result.parameters, parameters_path)
    _write_report(
        result.summary,
        report_path,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    trial_paths = _write_trials(result.trials, trials_dir)

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "experiment": "v2_regular_minutes_training",
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": file_sha256(dataset_path),
            **frame_summary(frame),
        },
        "configuration": {
            "test_seasons": list(args.test_seasons),
            "models": list(args.models),
            "training_cohorts": list(args.training_cohorts),
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
            "trust_n_iter": args.trust_n_iter,
            "model_n_iter": model_n_iter,
            "n_splits": args.n_splits,
            "min_training_seasons": args.min_training_seasons,
            "min_rows": args.min_rows,
            "random_state": args.random_state,
            "compute_threads": args.compute_threads,
        },
        "outputs": {
            "summary": str(summary_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "parameters": str(parameters_path.resolve()),
            "report": str(report_path.resolve()),
            "trials": trial_paths,
        },
    }
    write_json_atomic(manifest, manifest_path)
    tables = _report_tables(result.summary)
    print(tables["primary"].to_string(index=False))
    print("\nTraining-filter deltas (negative is better):")
    print(tables["deltas"].to_string(index=False))
    print(f"Wrote regular-minutes experiment to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

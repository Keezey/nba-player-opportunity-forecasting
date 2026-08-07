"""Compare robust matchup-multiplier bounds on rolling future seasons."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.v2.multiplier_bounds_experiment import (
    BOUND_STRATEGIES,
    run_multiplier_bounds_experiment,
)
from src.v2.release import (
    file_sha256,
    frame_summary,
    utc_now_iso,
    write_json_atomic,
)
from src.v2.rolling_experiment import ROLLING_MODEL_TYPES


DEFAULT_TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare unbounded, fixed, and training-percentile matchup "
            "multipliers inside rolling outer-season folds."
        )
    )
    parser.add_argument("dataset_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--test-seasons", nargs="+", default=DEFAULT_TEST_SEASONS
    )
    parser.add_argument(
        "--bound-strategies",
        nargs="+",
        choices=BOUND_STRATEGIES,
        default=list(BOUND_STRATEGIES),
    )
    parser.add_argument(
        "--training-cohorts",
        nargs="+",
        choices=["all", "regular"],
        default=["all", "regular"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ROLLING_MODEL_TYPES,
        default=["elastic_net"],
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


def _equal_season_table(
    summary: pd.DataFrame,
    evaluation_cohort: str,
) -> pd.DataFrame:
    scoped = summary[
        (summary["evaluation_cohort"] == evaluation_cohort)
        & (summary["aggregation"] == "game_weighted")
    ]
    return (
        scoped.groupby(
            ["training_cohort", "bound_strategy", "model_type", "stat"],
            as_index=False,
        )
        .agg(
            seasons=("test_season", "nunique"),
            mean_v2_mae=("v2_mae", "mean"),
            mean_v2_rmse=("v2_rmse", "mean"),
            mean_v2_p99_abs_error=("v2_p99_abs_error", "mean"),
            worst_v2_abs_error=("v2_max_abs_error", "max"),
            mean_bounded_test_pct=("bounded_test_pct", "mean"),
        )
    )


def _strategy_deltas(equal_season: pd.DataFrame) -> pd.DataFrame:
    keys = ["training_cohort", "model_type", "stat"]
    baseline = equal_season[
        equal_season["bound_strategy"] == "unbounded"
    ][keys + ["mean_v2_mae", "mean_v2_p99_abs_error"]].rename(
        columns={
            "mean_v2_mae": "unbounded_mae",
            "mean_v2_p99_abs_error": "unbounded_p99_abs_error",
        }
    )
    compared = equal_season.merge(baseline, on=keys, how="left")
    compared["mae_delta_vs_unbounded"] = (
        compared["mean_v2_mae"] - compared["unbounded_mae"]
    )
    compared["p99_delta_vs_unbounded"] = (
        compared["mean_v2_p99_abs_error"]
        - compared["unbounded_p99_abs_error"]
    )
    return compared[
        keys
        + [
            "bound_strategy",
            "mean_v2_mae",
            "mae_delta_vs_unbounded",
            "mean_v2_p99_abs_error",
            "p99_delta_vs_unbounded",
            "worst_v2_abs_error",
            "mean_bounded_test_pct",
        ]
    ]


def _resolved_bounds(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "test_season",
        "training_cohort",
        "bound_strategy",
        "stat",
        "multiplier_lower_bound",
        "multiplier_upper_bound",
    ]
    return (
        summary[
            (summary["evaluation_cohort"] == "all")
            & (summary["aggregation"] == "game_weighted")
            & summary["stat"].isin(["fga", "reb_chances"])
        ][columns]
        .drop_duplicates()
        .sort_values(columns[:4], kind="stable")
    )


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    regular = _equal_season_table(summary, "regular")
    complete = _equal_season_table(summary, "all")
    lines = [
        "# V2.2 Robust Multiplier Bounds Experiment",
        "",
        "Observed matchup ratios are bounded before sample-size shrinkage. ",
        "Training-percentile bounds are calculated from prior outer-fold rows only.",
        "Negative deltas versus unbounded indicate an improvement.",
        "",
        "## Resolved Bounds",
        "",
        _markdown_table(_resolved_bounds(summary)),
        "",
        "## Regular-Minutes Conditional Results",
        "",
        _markdown_table(_strategy_deltas(regular)),
        "",
        "## Complete-Season Results",
        "",
        _markdown_table(_strategy_deltas(complete)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_trials(
    trials: dict[str, pd.DataFrame],
    output_dir: Path,
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
        result = run_multiplier_bounds_experiment(
            frame,
            test_seasons=args.test_seasons,
            bound_strategies=args.bound_strategies,
            training_cohorts=args.training_cohorts,
            model_types=args.models,
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
    result.summary.to_csv(summary_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)
    write_json_atomic(result.parameters, parameters_path)
    _write_report(result.summary, report_path)
    trial_paths = _write_trials(result.trials, output_dir / "trials")

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "experiment": "v2_multiplier_bounds",
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": file_sha256(dataset_path),
            **frame_summary(frame),
        },
        "configuration": {
            "test_seasons": list(args.test_seasons),
            "bound_strategies": list(args.bound_strategies),
            "training_cohorts": list(args.training_cohorts),
            "models": list(args.models),
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
    print(_strategy_deltas(_equal_season_table(result.summary, "regular")))
    print(f"Wrote multiplier-bounds experiment to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

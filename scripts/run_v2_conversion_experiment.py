"""Run the V2.3 rebound-conversion experiment on frozen opportunity forecasts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.v2.conversion_experiment import (
    CONVERSION_METHODS,
    DEFAULT_SHRINKAGE_STRENGTHS,
    run_conversion_experiment,
)
from src.v2.release import (
    file_sha256,
    frame_summary,
    utc_now_iso,
    write_json_atomic,
)


DEFAULT_TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current, shrunken, and Elastic Net rebound-conversion "
            "rates using frozen robust V2 rebound-chance predictions."
        )
    )
    parser.add_argument("dataset_path")
    parser.add_argument("base_predictions_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--test-seasons", nargs="+", default=DEFAULT_TEST_SEASONS
    )
    parser.add_argument(
        "--training-cohorts",
        nargs="+",
        choices=["all", "regular"],
        default=["all", "regular"],
    )
    parser.add_argument("--base-model-type", default="elastic_net")
    parser.add_argument(
        "--base-bound-strategy", default="fixed_0.50_1.50"
    )
    parser.add_argument("--absolute-tolerance", type=float, default=3.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.15)
    parser.add_argument(
        "--shrinkage-strengths",
        nargs="+",
        type=float,
        default=list(DEFAULT_SHRINKAGE_STRENGTHS),
    )
    parser.add_argument("--elastic-n-iter", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-training-seasons", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compute-threads", type=int, default=2)
    return parser


def _display_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "n/a" if pd.isna(value) else f"{float(value):.4f}"
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


def _equal_season_results(
    summary: pd.DataFrame,
    *,
    evaluation_cohort: str,
) -> pd.DataFrame:
    scoped = summary[
        (summary["evaluation_cohort"] == evaluation_cohort)
        & (summary["aggregation"] == "game_weighted")
    ]
    grouped = (
        scoped.groupby(["training_cohort", "method"], as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            seasons_improved=(
                "mae_improvement_vs_current",
                lambda values: int((values > 0).sum()),
            ),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias=("bias", "mean"),
            mean_p99_abs_error=("p99_abs_error", "mean"),
            worst_abs_error=("max_abs_error", "max"),
            mean_weighted_conversion_mae=("weighted_conversion_mae", "mean"),
        )
    )
    current = grouped[grouped["method"] == "current"][
        ["training_cohort", "mean_mae"]
    ].rename(columns={"mean_mae": "current_mae"})
    result = grouped.merge(current, on="training_cohort", how="left")
    result["mae_improvement_vs_current"] = (
        result["current_mae"] - result["mean_mae"]
    )
    result["mae_improvement_pct_vs_current"] = np.where(
        result["current_mae"] > 0,
        100
        * result["mae_improvement_vs_current"]
        / result["current_mae"],
        np.nan,
    )
    return result[
        [
            "training_cohort",
            "method",
            "seasons",
            "seasons_improved",
            "mean_mae",
            "mae_improvement_vs_current",
            "mae_improvement_pct_vs_current",
            "mean_rmse",
            "mean_bias",
            "mean_p99_abs_error",
            "worst_abs_error",
            "mean_weighted_conversion_mae",
        ]
    ].sort_values(["training_cohort", "mean_mae"], kind="stable")


def _season_results(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["evaluation_cohort"] == "all")
        & (summary["aggregation"] == "game_weighted")
    ][
        [
            "test_season",
            "training_cohort",
            "method",
            "test_rows",
            "test_players",
            "mae",
            "mae_improvement_vs_current",
            "mae_improvement_pct_vs_current",
            "rmse",
            "bias",
            "p99_abs_error",
            "max_abs_error",
        ]
    ].sort_values(
        ["training_cohort", "test_season", "mae"], kind="stable"
    )


def _write_report(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    path: Path,
) -> None:
    complete = _equal_season_results(summary, evaluation_cohort="all")
    regular = _equal_season_results(summary, evaluation_cohort="regular")
    lines = [
        "# V2.3 Rebound Conversion Experiment",
        "",
        "Frozen robust V2 rebound-chance forecasts are converted to rebounds "
        "with three alternatives: the current recent-player ratio, empirical "
        "shrinkage toward a training-only position prior, and a weighted "
        "Elastic Net adjustment. Negative MAE deltas indicate regression; "
        "positive improvements indicate a better result than current.",
        "",
        "Rows with zero recorded rebound chances are excluded only from "
        "conversion-model fitting and conversion-rate diagnostics. They remain "
        "in final rebound evaluation.",
        "",
        "## Complete-Season Equal-Season Results",
        "",
        _markdown_table(complete),
        "",
        "## Regular-Minutes Conditional Results",
        "",
        _markdown_table(regular),
        "",
        "## Complete-Season Results By Test Season",
        "",
        _markdown_table(_season_results(summary)),
        "",
        "## Paired Player-Cluster Bootstrap",
        "",
        "Positive intervals favor the alternative conversion method. Players "
        "are resampled within each test season, preserving each sampled "
        "player's full sequence of games.",
        "",
        _markdown_table(bootstrap),
        "",
        "## Adoption Standard",
        "",
        "Adopt a conversion alternative only if it improves aggregate rebound "
        "MAE by about 1%, remains stable across held-out seasons, and does not "
        "materially worsen bias or tail errors.",
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
    base_predictions_path = Path(args.base_predictions_path)
    frame = pd.read_parquet(dataset_path)
    base_predictions = pd.read_parquet(base_predictions_path)

    with threadpool_limits(limits=args.compute_threads):
        result = run_conversion_experiment(
            frame,
            base_predictions,
            test_seasons=args.test_seasons,
            training_cohorts=args.training_cohorts,
            base_model_type=args.base_model_type,
            base_bound_strategy=args.base_bound_strategy,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            shrinkage_strengths=args.shrinkage_strengths,
            elastic_n_iter=args.elastic_n_iter,
            n_splits=args.n_splits,
            min_training_seasons=args.min_training_seasons,
            min_rows=args.min_rows,
            bootstrap_samples=args.bootstrap_samples,
            random_state=args.random_state,
            progress=print,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    predictions_path = output_dir / "predictions.parquet"
    parameters_path = output_dir / "parameters.json"
    bootstrap_path = output_dir / "bootstrap.csv"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    result.summary.to_csv(summary_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)
    result.bootstrap.to_csv(bootstrap_path, index=False)
    write_json_atomic(result.parameters, parameters_path)
    _write_report(result.summary, result.bootstrap, report_path)
    trial_paths = _write_trials(result.trials, output_dir / "trials")

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "experiment": "v2_rebound_conversion",
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": file_sha256(dataset_path),
            **frame_summary(frame),
        },
        "base_predictions": {
            "path": str(base_predictions_path.resolve()),
            "sha256": file_sha256(base_predictions_path),
            "rows": int(len(base_predictions)),
            "columns": int(len(base_predictions.columns)),
        },
        "configuration": {
            "test_seasons": list(args.test_seasons),
            "training_cohorts": list(args.training_cohorts),
            "methods": list(CONVERSION_METHODS),
            "base_model_type": args.base_model_type,
            "base_bound_strategy": args.base_bound_strategy,
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
            "shrinkage_strengths": list(args.shrinkage_strengths),
            "elastic_n_iter": args.elastic_n_iter,
            "n_splits": args.n_splits,
            "min_training_seasons": args.min_training_seasons,
            "min_rows": args.min_rows,
            "bootstrap_samples": args.bootstrap_samples,
            "random_state": args.random_state,
            "compute_threads": args.compute_threads,
        },
        "outputs": {
            "summary": str(summary_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "bootstrap": str(bootstrap_path.resolve()),
            "parameters": str(parameters_path.resolve()),
            "report": str(report_path.resolve()),
            "trials": trial_paths,
        },
    }
    write_json_atomic(manifest, manifest_path)
    print(_equal_season_results(result.summary, evaluation_cohort="all"))
    print(f"Wrote rebound-conversion experiment to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

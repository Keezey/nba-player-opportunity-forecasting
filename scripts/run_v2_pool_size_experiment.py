"""Compare fixed V2 model families across nested target-player pool sizes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.v2.models import SUPPORTED_MODEL_TYPES
from src.v2.player_pool import read_player_pool
from src.v2.pool_experiment import run_pool_size_experiment
from src.v2.release import file_sha256, utc_now_iso, write_json_atomic
from src.v2.tuning import TrustParameters


DEFAULT_MODELS = ["elastic_net", "hist_gradient_boosting", "pytorch_multitask"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit fixed Elastic Net, histogram boosting, and PyTorch residual "
            "models across nested player pools and score one common holdout."
        )
    )
    parser.add_argument("development_dataset")
    parser.add_argument("test_dataset")
    parser.add_argument("output_dir")
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--pool-prefix", default="v2_1_target_pool")
    parser.add_argument(
        "--pool-sizes", type=int, nargs="+", default=[30, 60, 90, 150, 250]
    )
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODEL_TYPES, default=DEFAULT_MODELS
    )
    parser.add_argument("--elastic-parameters")
    parser.add_argument("--boosting-parameters")
    parser.add_argument("--neural-parameters")
    parser.add_argument("--trust-parameters", required=True)
    parser.add_argument("--neural-ensemble-size", type=int)
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--compute-threads",
        type=int,
        default=1,
        help="Native math threads per fit. One avoids costly BLAS oversubscription.",
    )
    return parser


def _load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _parameter_paths(args) -> dict[str, str | None]:
    return {
        "elastic_net": args.elastic_parameters,
        "hist_gradient_boosting": args.boosting_parameters,
        "pytorch_multitask": args.neural_parameters,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    game_weighted = summary[
        (summary["scope"] == "all")
        & (summary["aggregation"] == "game_weighted")
    ][
        [
            "pool_size",
            "model_type",
            "stat",
            "training_rows",
            "fit_predict_seconds",
            "test_rows",
            "v1_mae",
            "v2_mae",
            "mae_improvement_pct",
            "v2_rmse",
            "v2_bias",
        ]
    ].copy()
    player_weighted = summary[
        (summary["scope"] == "all")
        & (summary["aggregation"] == "player_weighted")
    ][
        [
            "pool_size",
            "model_type",
            "stat",
            "test_players",
            "v1_mae",
            "v2_mae",
            "mae_improvement_pct",
        ]
    ].copy()
    winners = (
        game_weighted.sort_values(
            ["pool_size", "stat", "v2_mae", "fit_predict_seconds"]
        )
        .groupby(["pool_size", "stat"], as_index=False)
        .first()[
            [
                "pool_size",
                "stat",
                "model_type",
                "v2_mae",
                "mae_improvement_pct",
                "fit_predict_seconds",
            ]
        ]
    )
    lines = [
        "# V2.1 Pool-Size Learning Curve",
        "",
        "All models use fixed parameters and one common trust curve. Every row is",
        "scored on the same chronologically later test dataset. Full game-weighted,",
        "player-weighted, seen-player, and unseen-player metrics are in summary.csv.",
        "",
        "## Common-Test Results",
        "",
        _markdown_table(game_weighted),
        "",
        "## Lowest Game-Weighted MAE",
        "",
        _markdown_table(winners),
        "",
        "## Equal-Player Results",
        "",
        _markdown_table(player_weighted),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.compute_threads <= 0:
        raise ValueError("--compute-threads must be positive.")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.compute_threads))
    development_path = Path(args.development_dataset)
    test_path = Path(args.test_dataset)
    development = pd.read_parquet(development_path)
    test = pd.read_parquet(test_path)

    pool_dir = Path(args.pool_dir)
    pool_paths = {
        size: pool_dir / f"{args.pool_prefix}_{size}.csv"
        for size in sorted(set(args.pool_sizes))
    }
    pools = {size: read_player_pool(path) for size, path in pool_paths.items()}

    parameter_paths = _parameter_paths(args)
    model_parameters = {}
    parameter_manifest = {}
    for model_type in args.models:
        path_value = parameter_paths[model_type]
        if not path_value:
            raise ValueError(f"A parameter JSON is required for {model_type}.")
        path = Path(path_value)
        payload = _load_json(path)
        if payload.get("model_type") != model_type:
            raise ValueError(
                f"{path} contains model_type={payload.get('model_type')!r}, "
                f"expected {model_type!r}."
            )
        parameters = payload["model_parameters"]
        if model_type == "pytorch_multitask" and args.neural_ensemble_size:
            parameters = {
                target: {**values, "ensemble_size": args.neural_ensemble_size}
                for target, values in parameters.items()
            }
        model_parameters[model_type] = parameters
        parameter_manifest[model_type] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "parameters": parameters,
        }

    trust_path = Path(args.trust_parameters)
    trust_payload = _load_json(trust_path)
    trust = TrustParameters(**trust_payload["trust_parameters"])
    if "pytorch_multitask" in args.models:
        import torch

        torch.set_num_threads(args.compute_threads)
        torch.set_num_interop_threads(args.compute_threads)
    with threadpool_limits(limits=args.compute_threads):
        result = run_pool_size_experiment(
            development,
            test,
            pools=pools,
            model_parameters=model_parameters,
            trust_parameters=trust,
            min_rows=args.min_rows,
            random_state=args.random_state,
            progress=print,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    predictions_path = output_dir / "predictions.parquet"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    result.summary.to_csv(summary_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)
    _write_report(result.summary, report_path)

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "experiment": "v2_pool_size_learning_curve",
        "pool_sizes": sorted(pools),
        "models": list(model_parameters),
        "random_state": args.random_state,
        "compute_threads": args.compute_threads,
        "development": {
            "path": str(development_path.resolve()),
            "sha256": file_sha256(development_path),
            "rows": int(len(development)),
        },
        "test": {
            "path": str(test_path.resolve()),
            "sha256": file_sha256(test_path),
            "rows": int(len(test)),
        },
        "pools": {
            str(size): {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for size, path in pool_paths.items()
        },
        "model_parameters": parameter_manifest,
        "trust_parameters": {
            "path": str(trust_path.resolve()),
            "sha256": file_sha256(trust_path),
            "parameters": trust_payload["trust_parameters"],
        },
        "outputs": {
            "summary": str(summary_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    write_json_atomic(manifest, manifest_path)
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote report to {report_path}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

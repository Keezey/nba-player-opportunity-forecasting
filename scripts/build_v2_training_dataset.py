"""Build leakage-safe historical rows for V2 machine learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.live_data import resolve_player
from src.v2.player_pool import read_player_pool
from src.v2.training_data import build_training_dataset, write_training_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V1 historically and save V2 training rows to Parquet."
    )
    parser.add_argument("output_path")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument(
        "players",
        nargs="*",
        help="Target player names or IDs. Optional when --player-pool is supplied.",
    )
    parser.add_argument(
        "--player-pool",
        help="CSV produced by scripts.build_v2_player_pool.",
    )
    parser.add_argument("--skipped-output")
    parser.add_argument("--workflow-parameters-json")
    parser.add_argument("--candidate-player", action="append", default=[])
    parser.add_argument("--top-n-similar", type=int, default=10)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--min-minutes-ratio", type=float, default=0.75)
    parser.add_argument("--position-weight", type=float, default=0.35)
    parser.add_argument("--require-same-starter", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent target players to process in parallel.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--store-dir",
        default=None,
        help=(
            "Optional historical-store directory. Defaults to "
            "data/processed/player_games."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    player_queries: list[str | int] = list(args.players)
    if args.player_pool:
        player_queries.extend(read_player_pool(args.player_pool))
    player_queries = list(dict.fromkeys(player_queries))
    if not player_queries:
        parser.error("provide at least one player or use --player-pool")

    candidate_ids = None
    if args.candidate_player:
        candidate_ids = [resolve_player(player)[0] for player in args.candidate_player]

    workflow = {}
    if args.workflow_parameters_json:
        with open(args.workflow_parameters_json, encoding="utf-8") as handle:
            workflow = json.load(handle).get("workflow_parameters", {})
    settings = {
        "top_n_similar": args.top_n_similar,
        "lookback_days": args.lookback_days,
        "min_games": args.min_games,
        "min_minutes_ratio": args.min_minutes_ratio,
        "position_weight": args.position_weight,
        "require_same_starter_flag": args.require_same_starter,
    }
    settings.update(workflow)

    def report_progress(index, total, player_query, rows, skipped):
        if index == 1 or index % 10 == 0 or index == total:
            print(
                f"Targets: {index}/{total}; rows={rows}; skipped={skipped}; "
                f"latest={player_query}",
                flush=True,
            )

    result = build_training_dataset(
        player_queries,
        args.start_date,
        args.end_date,
        candidate_player_ids=candidate_ids,
        refresh=args.refresh,
        store_dir=args.store_dir,
        progress=report_progress,
        workers=args.workers,
        **settings,
    )
    skipped_path = args.skipped_output
    if skipped_path is None:
        output = Path(args.output_path)
        skipped_path = output.with_name(f"{output.stem}_skipped.csv")
    written, written_skipped = write_training_dataset(
        result,
        args.output_path,
        skipped_path=skipped_path,
    )
    print(f"Games found: {result.games_found}")
    print(f"Target players requested: {len(player_queries)}")
    print(f"Training rows: {result.predictions_built}")
    print(f"Skipped games: {len(result.skipped)}")
    print(f"Wrote training data to {written}")
    print(f"Wrote skipped-game details to {written_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

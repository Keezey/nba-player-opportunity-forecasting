"""Build a reproducible, stratified target-player pool for V2 evaluation."""

from __future__ import annotations

import argparse

from src.v2.player_pool import build_player_pool, write_player_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select eligible target players from local historical seasons, "
            "balanced across broad position and usage tiers."
        )
    )
    parser.add_argument("output_path", help="CSV path for the selected player pool.")
    parser.add_argument(
        "seasons",
        nargs="+",
        help="Source seasons used to determine eligibility, such as 2021-22 2022-23.",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help="Optional historical-store directory.",
    )
    parser.add_argument("--min-games", type=int, default=30)
    parser.add_argument("--min-average-minutes", type=float, default=20.0)
    parser.add_argument("--min-starter-rate", type=float, default=0.50)
    parser.add_argument("--min-data-coverage", type=float, default=0.95)
    parser.add_argument("--players-per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-eligible",
        action="store_true",
        help="Save every eligible player instead of sampling each stratum.",
    )
    parser.add_argument(
        "--eligible-output",
        default=None,
        help="Optional CSV path for the full eligible-player audit table.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_player_pool(
        args.seasons,
        store_dir=args.store_dir,
        min_games=args.min_games,
        min_average_minutes=args.min_average_minutes,
        min_starter_rate=args.min_starter_rate,
        min_data_coverage=args.min_data_coverage,
        players_per_stratum=args.players_per_stratum,
        random_state=args.seed,
        select_all=args.all_eligible,
    )
    selected_path, eligible_path = write_player_pool(
        result,
        args.output_path,
        eligible_output_path=args.eligible_output,
    )

    eligible_player_seasons = int(result.player_seasons["eligible"].sum())
    print(f"Source seasons: {', '.join(args.seasons)}")
    print(f"Eligible player-seasons: {eligible_player_seasons}")
    print(f"Unique eligible players: {len(result.eligible_players)}")
    print(f"Selected target players: {len(result.selected)}")
    print(result.strata_summary.to_string(index=False))
    print(f"Wrote selected pool to {selected_path}")
    print(f"Wrote eligible-player audit table to {eligible_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

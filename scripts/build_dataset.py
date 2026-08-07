"""Build and save a player-game dataset for historical backtesting.

Example:
    python -m scripts.build_dataset data/processed/player_game_2025-26.parquet "Amen Thompson" "Alperen Sengun" --season 2025-26
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NBA game logs/tracking for selected players and save a player_game dataset."
    )
    parser.add_argument("output_path", type=Path, help="Where to save the dataset, usually data/processed/*.parquet.")
    parser.add_argument(
        "players",
        nargs="+",
        help="Player names or NBA Stats IDs to include. Include target plus candidate/manual comparison players.",
    )
    parser.add_argument("--season", type=str, default="2025-26", help="NBA season string, e.g. 2025-26.")
    parser.add_argument("--season-type", type=str, default="Regular Season", help="NBA season type.")
    parser.add_argument("--date-from", type=str, default=None, help="Optional first game date, YYYY-MM-DD.")
    parser.add_argument("--date-to", type=str, default=None, help="Optional last game date, YYYY-MM-DD.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached NBA responses.")
    parser.add_argument("--verbose", action="store_true", help="Print raw fetch/debug details.")
    parser.add_argument(
        "--store-dir",
        default=None,
        help=(
            "Optional historical-store directory. Defaults to "
            "data/processed/player_games."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from src.dataset import build_player_game_df
    from src.live_data import resolve_player

    resolved = [resolve_player(player) for player in args.players]
    player_ids = sorted({player_id for player_id, _ in resolved})

    print("Resolved players:")
    for player_id, player_name in resolved:
        print(f"- {player_name}: {player_id}")

    df = build_player_game_df(
        player_ids=player_ids,
        season=args.season,
        season_type=args.season_type,
        date_from=args.date_from,
        date_to=args.date_to,
        refresh=args.refresh,
        verbose=args.verbose,
        store_dir=args.store_dir,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix.lower() == ".csv":
        df.to_csv(args.output_path, index=False)
    else:
        df.to_parquet(args.output_path, index=False)

    print(f"Wrote dataset to {args.output_path}")
    print(f"Rows: {len(df)}")
    print(f"Players: {df['player_id'].nunique() if not df.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

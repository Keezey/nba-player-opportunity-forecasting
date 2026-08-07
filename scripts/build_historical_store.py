"""Build resumable, season-partitioned local NBA player-game data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import HISTORICAL_STORE_DIR
from src.historical_store import (
    build_season_store,
    load_store_manifest,
    season_store_exists,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download season-wide box scores and game tracking into a local "
            "player-game warehouse. Successful endpoint calls are cached for resume."
        )
    )
    parser.add_argument("seasons", nargs="+", help='NBA seasons such as "2025-26".')
    parser.add_argument(
        "--store-dir",
        default=str(HISTORICAL_STORE_DIR),
        help="Season-partitioned output directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rebuild a season whose player_games.parquet already exists.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass every existing request cache entry and download again.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def _print_manifest(manifest: dict) -> None:
    print(
        f"  {manifest['rows']} player-games, {manifest['players']} players, "
        f"{manifest['games']} games"
    )
    print(
        f"  tracking coverage {100 * manifest['tracking_row_coverage']:.1f}%, "
        f"usage coverage {100 * manifest['advanced_row_coverage']:.1f}%"
    )
    print(f"  missing tracking games: {len(manifest['missing_tracking_games'])}")


def main() -> int:
    args = build_parser().parse_args()
    store_dir = Path(args.store_dir)
    for season in args.seasons:
        if args.skip_existing and season_store_exists(season, store_dir):
            print(f"{season}: existing partition skipped")
            manifest = load_store_manifest(season, store_dir)
            if manifest:
                _print_manifest(manifest)
            continue

        result = build_season_store(
            season,
            store_dir=store_dir,
            refresh=args.refresh,
            verbose=True,
            progress_every=args.progress_every,
        )
        print(f"{season}: wrote {result.data_path}")
        print(f"{season}: wrote {result.manifest_path}")
        _print_manifest(result.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

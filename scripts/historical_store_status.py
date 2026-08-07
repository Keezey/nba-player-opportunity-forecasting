"""Print coverage information for local historical season partitions."""

from __future__ import annotations

import argparse

from src.config import HISTORICAL_STORE_DIR
from src.historical_store import available_store_seasons, load_store_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show local NBA warehouse coverage.")
    parser.add_argument("--store-dir", default=str(HISTORICAL_STORE_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seasons = available_store_seasons(args.store_dir)
    if not seasons:
        print(f"No season partitions found in {args.store_dir}")
        return 0

    rows = []
    for season in seasons:
        manifest = load_store_manifest(season, args.store_dir) or {}
        rows.append(
            {
                "season": season,
                "games": manifest.get("games", "?"),
                "players": manifest.get("players", "?"),
                "rows": manifest.get("rows", "?"),
                "tracking_pct": (
                    f"{100 * manifest['tracking_row_coverage']:.1f}"
                    if "tracking_row_coverage" in manifest
                    else "?"
                ),
                "usage_pct": (
                    f"{100 * manifest['advanced_row_coverage']:.1f}"
                    if "advanced_row_coverage" in manifest
                    else "?"
                ),
                "missing_tracking_games": len(manifest.get("missing_tracking_games", [])),
            }
        )

    headers = list(rows[0])
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

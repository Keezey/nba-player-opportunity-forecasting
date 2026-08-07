"""Create reproducible nested target pools for V2 learning curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.v2.player_pool import build_nested_player_pools
from src.v2.release import file_sha256, utc_now_iso, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build nested, stratified target-player pools from an eligibility audit."
    )
    parser.add_argument("eligible_players_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--anchor-pool")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[30, 60, 90, 150, 250],
    )
    parser.add_argument("--prefix", default="v2_1_target_pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    eligible_path = Path(args.eligible_players_csv)
    anchor_path = Path(args.anchor_pool) if args.anchor_pool else None
    eligible = pd.read_csv(eligible_path)
    anchor = pd.read_csv(anchor_path) if anchor_path else None
    pools = build_nested_player_pools(
        eligible,
        pool_sizes=args.sizes,
        anchor_players=anchor,
        random_state=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_dir / f"{args.prefix}_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "seed": args.seed,
        "design": "nested_round_robin_stratified",
        "eligible_source": {
            "path": str(eligible_path.resolve()),
            "sha256": file_sha256(eligible_path),
            "players": int(eligible["player_id"].nunique()),
        },
        "anchor_source": None,
        "pools": {},
    }
    if anchor_path is not None:
        manifest["anchor_source"] = {
            "path": str(anchor_path.resolve()),
            "sha256": file_sha256(anchor_path),
            "players": int(anchor["player_id"].nunique()),
        }

    for size, pool in pools.items():
        path = output_dir / f"{args.prefix}_{size}.csv"
        pool.to_csv(path, index=False)
        manifest["pools"][str(size)] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "players": int(pool["player_id"].nunique()),
            "strata": {
                str(key): int(value)
                for key, value in pool["stratum"].value_counts().sort_index().items()
            },
        }
        print(f"Pool {size}: {path}")
        print(pool["stratum"].value_counts().sort_index().to_string())

    if anchor is not None:
        largest = pools[max(pools)]
        anchor_ids = set(pd.to_numeric(anchor["player_id"]).astype(int))
        incremental = largest[~largest["player_id"].isin(anchor_ids)].copy()
        incremental_path = output_dir / f"{args.prefix}_additional_{len(incremental)}.csv"
        incremental.to_csv(incremental_path, index=False)
        manifest["incremental_after_anchor"] = {
            "path": str(incremental_path.resolve()),
            "sha256": file_sha256(incremental_path),
            "players": int(incremental["player_id"].nunique()),
        }
        print(f"Incremental players after anchor: {incremental_path}")

    write_json_atomic(manifest, manifest_path)
    print(f"Wrote pool manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

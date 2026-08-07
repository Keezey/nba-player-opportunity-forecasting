"""Validate and merge disjoint V2 player-game datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.v2.release import file_sha256, frame_summary, merge_training_datasets, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge V2 Parquet datasets after schema and duplicate-key validation."
    )
    parser.add_argument("output_path")
    parser.add_argument("input_paths", nargs="+")
    parser.add_argument("--manifest")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = [Path(path) for path in args.input_paths]
    frames = {str(path): pd.read_parquet(path) for path in paths}
    combined = merge_training_datasets(frames)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.staging")
    combined.to_parquet(temporary, index=False)
    temporary.replace(output)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output.with_name(f"{output.stem}_manifest.json")
    )
    write_json_atomic(
        {
            "schema_version": 1,
            "inputs": [
                {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    **frame_summary(frames[str(path)]),
                }
                for path in paths
            ],
            "output": {
                "path": str(output.resolve()),
                "sha256": file_sha256(output),
                **frame_summary(combined),
            },
        },
        manifest_path,
    )
    print(f"Merged {len(paths)} datasets into {len(combined)} rows")
    print(f"Wrote {output}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

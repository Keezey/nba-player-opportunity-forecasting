"""Freeze validated rolling-experiment parameters for production training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.v2.release import file_sha256, utc_now_iso, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the final pre-holdout opportunity, trust, bounds, and "
            "conversion settings into one parameter file per training cohort."
        )
    )
    parser.add_argument("multiplier_parameters_json")
    parser.add_argument("conversion_parameters_json")
    parser.add_argument("output_dir")
    parser.add_argument("--selection-season", default="2025-26")
    parser.add_argument("--bound-strategy", default="fixed_0.50_1.50")
    parser.add_argument("--model-type", default="elastic_net")
    parser.add_argument(
        "--cohorts", nargs="+", choices=["all", "regular"], default=["all", "regular"]
    )
    parser.add_argument("--absolute-tolerance", type=float, default=3.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.15)
    return parser


def extract_production_parameters(
    multiplier_parameters: Mapping[str, Any],
    conversion_parameters: Mapping[str, Any],
    *,
    selection_season: str,
    training_cohort: str,
    bound_strategy: str,
    model_type: str,
    absolute_tolerance: float = 3.0,
    relative_tolerance: float = 0.15,
) -> dict[str, Any]:
    """Build the standard frozen-parameter payload for one cohort."""
    try:
        opportunity_fold = multiplier_parameters[selection_season]
        opportunity = opportunity_fold["training_cohorts"][training_cohort][
            bound_strategy
        ]
        model = opportunity["models"][model_type]
        conversion_fold = conversion_parameters[selection_season]
        conversion = conversion_fold["training_cohorts"][training_cohort]
    except KeyError as exc:
        raise KeyError(
            "The experiment outputs do not contain the requested production "
            f"selection: season={selection_season!r}, cohort={training_cohort!r}, "
            f"bounds={bound_strategy!r}, model={model_type!r}."
        ) from exc

    opportunity_seasons = list(opportunity_fold.get("training_seasons", []))
    conversion_seasons = list(conversion_fold.get("training_seasons", []))
    if opportunity_seasons != conversion_seasons:
        raise ValueError(
            "Opportunity and conversion parameters were not selected from the "
            "same training seasons."
        )

    targets = model.get("targets", {})
    required_targets = {"fga", "reb_chances"}
    if set(targets) != required_targets:
        raise ValueError(
            f"Expected opportunity targets {sorted(required_targets)}; "
            f"found {sorted(targets)}."
        )
    if "elastic_net" not in conversion:
        raise ValueError("Conversion output does not contain Elastic Net parameters.")

    return {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "parameter_role": "production_hyperparameters",
        "selection": {
            "outer_test_season": selection_season,
            "training_seasons": opportunity_seasons,
            "training_rows": int(opportunity["training_rows"]),
            "bound_strategy": bound_strategy,
            "methodology": (
                "Parameters selected using only seasons before the named outer "
                "test season, then frozen before final all-history fitting."
            ),
        },
        "model_type": model_type,
        "training_cohort": training_cohort,
        "model_parameters": {
            target: dict(targets[target]["parameters"])
            for target in sorted(required_targets)
        },
        "model_scores": {
            target: {
                "v1_mae": float(targets[target]["v1_mae"]),
                "v2_mae": float(targets[target]["v2_mae"]),
            }
            for target in sorted(required_targets)
        },
        "trust_parameters": dict(opportunity["trust"]["parameters"]),
        "trust_objective": float(opportunity["trust"]["objective"]),
        "multiplier_bounds": dict(opportunity["multiplier_bounds"]),
        "conversion_parameters": dict(conversion["elastic_net"]),
        "conversion_selection": {
            "training_rows": int(conversion["training_rows"]),
            "usable_rows": int(conversion["conversion_model_rows"]),
            "lower_bound": float(conversion["conversion_lower_bound"]),
            "upper_bound": float(conversion["conversion_upper_bound"]),
        },
        "regular_minutes": {
            "absolute_tolerance": float(absolute_tolerance),
            "relative_tolerance": float(relative_tolerance),
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    multiplier_path = Path(args.multiplier_parameters_json)
    conversion_path = Path(args.conversion_parameters_json)
    with multiplier_path.open(encoding="utf-8") as handle:
        multiplier_parameters = json.load(handle)
    with conversion_path.open(encoding="utf-8") as handle:
        conversion_parameters = json.load(handle)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cohort in args.cohorts:
        payload = extract_production_parameters(
            multiplier_parameters,
            conversion_parameters,
            selection_season=args.selection_season,
            training_cohort=cohort,
            bound_strategy=args.bound_strategy,
            model_type=args.model_type,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        payload["sources"] = {
            "multiplier_parameters": {
                "path": str(multiplier_path.resolve()),
                "sha256": file_sha256(multiplier_path),
            },
            "conversion_parameters": {
                "path": str(conversion_path.resolve()),
                "sha256": file_sha256(conversion_path),
            },
        }
        output = output_dir / f"parameters_{cohort}.json"
        write_json_atomic(payload, output)
        print(f"Wrote {cohort} production parameters to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

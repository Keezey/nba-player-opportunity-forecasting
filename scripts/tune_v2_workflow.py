"""Expensive V1 workflow tuning for V2 dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.v2.tuning import tune_workflow_parameters, write_tuning_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun V1 to optimize lookback, similar-player count, role weights, "
            "and position handling. This makes many cached NBA API requests."
        )
    )
    parser.add_argument("output_json")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument("players", nargs="+")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--coverage-penalty", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trials-csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = tune_workflow_parameters(
        args.players,
        args.start_date,
        args.end_date,
        n_iter=args.n_iter,
        coverage_penalty=args.coverage_penalty,
        random_state=args.random_state,
    )
    output = write_tuning_summary({}, args.output_json, workflow_result=result)
    trials_path = (
        Path(args.trials_csv)
        if args.trials_csv
        else output.with_name(f"{output.stem}_workflow_trials.csv")
    )
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    result.trials.to_csv(trials_path, index=False)
    print(f"Reference rows: {result.reference_rows}")
    print(f"Best objective: {result.best_objective:.3f}")
    print(f"Wrote best workflow parameters to {output}")
    print(f"Wrote workflow trials to {trials_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Run a backtest from a saved player_game parquet/csv and write a report.

Example:
    python -m scripts.run_backtest_report data/processed/player_game_2025-26.parquet reports/backtest.md
    python -m scripts.run_backtest_report data/processed/player_game_2025-26.parquet reports/jokic.md --target-player-id 203999 --start-date 2025-11-01 --end-date 2025-11-30
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_dataset(path: Path) -> pd.DataFrame:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Build it first, for example: "
            f"python -m scripts.build_dataset {path} \"Amen Thompson\" "
            f"\"Scottie Barnes\" \"Josh Giddey\" --season 2025-26"
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a historical backtest from a saved player_game dataset and write a Markdown report."
    )
    parser.add_argument("dataset_path", type=Path, help="Path to saved player_game parquet/csv.")
    parser.add_argument("report_path", type=Path, help="Where to write the Markdown report.")
    parser.add_argument(
        "--target-player-id",
        type=int,
        action="append",
        dest="target_player_ids",
        help="Limit evaluation to this player ID. Can be passed multiple times.",
    )
    parser.add_argument(
        "--candidate-player-id",
        type=int,
        action="append",
        dest="candidate_player_ids",
        help="Limit similarity candidates to this player ID. Can be passed multiple times.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="First game date to evaluate, YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Last game date to evaluate, YYYY-MM-DD.")
    parser.add_argument("--top-n-similar", type=int, default=10, help="Number of automated similar players to use.")
    parser.add_argument("--lookback-days", type=int, default=30, help="Recent history window for profiles/baselines.")
    parser.add_argument("--min-games", type=int, default=5, help="Minimum qualifying games required.")
    parser.add_argument(
        "--min-minutes-ratio",
        type=float,
        default=0.75,
        help="Minimum minutes as a share of recent usual minutes.",
    )
    parser.add_argument(
        "--require-same-starter",
        action="store_true",
        help="Only compare the target to players with the same starter/bench flag.",
    )
    parser.add_argument(
        "--backtest-csv",
        type=Path,
        default=None,
        help="Optional path to save the full row-level backtest results as CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from src.evaluate import backtest_predictions, write_markdown_report

    df = _read_dataset(args.dataset_path)
    backtest = backtest_predictions(
        df,
        target_player_ids=args.target_player_ids,
        candidate_player_ids=args.candidate_player_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n_similar=args.top_n_similar,
        lookback_days=args.lookback_days,
        min_games=args.min_games,
        min_minutes_ratio=args.min_minutes_ratio,
        require_same_starter_flag=args.require_same_starter,
    )
    write_markdown_report(backtest, args.report_path)

    if args.backtest_csv is not None:
        args.backtest_csv.parent.mkdir(parents=True, exist_ok=True)
        backtest.to_csv(args.backtest_csv, index=False)

    print(f"Wrote report to {args.report_path}")
    if args.backtest_csv is not None:
        print(f"Wrote backtest rows to {args.backtest_csv}")
    print(f"Predictions evaluated: {len(backtest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

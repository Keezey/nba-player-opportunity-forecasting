"""Run the automatic single-game predictor over every target game in a date range.

Example:
    python -m scripts.run_live_period_backtest "Amen Thompson" 2026-03-01 2026-04-01 reports/amen_live_period.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the exact automatic player/date prediction workflow over a historical period. "
            "This resolves the player, finds each game, auto-selects similar players, and compares "
            "predictions to actual FGA, rebounds, and rebound chances."
        )
    )
    parser.add_argument("target_player", help="Player full name or NBA Stats player ID.")
    parser.add_argument("start_date", help="First game date to evaluate, YYYY-MM-DD.")
    parser.add_argument("end_date", help="Last game date to evaluate, YYYY-MM-DD.")
    parser.add_argument("report_path", type=Path, help="Where to write the Markdown report.")
    parser.add_argument(
        "--season",
        default=None,
        help="Optional NBA season override, e.g. 2025-26. Normally inferred from the date range.",
    )
    parser.add_argument(
        "--candidate-player",
        action="append",
        dest="candidate_players",
        help="Restrict automated candidates to this name/ID. May be repeated.",
    )
    parser.add_argument(
        "--manual-similar-player",
        action="append",
        dest="manual_similar_players",
        help="Use this hand-selected comparison player name/ID. May be repeated.",
    )
    parser.add_argument("--top-n-similar", type=int, default=10)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--min-minutes-ratio", type=float, default=0.75)
    parser.add_argument("--position-weight", type=float, default=0.35)
    parser.add_argument(
        "--require-same-starter",
        action="store_true",
        help="Require the same recent starter/bench role as the target.",
    )
    parser.add_argument(
        "--backtest-csv",
        type=Path,
        default=None,
        help="Optional path to save the row-level results as CSV.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached NBA responses and fetch them again.",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help=(
            "Optional historical-store directory. Defaults to "
            "data/processed/player_games."
        ),
    )
    return parser.parse_args()


def _resolve_queries(queries):
    if not queries:
        return None
    from src.live_data import resolve_player

    return [resolve_player(query)[0] for query in queries]


def main() -> int:
    args = _parse_args()

    from src.live_evaluate import run_live_period_backtest, write_live_period_report

    results = run_live_period_backtest(
        args.target_player,
        args.start_date,
        args.end_date,
        season=args.season,
        candidate_player_ids=_resolve_queries(args.candidate_players),
        manual_similar_player_ids=_resolve_queries(args.manual_similar_players),
        top_n_similar=args.top_n_similar,
        lookback_days=args.lookback_days,
        min_games=args.min_games,
        min_minutes_ratio=args.min_minutes_ratio,
        position_weight=args.position_weight,
        require_same_starter_flag=args.require_same_starter,
        refresh=args.refresh,
        store_dir=args.store_dir,
    )
    write_live_period_report(results, args.report_path)

    if args.backtest_csv is not None:
        args.backtest_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.backtest_csv, index=False)

    predicted = int((results["status"] == "predicted").sum()) if not results.empty else 0
    skipped = len(results) - predicted
    print(f"Wrote report to {args.report_path}")
    if args.backtest_csv is not None:
        print(f"Wrote backtest rows to {args.backtest_csv}")
    print(f"Games found: {len(results)}")
    print(f"Predictions evaluated: {predicted}")
    print(f"Skipped games: {skipped}")
    return 0


def cli() -> int:
    try:
        return main()
    except ModuleNotFoundError as exc:
        if exc.name in {"nba_api", "pyarrow"}:
            print(
                "Error: required packages are missing. Run "
                "`python -m pip install -r requirements.txt`.",
                file=sys.stderr,
            )
            return 2
        raise
    except Exception as exc:
        from src.fetch_nba import NBADataFetchError
        from src.predict import PredictionUnavailableError

        if isinstance(exc, (ValueError, NBADataFetchError, PredictionUnavailableError)):
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(cli())

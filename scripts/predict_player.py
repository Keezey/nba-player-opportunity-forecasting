"""Predict one supported historical regular-season player-game.

Example:
    python -m scripts.predict_player "Amen Thompson" 2026-04-07
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Look up a regular-season player-game and predict FGA, rebound chances, "
            "and rebounds using only data available before that game."
        )
    )
    parser.add_argument("target_player", help="Player full name or NBA Stats player ID.")
    parser.add_argument("game_date", help="Historical regular-season game date, YYYY-MM-DD.")
    parser.add_argument(
        "--season",
        default=None,
        help="Optional NBA season override, e.g. 2025-26. Normally inferred from the date.",
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
        "--unbounded-multipliers",
        action="store_true",
        help="Reproduce the legacy matchup calculation without safety bounds.",
    )
    parser.add_argument(
        "--similar-preview-count",
        type=int,
        default=10,
        help="Number of selected comparison profiles to display.",
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

    from src.predict import predict_player_game
    from src.trust import DEFAULT_MULTIPLIER_BOUNDS

    result = predict_player_game(
        args.target_player,
        args.game_date,
        season=args.season,
        candidate_player_ids=_resolve_queries(args.candidate_players),
        manual_similar_player_ids=_resolve_queries(args.manual_similar_players),
        top_n_similar=args.top_n_similar,
        lookback_days=args.lookback_days,
        min_games=args.min_games,
        min_minutes_ratio=args.min_minutes_ratio,
        position_weight=args.position_weight,
        require_same_starter_flag=args.require_same_starter,
        multiplier_bounds=(
            None if args.unbounded_multipliers else DEFAULT_MULTIPLIER_BOUNDS
        ),
        refresh=args.refresh,
        store_dir=args.store_dir,
    )

    projection = result["projection"]
    similar_players = result["similar_players"]
    print(f"Target: {result['target_player_name']} ({result['target_player_id']})")
    print(f"Game date: {projection['game_date'].date()}")
    print(f"Season: {projection['season']}")
    print(f"Opponent: {projection['target_opp_abbr']}")
    print(f"Data cutoff: {projection['as_of_date'].date()}")

    output_fields = [
        "minutes_pred",
        "pred_fga",
        "pred_reb_chances",
        "pred_reb",
        "baseline_fga",
        "baseline_reb_chances",
        "baseline_reb",
        "reb_conversion",
        "baseline_games_used",
        "tracking_games_used",
        "n_similar_players",
        "fga_n_samples",
        "fga_trust_weight",
        "reb_chances_n_samples",
        "reb_chances_trust_weight",
        "raw_fga_multiplier",
        "bounded_fga_multiplier",
        "fga_multiplier_was_bounded",
        "fga_multiplier",
        "raw_reb_chances_multiplier",
        "bounded_reb_chances_multiplier",
        "reb_chances_multiplier_was_bounded",
        "reb_chances_multiplier",
    ]
    print("\nProjection")
    print("----------")
    print(projection[output_fields].to_string())

    print("\nSimilar Players")
    print("---------------")
    preview_columns = [
        "player_name",
        "similarity_score",
        "minutes",
        "fga_per_min",
        "reb_chances_per_min",
        "usage_rate",
        "touches_per_min",
        "listed_position",
        "starter_rate",
    ]
    print(
        similar_players[
            [column for column in preview_columns if column in similar_players.columns]
        ]
        .head(args.similar_preview_count)
        .to_string()
    )
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

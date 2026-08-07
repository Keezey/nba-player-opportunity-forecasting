"""Predict one historical player-game with a saved V2 model bundle."""

from __future__ import annotations

import argparse

import pandas as pd

from src.v2.models import load_model_bundle
from src.v2.predict import predict_player_game_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a saved V2 model for one player-game.")
    parser.add_argument("model_path")
    parser.add_argument("player")
    parser.add_argument("game_date")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--top-n-similar", type=int)
    parser.add_argument(
        "--store-dir",
        default=None,
        help=(
            "Optional historical-store directory. Defaults to "
            "data/processed/player_games."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overrides = {"refresh": args.refresh}
    if args.top_n_similar is not None:
        overrides["top_n_similar"] = args.top_n_similar
    if args.store_dir is not None:
        overrides["store_dir"] = args.store_dir

    bundle = load_model_bundle(args.model_path)
    result = predict_player_game_v2(
        bundle,
        args.player,
        args.game_date,
        prediction_overrides=overrides,
    )
    v1 = result["v1_projection"]
    v2 = result["v2_projection"]
    print(
        f"{result['target_player_name']} vs {v1['target_opp_abbr']} "
        f"on {pd.to_datetime(args.game_date).date()}"
    )
    print(f"V1 FGA: {float(v1['pred_fga']):.2f}")
    print(f"V2 FGA: {float(v2['v2_pred_fga']):.2f}")
    print(f"V1 rebound chances: {float(v1['pred_reb_chances']):.2f}")
    print(f"V2 rebound chances: {float(v2['v2_pred_reb_chances']):.2f}")
    print(
        "Baseline rebound conversion: "
        f"{100 * float(v2['baseline_reb_conversion']):.1f}%"
    )
    print(
        "V2 rebound conversion: "
        f"{100 * float(v2['pred_reb_conversion']):.1f}%"
    )
    print(f"V1 rebounds: {float(v1['pred_reb']):.2f}")
    print(f"V2 rebounds: {float(v2['v2_pred_reb']):.2f}")
    print(f"Similar players used: {len(result['similar_players'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

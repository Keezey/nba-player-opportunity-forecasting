"""Live-style historical evaluation using the automatic prediction workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .dataset import build_player_game_df, clean_gamelog
from .fetch_nba import NBADataFetchError, fetch_player_gamelog
from .historical_store import load_season_player_games, season_store_exists
from .live_data import resolve_player, season_from_game_date
from .predict import PredictionUnavailableError, predict_player_game


LIVE_BACKTEST_COLUMNS = [
    "status",
    "player_id",
    "player_name",
    "game_id",
    "game_date",
    "opp_abbr",
    "actual_fga",
    "actual_reb",
    "actual_reb_chances",
    "pred_fga",
    "pred_reb",
    "pred_reb_chances",
    "error_fga",
    "error_reb",
    "error_reb_chances",
    "abs_error_fga",
    "abs_error_reb",
    "abs_error_reb_chances",
    "baseline_games_used",
    "tracking_games_used",
    "n_similar_players",
    "n_matchup_samples",
    "trust_weight",
    "trust_label",
    "fga_n_samples",
    "reb_chances_n_samples",
    "similar_player_ids",
    "similar_player_names",
    "message",
]


def _mae(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.abs().mean())


def _rmse(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(np.sqrt((clean**2).mean()))


def _season_for_range(start_date, end_date, season: str | None) -> str:
    start_season = season_from_game_date(start_date)
    end_season = season_from_game_date(end_date)
    if season is not None:
        return season
    if start_season != end_season:
        raise ValueError(
            "Live period backtests currently expect a date range within one NBA season. "
            f"Got {start_season} through {end_season}."
        )
    return start_season


def _target_games_in_range(
    player_id: int,
    *,
    season: str,
    start_date,
    end_date,
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    if not refresh and season_store_exists(season, store_dir):
        return load_season_player_games(
            season,
            player_ids=[int(player_id)],
            date_from=start_date,
            date_to=end_date,
            store_dir=store_dir,
        ).sort_values("game_date")

    raw = fetch_player_gamelog(
        int(player_id),
        season,
        "Regular Season",
        refresh=refresh,
    )
    games = clean_gamelog(raw)
    if games.empty:
        return games

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    return games[
        (games["game_date"].dt.normalize() >= start)
        & (games["game_date"].dt.normalize() <= end)
    ].sort_values("game_date")


def _target_actuals_in_range(
    player_id: int,
    *,
    season: str,
    start_date,
    end_date,
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    actuals = build_player_game_df(
        [int(player_id)],
        season=season,
        season_type="Regular Season",
        date_from=start_date,
        date_to=end_date,
        include_advanced=False,
        refresh=refresh,
        store_dir=store_dir,
    )
    if actuals.empty:
        return pd.DataFrame()
    return actuals.set_index("game_id", drop=False)


def _similar_names(similar_players: pd.DataFrame) -> list[str]:
    if similar_players.empty or "player_name" not in similar_players.columns:
        return []
    return [str(name) for name in similar_players["player_name"].dropna().tolist()]


def _prediction_row(
    *,
    player_id: int,
    player_name: str,
    game: pd.Series,
    actual: pd.Series,
    projection: pd.Series,
    similar_players: pd.DataFrame,
) -> dict:
    row = {
        "status": "predicted",
        "player_id": player_id,
        "player_name": player_name,
        "game_id": game["game_id"],
        "game_date": pd.to_datetime(game["game_date"]),
        "opp_abbr": game["opp_abbr"],
        "actual_fga": actual.get("fga", game.get("fga", np.nan)),
        "actual_reb": actual.get("reb", game.get("reb", np.nan)),
        "actual_reb_chances": actual.get("reb_chances", np.nan),
        "pred_fga": projection.get("pred_fga", np.nan),
        "pred_reb": projection.get("pred_reb", np.nan),
        "pred_reb_chances": projection.get("pred_reb_chances", np.nan),
        "baseline_games_used": projection.get("baseline_games_used", np.nan),
        "tracking_games_used": projection.get("tracking_games_used", np.nan),
        "n_similar_players": projection.get("n_similar_players", 0),
        "n_matchup_samples": projection.get("n_matchup_samples", 0),
        "trust_weight": projection.get("trust_weight", 0.0),
        "trust_label": projection.get("trust_label", "none"),
        "fga_n_samples": projection.get("fga_n_samples", 0),
        "reb_chances_n_samples": projection.get("reb_chances_n_samples", 0),
        "similar_player_ids": projection.get("similar_player_ids", []),
        "similar_player_names": _similar_names(similar_players),
        "message": "",
    }
    row["error_fga"] = row["pred_fga"] - row["actual_fga"]
    row["error_reb"] = row["pred_reb"] - row["actual_reb"]
    row["error_reb_chances"] = row["pred_reb_chances"] - row["actual_reb_chances"]
    row["abs_error_fga"] = abs(row["error_fga"])
    row["abs_error_reb"] = abs(row["error_reb"])
    row["abs_error_reb_chances"] = abs(row["error_reb_chances"])
    return row


def run_live_period_backtest(
    player_query: str | int,
    start_date,
    end_date,
    *,
    season: str | None = None,
    candidate_player_ids: Optional[Iterable[int]] = None,
    manual_similar_player_ids: Optional[Sequence[int]] = None,
    top_n_similar: int = 10,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
    position_weight: float = 0.35,
    require_same_starter_flag: bool = False,
    refresh: bool = False,
    store_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the exact automatic single-game workflow for every target game in a range."""
    player_id, player_name = resolve_player(player_query)
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date.")

    resolved_season = _season_for_range(start, end, season)
    target_games = _target_games_in_range(
        player_id,
        season=resolved_season,
        start_date=start,
        end_date=end,
        refresh=refresh,
        store_dir=store_dir,
    )
    if target_games.empty:
        raise ValueError(
            f"No regular-season games found for {player_name} from {start.date()} to {end.date()}."
        )

    actuals = _target_actuals_in_range(
        player_id,
        season=resolved_season,
        start_date=start,
        end_date=end,
        refresh=refresh,
        store_dir=store_dir,
    )

    rows: list[dict] = []
    for _, game in target_games.iterrows():
        game_date = pd.to_datetime(game["game_date"]).normalize()
        try:
            result = predict_player_game(
                player_id,
                game_date,
                season=resolved_season,
                candidate_player_ids=candidate_player_ids,
                manual_similar_player_ids=manual_similar_player_ids,
                top_n_similar=top_n_similar,
                lookback_days=lookback_days,
                min_games=min_games,
                min_minutes_ratio=min_minutes_ratio,
                position_weight=position_weight,
                require_same_starter_flag=require_same_starter_flag,
                refresh=refresh,
                store_dir=store_dir,
            )
            actual = actuals.loc[game["game_id"]] if game["game_id"] in actuals.index else game
            rows.append(
                _prediction_row(
                    player_id=player_id,
                    player_name=player_name,
                    game=game,
                    actual=actual,
                    projection=result["projection"],
                    similar_players=result["similar_players"],
                )
            )
        except (ValueError, NBADataFetchError, PredictionUnavailableError) as exc:
            rows.append(
                {
                    "status": "skipped",
                    "player_id": player_id,
                    "player_name": player_name,
                    "game_id": game.get("game_id", pd.NA),
                    "game_date": game_date,
                    "opp_abbr": game.get("opp_abbr", pd.NA),
                    "message": str(exc),
                }
            )

    return pd.DataFrame(rows).reindex(columns=LIVE_BACKTEST_COLUMNS)


def summarize_live_period_backtest(backtest_df: pd.DataFrame) -> pd.Series:
    predicted = backtest_df[backtest_df["status"] == "predicted"].copy()
    if predicted.empty:
        return pd.Series(
            {
                "games_found": len(backtest_df),
                "predictions_evaluated": 0,
                "skipped": int((backtest_df["status"] != "predicted").sum()) if not backtest_df.empty else 0,
                "avg_matchup_samples": np.nan,
                "avg_trust_weight": np.nan,
                "mae_fga": np.nan,
                "rmse_fga": np.nan,
                "bias_fga": np.nan,
                "mae_reb": np.nan,
                "rmse_reb": np.nan,
                "bias_reb": np.nan,
                "mae_reb_chances": np.nan,
                "rmse_reb_chances": np.nan,
                "bias_reb_chances": np.nan,
            }
        )

    return pd.Series(
        {
            "games_found": len(backtest_df),
            "predictions_evaluated": len(predicted),
            "skipped": int((backtest_df["status"] != "predicted").sum()),
            "avg_matchup_samples": pd.to_numeric(predicted["n_matchup_samples"], errors="coerce").mean(),
            "avg_trust_weight": pd.to_numeric(predicted["trust_weight"], errors="coerce").mean(),
            "mae_fga": _mae(predicted["error_fga"]),
            "rmse_fga": _rmse(predicted["error_fga"]),
            "bias_fga": pd.to_numeric(predicted["error_fga"], errors="coerce").mean(),
            "mae_reb": _mae(predicted["error_reb"]),
            "rmse_reb": _rmse(predicted["error_reb"]),
            "bias_reb": pd.to_numeric(predicted["error_reb"], errors="coerce").mean(),
            "mae_reb_chances": _mae(predicted["error_reb_chances"]),
            "rmse_reb_chances": _rmse(predicted["error_reb_chances"]),
            "bias_reb_chances": pd.to_numeric(predicted["error_reb_chances"], errors="coerce").mean(),
        }
    )


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif isinstance(value, pd.Timestamp):
                values.append(str(value.date()))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_live_period_report(
    backtest_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Live-Style Period Backtest Report",
) -> Path:
    """Write a Markdown report for the live-style period backtest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_live_period_backtest(backtest_df)

    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Games found: {int(summary['games_found'])}",
        f"- Predictions evaluated: {int(summary['predictions_evaluated'])}",
        f"- Skipped games: {int(summary['skipped'])}",
        f"- Avg matchup samples: {summary['avg_matchup_samples']:.2f}" if pd.notna(summary["avg_matchup_samples"]) else "- Avg matchup samples: n/a",
        f"- Avg trust weight: {summary['avg_trust_weight']:.2f}" if pd.notna(summary["avg_trust_weight"]) else "- Avg trust weight: n/a",
        "",
        "## Error Metrics",
        "",
        "| Stat | MAE | RMSE | Bias |",
        "| --- | ---: | ---: | ---: |",
        f"| FGA | {summary['mae_fga']:.3f} | {summary['rmse_fga']:.3f} | {summary['bias_fga']:.3f} |",
        f"| REB | {summary['mae_reb']:.3f} | {summary['rmse_reb']:.3f} | {summary['bias_reb']:.3f} |",
        f"| REB chances | {summary['mae_reb_chances']:.3f} | {summary['rmse_reb_chances']:.3f} | {summary['bias_reb_chances']:.3f} |",
        "",
    ]

    predicted = backtest_df[backtest_df["status"] == "predicted"].copy()
    if not predicted.empty:
        trust_counts = predicted["trust_label"].value_counts(dropna=False).sort_index()
        lines.extend(["## Trust Buckets", "", "| Trust label | Count |", "| --- | ---: |"])
        for label, count in trust_counts.items():
            lines.append(f"| {label} | {count} |")
        lines.append("")

        preview_cols = [
            "game_date",
            "opp_abbr",
            "actual_fga",
            "pred_fga",
            "actual_reb",
            "pred_reb",
            "actual_reb_chances",
            "pred_reb_chances",
            "n_matchup_samples",
            "trust_label",
            "similar_player_names",
        ]
        lines.extend(["## Game Results", ""])
        lines.append(_markdown_table(predicted[preview_cols]))
        lines.append("")

    skipped = backtest_df[backtest_df["status"] != "predicted"].copy()
    if not skipped.empty:
        lines.extend(["## Skipped Games", ""])
        lines.append(_markdown_table(skipped[["game_date", "opp_abbr", "message"]]))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

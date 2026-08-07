"""Backtesting and reporting utilities for the prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .predict import predict_target


PREDICTION_COLUMNS = [
    "player_id",
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
    "n_similar_players",
    "n_matchup_samples",
    "trust_weight",
    "trust_label",
    "similar_player_ids",
]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _rmse(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(np.sqrt((clean**2).mean()))


def _mae(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.abs().mean())


def backtest_predictions(
    player_game_df: pd.DataFrame,
    target_player_ids: Optional[Iterable[int]] = None,
    candidate_player_ids: Optional[Iterable[int]] = None,
    manual_similar_players: Optional[Mapping[int, Sequence[int]]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    top_n_similar: int = 10,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
    require_same_starter_flag: bool = False,
) -> pd.DataFrame:
    """Backtest predictions on historical player-game rows.

    Each target game is predicted using `as_of_date = game_date - 1 day`, so the
    actual game being tested is not included in the baseline/profile window.
    """
    if player_game_df.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    games = player_game_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    eval_rows = games.copy()
    if target_player_ids is not None:
        eval_rows = eval_rows[eval_rows["player_id"].isin(set(target_player_ids))].copy()
    if start_date is not None:
        eval_rows = eval_rows[eval_rows["game_date"] >= pd.to_datetime(start_date)].copy()
    if end_date is not None:
        eval_rows = eval_rows[eval_rows["game_date"] <= pd.to_datetime(end_date)].copy()

    rows = []
    for _, actual in eval_rows.sort_values(["game_date", "player_id"]).iterrows():
        target_player_id = int(actual["player_id"])
        game_date = pd.to_datetime(actual["game_date"])
        as_of_date = game_date - pd.Timedelta(days=1)
        manual_ids = None
        if manual_similar_players is not None:
            manual_ids = manual_similar_players.get(target_player_id)

        result = predict_target(
            games,
            target_player_id=target_player_id,
            target_opp_abbr=actual["opp_abbr"],
            as_of_date=as_of_date,
            candidate_player_ids=candidate_player_ids,
            manual_similar_player_ids=manual_ids,
            top_n_similar=top_n_similar,
            lookback_days=lookback_days,
            min_games=min_games,
            min_minutes_ratio=min_minutes_ratio,
            require_same_starter_flag=require_same_starter_flag,
        )
        projection = result["projection"]
        if projection.empty:
            continue

        pred_fga = projection.get("pred_fga", np.nan)
        pred_reb = projection.get("pred_reb", np.nan)
        pred_reb_chances = projection.get("pred_reb_chances", np.nan)

        row = {
            "player_id": target_player_id,
            "game_id": actual["game_id"],
            "game_date": game_date,
            "opp_abbr": actual["opp_abbr"],
            "actual_fga": actual.get("fga", np.nan),
            "actual_reb": actual.get("reb", np.nan),
            "actual_reb_chances": actual.get("reb_chances", np.nan),
            "pred_fga": pred_fga,
            "pred_reb": pred_reb,
            "pred_reb_chances": pred_reb_chances,
            "n_similar_players": projection.get("n_similar_players", 0),
            "n_matchup_samples": projection.get("n_matchup_samples", 0),
            "trust_weight": projection.get("trust_weight", 0.0),
            "trust_label": projection.get("trust_label", "none"),
            "similar_player_ids": projection.get("similar_player_ids", []),
        }
        row["error_fga"] = row["pred_fga"] - row["actual_fga"]
        row["error_reb"] = row["pred_reb"] - row["actual_reb"]
        row["error_reb_chances"] = row["pred_reb_chances"] - row["actual_reb_chances"]
        row["abs_error_fga"] = abs(row["error_fga"])
        row["abs_error_reb"] = abs(row["error_reb"])
        row["abs_error_reb_chances"] = abs(row["error_reb_chances"])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    return pd.DataFrame(rows)[PREDICTION_COLUMNS]


def summarize_backtest(backtest_df: pd.DataFrame) -> pd.Series:
    """Summarize backtest error metrics."""
    if backtest_df.empty:
        return pd.Series(
            {
                "n_predictions": 0,
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
            "n_predictions": len(backtest_df),
            "avg_matchup_samples": pd.to_numeric(backtest_df["n_matchup_samples"], errors="coerce").mean(),
            "avg_trust_weight": pd.to_numeric(backtest_df["trust_weight"], errors="coerce").mean(),
            "mae_fga": _mae(backtest_df["error_fga"]),
            "rmse_fga": _rmse(backtest_df["error_fga"]),
            "bias_fga": pd.to_numeric(backtest_df["error_fga"], errors="coerce").mean(),
            "mae_reb": _mae(backtest_df["error_reb"]),
            "rmse_reb": _rmse(backtest_df["error_reb"]),
            "bias_reb": pd.to_numeric(backtest_df["error_reb"], errors="coerce").mean(),
            "mae_reb_chances": _mae(backtest_df["error_reb_chances"]),
            "rmse_reb_chances": _rmse(backtest_df["error_reb_chances"]),
            "bias_reb_chances": pd.to_numeric(backtest_df["error_reb_chances"], errors="coerce").mean(),
        }
    )


def write_markdown_report(
    backtest_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Basketball Performance Predictor Backtest Report",
) -> Path:
    """Write a markdown report for a backtest run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_backtest(backtest_df)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Predictions evaluated: {int(summary['n_predictions'])}",
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
        "## Trust Buckets",
        "",
    ]

    if backtest_df.empty:
        lines.extend(["No predictions were produced.", ""])
    else:
        trust_counts = backtest_df["trust_label"].value_counts(dropna=False).sort_index()
        lines.extend(["| Trust label | Count |", "| --- | ---: |"])
        for label, count in trust_counts.items():
            lines.append(f"| {label} | {count} |")
        lines.append("")

        preview_cols = [
            "player_id",
            "game_date",
            "opp_abbr",
            "actual_fga",
            "pred_fga",
            "actual_reb",
            "pred_reb",
            "n_matchup_samples",
            "trust_label",
        ]
        lines.extend(["## Prediction Preview", ""])
        lines.append(_markdown_table(backtest_df[preview_cols].head(20)))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

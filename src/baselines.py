"""Recent target-player baseline calculations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_player_baseline(
    player_games: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
) -> pd.Series:
    """Compute FGA and rebound-opportunity baselines for one player.

    A game qualifies when its minutes are at least `min_minutes_ratio` times
    the player's mean minutes in the window. At least `min_games` qualifying
    box-score rows and tracking rows are required.
    """
    if player_games.empty:
        return pd.Series(dtype="float64")

    games = player_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    unique_ids = games["player_id"].dropna().unique()
    if len(unique_ids) != 1:
        raise ValueError("compute_player_baseline expects exactly one player_id.")

    if as_of_date is None:
        as_of_date = games["game_date"].max()
    as_of_date = pd.to_datetime(as_of_date)

    window_start = as_of_date - pd.Timedelta(days=lookback_days)
    recent = games[
        (games["game_date"] >= window_start) & (games["game_date"] <= as_of_date)
    ].copy()
    if recent.empty:
        return pd.Series(dtype="float64")

    for column in ["minutes", "fga", "reb", "reb_chances"]:
        recent[column] = pd.to_numeric(recent[column], errors="coerce")

    usual_minutes = recent["minutes"].mean()
    if pd.isna(usual_minutes) or usual_minutes <= 0:
        return pd.Series(dtype="float64")

    min_minutes = usual_minutes * min_minutes_ratio
    qualifying = recent[
        (recent["minutes"] >= min_minutes)
        & (recent["minutes"] > 0)
        & recent["fga"].notna()
        & recent["reb"].notna()
    ].copy()
    if len(qualifying) < min_games:
        return pd.Series(dtype="float64")

    tracking_games = qualifying[qualifying["reb_chances"].notna()].copy()
    if len(tracking_games) < min_games:
        return pd.Series(dtype="float64")

    baseline_minutes = qualifying["minutes"].mean()
    baseline_fga = qualifying["fga"].mean()
    baseline_reb = qualifying["reb"].mean()
    baseline_reb_chances = tracking_games["reb_chances"].mean()

    fga_per_min = (qualifying["fga"] / qualifying["minutes"]).mean()
    reb_chances_per_min = (
        tracking_games["reb_chances"] / tracking_games["minutes"]
    ).mean()

    total_reb_chances = tracking_games["reb_chances"].sum()
    reb_conversion = (
        tracking_games["reb"].sum() / total_reb_chances
        if total_reb_chances > 0
        else np.nan
    )

    return pd.Series(
        {
            "player_id": int(unique_ids[0]),
            "as_of_date": as_of_date,
            "games_used": len(qualifying),
            "tracking_games_used": len(tracking_games),
            "usual_minutes": usual_minutes,
            "min_minutes_threshold": min_minutes,
            "baseline_minutes": baseline_minutes,
            "baseline_fga": baseline_fga,
            "baseline_reb": baseline_reb,
            "baseline_reb_chances": baseline_reb_chances,
            "fga_per_min": fga_per_min,
            "reb_chances_per_min": reb_chances_per_min,
            "reb_conversion": reb_conversion,
        }
    )

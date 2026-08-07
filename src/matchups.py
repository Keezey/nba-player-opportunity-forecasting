"""Matchup ratios, sample trust, and final FGA/rebound projections."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .baselines import compute_player_baseline
from .trust import (
    DEFAULT_MULTIPLIER_BOUNDS,
    MultiplierBounds,
    bound_multiplier,
    effective_multiplier,
    sample_trust_weight,
    trust_label,
)


RATIO_COLUMNS = {
    "fga_per_min": "fga_per_min",
    "reb_chances_per_min": "reb_chances_per_min",
}


def _safe_ratio(numerator, denominator) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _metric_summary(
    values: pd.Series,
    *,
    metric: str,
    multiplier_bounds: MultiplierBounds,
) -> tuple[float, float, bool, int, float, str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n_samples = len(clean)
    raw = float(clean.mean()) if n_samples else np.nan
    lower, upper = multiplier_bounds.for_metric(metric)
    bounded = bound_multiplier(
        raw,
        lower_bound=lower,
        upper_bound=upper,
    )
    was_bounded = bool(
        np.isfinite(raw)
        and np.isfinite(bounded)
        and not np.isclose(raw, bounded)
    )
    trust = sample_trust_weight(n_samples)
    effective = effective_multiplier(
        raw,
        n_samples,
        lower_bound=lower,
        upper_bound=upper,
    )
    return (
        raw,
        bounded,
        was_bounded,
        n_samples,
        trust,
        trust_label(n_samples),
        effective,
    )


def _empty_multipliers(
    target_opp_abbr: str,
    multiplier_bounds: MultiplierBounds = DEFAULT_MULTIPLIER_BOUNDS,
) -> pd.Series:
    fga_lower, fga_upper = multiplier_bounds.for_metric("fga")
    reb_lower, reb_upper = multiplier_bounds.for_metric("reb_chances")
    return pd.Series(
        {
            "target_opp_abbr": target_opp_abbr,
            "n_samples": 0,
            "trust_weight": 0.0,
            "trust_label": "none",
            "fga_n_samples": 0,
            "fga_trust_weight": 0.0,
            "fga_trust_label": "none",
            "raw_fga_multiplier": np.nan,
            "bounded_fga_multiplier": np.nan,
            "fga_multiplier_was_bounded": False,
            "fga_multiplier_lower_bound": fga_lower,
            "fga_multiplier_upper_bound": fga_upper,
            "fga_multiplier": 1.0,
            "reb_chances_n_samples": 0,
            "reb_chances_trust_weight": 0.0,
            "reb_chances_trust_label": "none",
            "raw_reb_chances_multiplier": np.nan,
            "bounded_reb_chances_multiplier": np.nan,
            "reb_chances_multiplier_was_bounded": False,
            "reb_chances_multiplier_lower_bound": reb_lower,
            "reb_chances_multiplier_upper_bound": reb_upper,
            "reb_chances_multiplier": 1.0,
        }
    )


def compute_player_matchup_ratios(
    player_games: pd.DataFrame,
    target_opp_abbr: str,
    as_of_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
) -> pd.DataFrame:
    """Compare one player's per-minute opponent games with their baseline."""
    if player_games.empty:
        return pd.DataFrame()

    games = player_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    if as_of_date is None:
        as_of_date = games["game_date"].max()
    as_of_date = pd.to_datetime(as_of_date)

    baseline = compute_player_baseline(
        games,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        min_games=min_games,
        min_minutes_ratio=min_minutes_ratio,
    )
    if baseline.empty:
        return pd.DataFrame()

    window_start = as_of_date - pd.Timedelta(days=lookback_days)
    vs_opponent = games[
        (games["game_date"] >= window_start)
        & (games["game_date"] <= as_of_date)
        & (games["opp_abbr"].astype(str).str.upper() == target_opp_abbr.upper())
    ].copy()
    if vs_opponent.empty:
        return pd.DataFrame()

    rows = []
    for _, game in vs_opponent.iterrows():
        minutes = game.get("minutes")
        row = {
            "player_id": game["player_id"],
            "game_id": game["game_id"],
            "game_date": game["game_date"],
            "opp_abbr": game["opp_abbr"],
        }
        for per_min_stat, baseline_column in RATIO_COLUMNS.items():
            source_stat = per_min_stat.removesuffix("_per_min")
            game_rate = _safe_ratio(game.get(source_stat), minutes)
            row[f"{per_min_stat}_ratio"] = _safe_ratio(
                game_rate,
                baseline.get(baseline_column),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_matchup_multipliers(
    player_game_df: pd.DataFrame,
    similar_player_ids: Iterable[int],
    target_opp_abbr: str,
    as_of_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = (
        DEFAULT_MULTIPLIER_BOUNDS
    ),
) -> pd.Series:
    """Aggregate per-game matchup ratios with metric-specific trust."""
    bounds = MultiplierBounds.from_value(multiplier_bounds)
    if player_game_df.empty:
        return _empty_multipliers(target_opp_abbr, bounds)

    games = player_game_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    if as_of_date is None:
        as_of_date = games["game_date"].max()
    as_of_date = pd.to_datetime(as_of_date)

    ratio_frames = []
    for player_id in similar_player_ids:
        ratios = compute_player_matchup_ratios(
            games[games["player_id"] == player_id],
            target_opp_abbr=target_opp_abbr,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            min_games=min_games,
            min_minutes_ratio=min_minutes_ratio,
        )
        if not ratios.empty:
            ratio_frames.append(ratios)
    if not ratio_frames:
        return _empty_multipliers(target_opp_abbr, bounds)

    ratios = pd.concat(ratio_frames, ignore_index=True)
    (
        raw_fga,
        bounded_fga,
        fga_was_bounded,
        fga_n,
        fga_trust,
        fga_label,
        fga_effective,
    ) = _metric_summary(
        ratios["fga_per_min_ratio"],
        metric="fga",
        multiplier_bounds=bounds,
    )
    (
        raw_reb,
        bounded_reb,
        reb_was_bounded,
        reb_n,
        reb_trust,
        reb_label,
        reb_effective,
    ) = _metric_summary(
        ratios["reb_chances_per_min_ratio"],
        metric="reb_chances",
        multiplier_bounds=bounds,
    )
    conservative_n = min(fga_n, reb_n)
    fga_lower, fga_upper = bounds.for_metric("fga")
    reb_lower, reb_upper = bounds.for_metric("reb_chances")

    return pd.Series(
        {
            "target_opp_abbr": target_opp_abbr.upper(),
            "n_samples": conservative_n,
            "trust_weight": min(fga_trust, reb_trust),
            "trust_label": trust_label(conservative_n),
            "fga_n_samples": fga_n,
            "fga_trust_weight": fga_trust,
            "fga_trust_label": fga_label,
            "raw_fga_multiplier": raw_fga,
            "bounded_fga_multiplier": bounded_fga,
            "fga_multiplier_was_bounded": fga_was_bounded,
            "fga_multiplier_lower_bound": fga_lower,
            "fga_multiplier_upper_bound": fga_upper,
            "fga_multiplier": fga_effective,
            "reb_chances_n_samples": reb_n,
            "reb_chances_trust_weight": reb_trust,
            "reb_chances_trust_label": reb_label,
            "raw_reb_chances_multiplier": raw_reb,
            "bounded_reb_chances_multiplier": bounded_reb,
            "reb_chances_multiplier_was_bounded": reb_was_bounded,
            "reb_chances_multiplier_lower_bound": reb_lower,
            "reb_chances_multiplier_upper_bound": reb_upper,
            "reb_chances_multiplier": reb_effective,
        }
    )


def _weighted_rate_multiplier(
    profiles: pd.DataFrame,
    opponent_rates: pd.DataFrame,
    player_ids: Iterable[int],
    *,
    rate_column: str,
    games_column: str,
    metric: str,
    multiplier_bounds: MultiplierBounds,
) -> tuple[float, float, bool, int, float, str, float]:
    ids = list(dict.fromkeys(int(player_id) for player_id in player_ids))
    common = profiles.index.intersection(opponent_rates.index).intersection(ids)
    if common.empty:
        return np.nan, np.nan, False, 0, 0.0, "none", 1.0

    overall = pd.to_numeric(profiles.loc[common, rate_column], errors="coerce")
    opponent = pd.to_numeric(opponent_rates.loc[common, rate_column], errors="coerce")
    sample_games = pd.to_numeric(opponent_rates.loc[common, games_column], errors="coerce")
    ratios = opponent / overall
    valid = ratios.notna() & np.isfinite(ratios) & sample_games.notna() & (sample_games > 0)
    if not valid.any():
        return np.nan, np.nan, False, 0, 0.0, "none", 1.0

    weights = sample_games[valid].astype(float)
    raw = float(np.average(ratios[valid].astype(float), weights=weights))
    n_samples = int(weights.sum())
    lower, upper = multiplier_bounds.for_metric(metric)
    bounded = bound_multiplier(
        raw,
        lower_bound=lower,
        upper_bound=upper,
    )
    was_bounded = bool(
        np.isfinite(raw)
        and np.isfinite(bounded)
        and not np.isclose(raw, bounded)
    )
    trust = sample_trust_weight(n_samples)
    effective = effective_multiplier(
        raw,
        n_samples,
        lower_bound=lower,
        upper_bound=upper,
    )
    return raw, bounded, was_bounded, n_samples, trust, trust_label(n_samples), effective


def compute_profile_matchup_multipliers(
    profiles: pd.DataFrame,
    opponent_rates: pd.DataFrame,
    similar_player_ids: Iterable[int],
    target_opp_abbr: str,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = (
        DEFAULT_MULTIPLIER_BOUNDS
    ),
) -> pd.Series:
    """Build matchup multipliers from dashboard profile and opponent splits."""
    bounds = MultiplierBounds.from_value(multiplier_bounds)
    if profiles.empty or opponent_rates.empty:
        return _empty_multipliers(target_opp_abbr, bounds)

    (
        raw_fga,
        bounded_fga,
        fga_was_bounded,
        fga_n,
        fga_trust,
        fga_label,
        fga_effective,
    ) = _weighted_rate_multiplier(
        profiles,
        opponent_rates,
        similar_player_ids,
        rate_column="fga_per_min",
        games_column="fga_games",
        metric="fga",
        multiplier_bounds=bounds,
    )
    (
        raw_reb,
        bounded_reb,
        reb_was_bounded,
        reb_n,
        reb_trust,
        reb_label,
        reb_effective,
    ) = _weighted_rate_multiplier(
        profiles,
        opponent_rates,
        similar_player_ids,
        rate_column="reb_chances_per_min",
        games_column="reb_chances_games",
        metric="reb_chances",
        multiplier_bounds=bounds,
    )
    conservative_n = min(fga_n, reb_n)
    fga_lower, fga_upper = bounds.for_metric("fga")
    reb_lower, reb_upper = bounds.for_metric("reb_chances")
    return pd.Series(
        {
            "target_opp_abbr": target_opp_abbr.upper(),
            "n_samples": conservative_n,
            "trust_weight": min(fga_trust, reb_trust),
            "trust_label": trust_label(conservative_n),
            "fga_n_samples": fga_n,
            "fga_trust_weight": fga_trust,
            "fga_trust_label": fga_label,
            "raw_fga_multiplier": raw_fga,
            "bounded_fga_multiplier": bounded_fga,
            "fga_multiplier_was_bounded": fga_was_bounded,
            "fga_multiplier_lower_bound": fga_lower,
            "fga_multiplier_upper_bound": fga_upper,
            "fga_multiplier": fga_effective,
            "reb_chances_n_samples": reb_n,
            "reb_chances_trust_weight": reb_trust,
            "reb_chances_trust_label": reb_label,
            "raw_reb_chances_multiplier": raw_reb,
            "bounded_reb_chances_multiplier": bounded_reb,
            "reb_chances_multiplier_was_bounded": reb_was_bounded,
            "reb_chances_multiplier_lower_bound": reb_lower,
            "reb_chances_multiplier_upper_bound": reb_upper,
            "reb_chances_multiplier": reb_effective,
        }
    )


def apply_matchup_to_target_baseline(
    target_baseline: pd.Series,
    matchup_multipliers: pd.Series,
) -> pd.Series:
    """Apply effective matchup opportunity multipliers to a target baseline."""
    if target_baseline.empty:
        return pd.Series(dtype="float64")

    pred_fga = target_baseline["baseline_fga"] * matchup_multipliers.get("fga_multiplier", 1.0)
    pred_reb_chances = target_baseline["baseline_reb_chances"] * matchup_multipliers.get(
        "reb_chances_multiplier",
        1.0,
    )
    pred_reb = pred_reb_chances * target_baseline["reb_conversion"]

    return pd.Series(
        {
            "player_id": target_baseline.get("player_id", pd.NA),
            "as_of_date": target_baseline.get("as_of_date", pd.NaT),
            "target_opp_abbr": matchup_multipliers.get("target_opp_abbr", pd.NA),
            "minutes_pred": target_baseline["baseline_minutes"],
            "pred_fga": pred_fga,
            "pred_reb_chances": pred_reb_chances,
            "pred_reb": pred_reb,
            "baseline_fga": target_baseline["baseline_fga"],
            "baseline_reb_chances": target_baseline["baseline_reb_chances"],
            "baseline_reb": target_baseline["baseline_reb"],
            "reb_conversion": target_baseline["reb_conversion"],
            "baseline_games_used": target_baseline["games_used"],
            "tracking_games_used": target_baseline["tracking_games_used"],
            "n_matchup_samples": matchup_multipliers.get("n_samples", 0),
            "trust_weight": matchup_multipliers.get("trust_weight", 0.0),
            "trust_label": matchup_multipliers.get("trust_label", "none"),
            "fga_n_samples": matchup_multipliers.get("fga_n_samples", 0),
            "fga_trust_weight": matchup_multipliers.get("fga_trust_weight", 0.0),
            "reb_chances_n_samples": matchup_multipliers.get("reb_chances_n_samples", 0),
            "reb_chances_trust_weight": matchup_multipliers.get(
                "reb_chances_trust_weight",
                0.0,
            ),
            "raw_fga_multiplier": matchup_multipliers.get(
                "raw_fga_multiplier", np.nan
            ),
            "bounded_fga_multiplier": matchup_multipliers.get(
                "bounded_fga_multiplier", np.nan
            ),
            "fga_multiplier_was_bounded": matchup_multipliers.get(
                "fga_multiplier_was_bounded", False
            ),
            "fga_multiplier": matchup_multipliers.get("fga_multiplier", 1.0),
            "raw_reb_chances_multiplier": matchup_multipliers.get(
                "raw_reb_chances_multiplier", np.nan
            ),
            "bounded_reb_chances_multiplier": matchup_multipliers.get(
                "bounded_reb_chances_multiplier", np.nan
            ),
            "reb_chances_multiplier_was_bounded": matchup_multipliers.get(
                "reb_chances_multiplier_was_bounded", False
            ),
            "reb_chances_multiplier": matchup_multipliers.get(
                "reb_chances_multiplier", 1.0
            ),
        }
    )


def project_target_matchup(
    player_game_df: pd.DataFrame,
    target_player_id: int,
    similar_player_ids: Iterable[int],
    target_opp_abbr: str,
    as_of_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 30,
    min_games: int = 5,
    min_minutes_ratio: float = 0.75,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = (
        DEFAULT_MULTIPLIER_BOUNDS
    ),
) -> pd.Series:
    """Produce a saved-dataset projection for one target and opponent."""
    if player_game_df.empty:
        return pd.Series(dtype="float64")

    games = player_game_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    if as_of_date is None:
        as_of_date = games["game_date"].max()
    as_of_date = pd.to_datetime(as_of_date)

    baseline = compute_player_baseline(
        games[games["player_id"] == target_player_id],
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        min_games=min_games,
        min_minutes_ratio=min_minutes_ratio,
    )
    if baseline.empty:
        return pd.Series(dtype="float64")

    multipliers = compute_matchup_multipliers(
        games,
        similar_player_ids=similar_player_ids,
        target_opp_abbr=target_opp_abbr,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        min_games=min_games,
        min_minutes_ratio=min_minutes_ratio,
        multiplier_bounds=multiplier_bounds,
    )
    return apply_matchup_to_target_baseline(baseline, multipliers)

"""Chronological validation and V1-versus-V2 reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .models import V2ModelBundle, predict_v2


STAT_COLUMNS = {
    "fga": ("actual_fga", "v1_pred_fga", "v2_pred_fga"),
    "reb_chances": (
        "actual_reb_chances",
        "v1_pred_reb_chances",
        "v2_pred_reb_chances",
    ),
    "reb": ("actual_reb", "v1_pred_reb", "v2_pred_reb"),
}

FROZEN_V1_COLUMNS = {
    "fga": "frozen_v1_pred_fga",
    "reb_chances": "frozen_v1_pred_reb_chances",
    "reb": "frozen_v1_pred_reb",
}


@dataclass(frozen=True)
class V2EvaluationResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    gap_days: int = 0,
    date_column: str = "game_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on whole dates so no game date appears in both sets."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")

    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{date_column} contains missing or invalid dates.")
    unique_dates = pd.Index(dates.unique()).sort_values()
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct game dates are required.")

    test_date_count = min(
        max(1, int(np.ceil(len(unique_dates) * test_fraction))),
        len(unique_dates) - 1,
    )
    test_dates = unique_dates[-test_date_count:]
    test_start = pd.Timestamp(test_dates[0])
    train_end_exclusive = test_start - pd.Timedelta(days=max(gap_days, 0))
    train_mask = dates < train_end_exclusive
    test_mask = dates.isin(test_dates)
    train = frame.loc[train_mask].copy()
    test = frame.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("The requested chronological split produced an empty partition.")
    return train, test


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    gap_days: int = 0,
    date_column: str = "game_date",
) -> list[tuple[pd.Index, pd.Index]]:
    """Return expanding, date-based train/validation index pairs."""
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if not 0 < min_train_fraction < 1:
        raise ValueError("min_train_fraction must be between 0 and 1.")

    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{date_column} contains missing or invalid dates.")
    unique_dates = pd.Index(dates.unique()).sort_values()
    first_test_position = max(1, int(np.floor(len(unique_dates) * min_train_fraction)))
    remaining = unique_dates[first_test_position:]
    if len(remaining) < 1:
        raise ValueError("Not enough distinct dates for walk-forward validation.")

    chunks = [chunk for chunk in np.array_split(remaining, min(n_splits, len(remaining))) if len(chunk)]
    splits: list[tuple[pd.Index, pd.Index]] = []
    for test_dates in chunks:
        test_start = pd.Timestamp(test_dates[0])
        train_end_exclusive = test_start - pd.Timedelta(days=max(gap_days, 0))
        train_index = frame.index[dates < train_end_exclusive]
        test_index = frame.index[dates.isin(test_dates)]
        if len(train_index) and len(test_index):
            splits.append((train_index, test_index))
    if not splits:
        raise ValueError("No usable walk-forward splits were produced.")
    return splits


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = pd.to_numeric(actual, errors="coerce")
    predicted_values = pd.to_numeric(predicted, errors="coerce")
    valid = actual_values.notna() & predicted_values.notna()
    if not valid.any():
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    error = predicted_values[valid] - actual_values[valid]
    return {
        "n": int(valid.sum()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
    }


def player_weighted_regression_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    player_ids: pd.Series,
) -> dict[str, float | int]:
    """Give every player equal influence regardless of games available."""
    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(actual, errors="coerce"),
            "predicted": pd.to_numeric(predicted, errors="coerce"),
            "player_id": pd.to_numeric(player_ids, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return {
            "n": 0,
            "players": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }
    frame["error"] = frame["predicted"] - frame["actual"]
    per_player = frame.groupby("player_id", sort=False).agg(
        mae=("error", lambda values: values.abs().mean()),
        rmse=(
            "error",
            lambda values: float(np.sqrt(np.mean(np.square(values)))),
        ),
        bias=("error", "mean"),
    )
    return {
        "n": int(len(frame)),
        "players": int(len(per_player)),
        "mae": float(per_player["mae"].mean()),
        "rmse": float(per_player["rmse"].mean()),
        "bias": float(per_player["bias"].mean()),
    }


def regression_metrics_for_aggregation(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
    aggregation: str,
) -> dict[str, float | int]:
    """Evaluate one prediction column by games or by equal-weight players."""
    if aggregation not in {"game_weighted", "player_weighted"}:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")
    if frame.empty:
        return {
            "n": 0,
            "players": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }
    if aggregation == "player_weighted":
        return player_weighted_regression_metrics(
            frame[actual_column],
            frame[prediction_column],
            frame["player_id"],
        )
    result = regression_metrics(frame[actual_column], frame[prediction_column])
    result["players"] = int(frame["player_id"].nunique())
    return result


def evaluate_model_bundle(
    bundle: V2ModelBundle,
    test_df: pd.DataFrame,
) -> V2EvaluationResult:
    """Compare held-out V1 and V2 errors for every projected statistic."""
    predictions = predict_v2(bundle, test_df)
    summary_rows: list[dict] = []

    for stat, (actual_column, v1_column, v2_column) in STAT_COLUMNS.items():
        v1_metrics = regression_metrics(predictions[actual_column], predictions[v1_column])
        v2_metrics = regression_metrics(predictions[actual_column], predictions[v2_column])
        frozen_column = FROZEN_V1_COLUMNS[stat]
        frozen_metrics = (
            regression_metrics(predictions[actual_column], predictions[frozen_column])
            if frozen_column in predictions.columns
            else {"mae": np.nan, "rmse": np.nan, "bias": np.nan}
        )
        improvement = v1_metrics["mae"] - v2_metrics["mae"]
        improvement_pct = (
            100 * improvement / v1_metrics["mae"]
            if pd.notna(v1_metrics["mae"]) and v1_metrics["mae"] > 0
            else np.nan
        )
        summary_rows.append(
            {
                "stat": stat,
                "n": min(v1_metrics["n"], v2_metrics["n"]),
                "frozen_v1_mae": frozen_metrics["mae"],
                "v1_mae": v1_metrics["mae"],
                "v2_mae": v2_metrics["mae"],
                "mae_improvement": improvement,
                "mae_improvement_pct": improvement_pct,
                "v1_rmse": v1_metrics["rmse"],
                "v2_rmse": v2_metrics["rmse"],
                "v1_bias": v1_metrics["bias"],
                "v2_bias": v2_metrics["bias"],
            }
        )
        predictions[f"v1_error_{stat}"] = (
            predictions[v1_column] - predictions[actual_column]
        )
        predictions[f"v2_error_{stat}"] = (
            predictions[v2_column] - predictions[actual_column]
        )

    return V2EvaluationResult(
        predictions=predictions,
        summary=pd.DataFrame(summary_rows),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("n/a" if pd.isna(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_evaluation_report(
    evaluation: V2EvaluationResult,
    output_path: str | Path,
    *,
    title: str = "V2 Chronological Backtest Report",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "V2 is evaluated only on dates later than its training data.",
        "Positive MAE improvement means V2 reduced error relative to V1.",
        "",
        "## Error Comparison",
        "",
        _markdown_table(evaluation.summary),
        "",
        "## Coverage",
        "",
        f"- Held-out predictions: {len(evaluation.predictions)}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output

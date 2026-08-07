"""Validation and provenance helpers for a finalized V2 model release."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .evaluate import regression_metrics
from .features import TRAINING_ROW_COLUMNS


RELEASE_SCHEMA_VERSION = 2
RELEASE_KEY_COLUMNS = ["game_id", "player_id"]
REQUIRED_NUMERIC_COLUMNS = [
    "actual_fga",
    "actual_reb_chances",
    "actual_reb",
    "fga_residual",
    "reb_chances_residual",
    "reb_residual",
    "v1_pred_fga",
    "v1_pred_reb_chances",
    "v1_pred_reb",
    "baseline_reb_conversion",
]
TRACKED_PACKAGES = [
    "nba_api",
    "pandas",
    "numpy",
    "pyarrow",
    "scikit-learn",
    "joblib",
    "torch",
]


def file_sha256(path: str | Path) -> str:
    """Return a stable SHA-256 digest for one release input or artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Validate one leakage-safe V2 frame and return a normalized copy."""
    if frame.empty:
        raise ValueError(f"{label} is empty.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{label} contains duplicate columns: {duplicates}")

    missing = [column for column in TRAINING_ROW_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing V2 columns: {missing}")

    result = frame.copy()
    for column in ["game_date", "as_of_date"]:
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
        if result[column].isna().any():
            raise ValueError(f"{label}.{column} contains missing or invalid dates.")

    leakage = result["as_of_date"] >= result["game_date"]
    if leakage.any():
        raise ValueError(
            f"{label} contains {int(leakage.sum())} rows whose feature cutoff is "
            "not before the target game."
        )

    for column in RELEASE_KEY_COLUMNS:
        if result[column].isna().any():
            raise ValueError(f"{label}.{column} contains missing values.")
    duplicate_keys = result.duplicated(RELEASE_KEY_COLUMNS, keep=False)
    if duplicate_keys.any():
        raise ValueError(
            f"{label} contains {int(duplicate_keys.sum())} rows with duplicate "
            f"{tuple(RELEASE_KEY_COLUMNS)} keys."
        )

    for column in REQUIRED_NUMERIC_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values)
        if invalid.any():
            raise ValueError(
                f"{label}.{column} contains {int(invalid.sum())} missing or non-finite values."
            )
        result[column] = values

    conversion = result["baseline_reb_conversion"]
    outside_probability = ~conversion.between(0, 1, inclusive="both")
    if outside_probability.any():
        raise ValueError(
            f"{label}.baseline_reb_conversion contains values outside [0, 1]."
        )

    return result.sort_values(
        ["game_date", "player_id", "game_id"], kind="stable"
    ).reset_index(drop=True)


def combine_release_datasets(
    development: pd.DataFrame,
    final_holdout: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and combine strictly chronological development and holdout rows."""
    development = validate_training_frame(development, label="development dataset")
    final_holdout = validate_training_frame(final_holdout, label="final holdout dataset")

    development_columns = set(development.columns)
    holdout_columns = set(final_holdout.columns)
    if development_columns != holdout_columns:
        missing = sorted(development_columns - holdout_columns)
        extra = sorted(holdout_columns - development_columns)
        raise ValueError(
            "Release dataset schemas differ. "
            f"Missing from holdout: {missing}; extra in holdout: {extra}."
        )

    development_end = development["game_date"].max()
    holdout_start = final_holdout["game_date"].min()
    if holdout_start <= development_end:
        raise ValueError(
            "The final holdout must begin strictly after development data ends: "
            f"development ends {development_end.date()}, "
            f"holdout begins {holdout_start.date()}."
        )

    final_holdout = final_holdout.reindex(columns=development.columns)
    combined = pd.concat([development, final_holdout], ignore_index=True)
    return validate_training_frame(combined, label="combined release dataset")


def merge_training_datasets(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge compatible V2 datasets that may cover the same date range."""
    if not frames:
        raise ValueError("At least one training dataset is required.")
    validated = {
        label: validate_training_frame(frame, label=label)
        for label, frame in frames.items()
    }
    first_label = next(iter(validated))
    columns = set(validated[first_label].columns)
    for label, frame in validated.items():
        if set(frame.columns) != columns:
            raise ValueError(f"{label} does not match the {first_label} schema.")
    ordered_columns = validated[first_label].columns
    combined = pd.concat(
        [frame.reindex(columns=ordered_columns) for frame in validated.values()],
        ignore_index=True,
    )
    return validate_training_frame(combined, label="merged training datasets")


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-safe coverage metadata for one validated training frame."""
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    seasons = sorted(str(value) for value in frame["season"].dropna().unique())
    return {
        "rows": int(len(frame)),
        "games": int(frame["game_id"].nunique()),
        "players": int(frame["player_id"].nunique()),
        "columns": int(len(frame.columns)),
        "date_from": dates.min().date().isoformat(),
        "date_to": dates.max().date().isoformat(),
        "seasons": seasons,
    }


def evaluation_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    """Summarize the frozen final-test predictions without changing the model."""
    required = [
        "game_id",
        "game_date",
        "player_id",
        "actual_fga",
        "actual_reb_chances",
        "actual_reb",
        "v1_pred_fga",
        "v1_pred_reb_chances",
        "v1_pred_reb",
        "v2_pred_fga",
        "v2_pred_reb_chances",
        "v2_pred_reb",
    ]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"Evaluation predictions are missing columns: {missing}")

    result: dict[str, Any] = {
        "rows": int(len(predictions)),
        "players": int(predictions["player_id"].nunique()),
        "date_from": pd.to_datetime(predictions["game_date"]).min().date().isoformat(),
        "date_to": pd.to_datetime(predictions["game_date"]).max().date().isoformat(),
        "metrics": {},
    }
    for stat in ["fga", "reb_chances", "reb"]:
        actual = predictions[f"actual_{stat}"]
        stat_metrics: dict[str, Mapping[str, float | int]] = {
            "v1": regression_metrics(actual, predictions[f"v1_pred_{stat}"]),
            "v2": regression_metrics(actual, predictions[f"v2_pred_{stat}"]),
        }
        frozen_column = f"frozen_v1_pred_{stat}"
        if frozen_column in predictions:
            stat_metrics["frozen_v1"] = regression_metrics(
                actual, predictions[frozen_column]
            )
        result["metrics"][stat] = stat_metrics
    return result


def runtime_summary() -> dict[str, Any]:
    """Capture the runtime versions needed to reproduce a serialized model."""
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_atomic(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON release manifest atomically."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output

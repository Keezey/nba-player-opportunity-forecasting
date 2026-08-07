"""Controlled model learning curves across nested target-player pools."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .evaluate import regression_metrics_for_aggregation
from .models import fit_residual_models
from .release import validate_training_frame
from .tuning import TrustParameters, rebase_training_frame_with_trust


EXPERIMENT_STATS = ("fga", "reb_chances", "reb")
EXPERIMENT_SCOPES = ("all", "seen", "unseen")
EXPERIMENT_AGGREGATIONS = ("game_weighted", "player_weighted")


@dataclass(frozen=True)
class PoolSizeExperimentResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame


def _evaluation_rows(
    predictions: pd.DataFrame,
    *,
    pool_size: int,
    model_type: str,
    pool_player_ids: set[int],
    training_player_ids: set[int],
    training_rows: int,
    training_players: int,
    fit_predict_seconds: float,
) -> list[dict[str, Any]]:
    seen = predictions["player_id"].astype(int).isin(training_player_ids)
    scope_masks = {
        "all": pd.Series(True, index=predictions.index),
        "seen": seen,
        "unseen": ~seen,
    }
    rows: list[dict[str, Any]] = []
    for scope in EXPERIMENT_SCOPES:
        scoped = predictions.loc[scope_masks[scope]].copy()
        if scoped.empty:
            continue
        for aggregation in EXPERIMENT_AGGREGATIONS:
            for stat in EXPERIMENT_STATS:
                actual_column = f"actual_{stat}"
                v1_column = f"v1_pred_{stat}"
                v2_column = f"v2_pred_{stat}"
                v1 = regression_metrics_for_aggregation(
                    scoped,
                    actual_column=actual_column,
                    prediction_column=v1_column,
                    aggregation=aggregation,
                )
                v2 = regression_metrics_for_aggregation(
                    scoped,
                    actual_column=actual_column,
                    prediction_column=v2_column,
                    aggregation=aggregation,
                )
                frozen_column = f"frozen_v1_pred_{stat}"
                frozen = (
                    regression_metrics_for_aggregation(
                        scoped,
                        actual_column=actual_column,
                        prediction_column=frozen_column,
                        aggregation=aggregation,
                    )
                    if frozen_column in scoped
                    else {"mae": np.nan}
                )
                improvement = float(v1["mae"] - v2["mae"])
                improvement_pct = (
                    100 * improvement / float(v1["mae"])
                    if np.isfinite(v1["mae"]) and float(v1["mae"]) > 0
                    else np.nan
                )
                rows.append(
                    {
                        "pool_size": int(pool_size),
                        "model_type": model_type,
                        "scope": scope,
                        "aggregation": aggregation,
                        "stat": stat,
                        "pool_players": int(len(pool_player_ids)),
                        "training_rows": int(training_rows),
                        "training_players": int(training_players),
                        "fit_predict_seconds": float(fit_predict_seconds),
                        "test_rows": int(v2["n"]),
                        "test_players": int(v2["players"]),
                        "frozen_v1_mae": float(frozen["mae"]),
                        "v1_mae": float(v1["mae"]),
                        "v2_mae": float(v2["mae"]),
                        "mae_improvement": improvement,
                        "mae_improvement_pct": improvement_pct,
                        "v1_rmse": float(v1["rmse"]),
                        "v2_rmse": float(v2["rmse"]),
                        "v1_bias": float(v1["bias"]),
                        "v2_bias": float(v2["bias"]),
                    }
                )
    return rows


def _validate_nested_pool_ids(
    pools: Mapping[int, Iterable[int]],
) -> dict[int, set[int]]:
    resolved: dict[int, set[int]] = {}
    prior: set[int] = set()
    for size in sorted(pools):
        ids = {int(player_id) for player_id in pools[size]}
        if len(ids) != int(size):
            raise ValueError(f"Pool {size} contains {len(ids)} unique player IDs.")
        if not prior.issubset(ids):
            raise ValueError(f"Pool {size} is not nested over the preceding pool.")
        resolved[int(size)] = ids
        prior = ids
    return resolved


def run_pool_size_experiment(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    pools: Mapping[int, Iterable[int]],
    model_parameters: Mapping[str, Mapping[str, Mapping[str, Any]]],
    trust_parameters: TrustParameters,
    min_rows: int = 30,
    random_state: int = 42,
    progress: Callable[[str], None] | None = None,
) -> PoolSizeExperimentResult:
    """Fit every fixed model at every nested pool size on one fixed test set."""
    development = validate_training_frame(
        development_df, label="pool experiment development dataset"
    )
    test = validate_training_frame(test_df, label="pool experiment test dataset")
    if set(development.columns) != set(test.columns):
        raise ValueError("Development and test schemas differ.")
    test = test.reindex(columns=development.columns)
    if test["game_date"].min() <= development["game_date"].max():
        raise ValueError("Pool experiment test dates must follow all development dates.")

    nested_pools = _validate_nested_pool_ids(pools)
    rebased_test = rebase_training_frame_with_trust(test, trust_parameters)

    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for pool_size, pool_ids in nested_pools.items():
        training = development[development["player_id"].astype(int).isin(pool_ids)].copy()
        training_player_ids = set(training["player_id"].astype(int))
        rebased_training = rebase_training_frame_with_trust(
            training, trust_parameters
        )
        for model_type, parameters in model_parameters.items():
            if progress:
                progress(
                    f"Pool {pool_size}, {model_type}: fitting {len(training)} rows "
                    f"from {training['player_id'].nunique()} players"
                )
            started_at = perf_counter()
            bundle = fit_residual_models(
                rebased_training,
                model_type=model_type,
                parameters_by_target=parameters,
                min_rows=min_rows,
                random_state=random_state,
            )
            predictions = bundle.predict(rebased_test)
            fit_predict_seconds = perf_counter() - started_at
            if progress:
                progress(
                    f"Pool {pool_size}, {model_type}: completed in "
                    f"{fit_predict_seconds:.1f} seconds"
                )
            predictions.insert(0, "model_type", model_type)
            predictions.insert(0, "pool_size", int(pool_size))
            predictions["target_in_training_pool"] = predictions[
                "player_id"
            ].astype(int).isin(pool_ids)
            predictions["target_seen_in_training"] = predictions[
                "player_id"
            ].astype(int).isin(training_player_ids)
            prediction_frames.append(predictions)
            summaries.extend(
                _evaluation_rows(
                    predictions,
                    pool_size=pool_size,
                    model_type=model_type,
                    pool_player_ids=pool_ids,
                    training_player_ids=training_player_ids,
                    training_rows=len(training),
                    training_players=training["player_id"].nunique(),
                    fit_predict_seconds=fit_predict_seconds,
                )
            )

    return PoolSizeExperimentResult(
        summary=pd.DataFrame(summaries),
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )

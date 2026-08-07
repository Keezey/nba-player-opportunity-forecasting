"""Leakage-safe rolling outer-season evaluation for V2 model families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..trust import MultiplierBounds
from .evaluate import regression_metrics_for_aggregation
from .models import TARGET_SPECS, fit_residual_models
from .release import validate_training_frame
from .tuning import (
    MODEL_PARAMETER_SPACES,
    rebase_feature_frame_with_trust,
    rebase_training_frame_with_trust,
    tune_multitask_residual_models,
    tune_residual_model,
    tune_trust_parameters,
)


ROLLING_STATS = ("fga", "reb_chances", "reb")
ROLLING_SCOPES = ("all", "seen", "unseen")
ROLLING_AGGREGATIONS = ("game_weighted", "player_weighted")
ROLLING_MODEL_TYPES = ("elastic_net", "pytorch_multitask")


@dataclass(frozen=True)
class RollingSeasonFold:
    """One expanding training window and its untouched test season."""

    test_season: str
    training_seasons: tuple[str, ...]
    training: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class RollingSeasonExperimentResult:
    """All metrics, predictions, tuning choices, and trial tables."""

    summary: pd.DataFrame
    predictions: pd.DataFrame
    parameters: dict[str, Any]
    trials: dict[str, pd.DataFrame]


def _ordered_seasons(frame: pd.DataFrame) -> list[str]:
    season_starts = (
        frame.assign(season=frame["season"].astype(str))
        .groupby("season", as_index=False)["game_date"]
        .min()
        .sort_values(["game_date", "season"], kind="stable")
    )
    return season_starts["season"].tolist()


def build_rolling_season_folds(
    frame: pd.DataFrame,
    *,
    test_seasons: Sequence[str],
    min_training_seasons: int = 2,
) -> list[RollingSeasonFold]:
    """Build expanding folds whose training rows strictly precede each test season."""
    if min_training_seasons < 1:
        raise ValueError("min_training_seasons must be at least 1.")
    if not test_seasons:
        raise ValueError("At least one test season is required.")

    data = validate_training_frame(frame, label="rolling-season dataset")
    data["season"] = data["season"].astype(str)
    available = set(data["season"].unique())
    requested = [str(season) for season in test_seasons]
    duplicates = sorted(
        season for season in set(requested) if requested.count(season) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate test seasons requested: {duplicates}")
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(
            f"Test seasons are absent from the dataset: {missing}. "
            f"Available seasons: {_ordered_seasons(data)}"
        )

    folds: list[RollingSeasonFold] = []
    for test_season in requested:
        test = data[data["season"] == test_season].copy()
        test_start = test["game_date"].min()
        training = data[data["game_date"] < test_start].copy()
        training_seasons = (
            tuple(_ordered_seasons(training)) if not training.empty else ()
        )
        if len(training_seasons) < min_training_seasons:
            raise ValueError(
                f"Test season {test_season} has only {len(training_seasons)} prior "
                f"training seasons; at least {min_training_seasons} are required."
            )
        if training["game_date"].max() >= test_start:
            raise ValueError(
                f"Training dates for {test_season} overlap its test window."
            )
        folds.append(
            RollingSeasonFold(
                test_season=test_season,
                training_seasons=training_seasons,
                training=training.reset_index(drop=True),
                test=test.reset_index(drop=True),
            )
        )

    return sorted(folds, key=lambda fold: fold.test["game_date"].min())


def _evaluation_rows(
    predictions: pd.DataFrame,
    *,
    fold: RollingSeasonFold,
    model_type: str,
    trust_tuning_seconds: float,
    model_tuning_seconds: float,
    fit_predict_seconds: float,
) -> list[dict[str, Any]]:
    training_player_ids = set(fold.training["player_id"].astype(int))
    seen = predictions["player_id"].astype(int).isin(training_player_ids)
    scope_masks = {
        "all": pd.Series(True, index=predictions.index),
        "seen": seen,
        "unseen": ~seen,
    }
    rows: list[dict[str, Any]] = []
    for scope in ROLLING_SCOPES:
        scoped = predictions.loc[scope_masks[scope]].copy()
        if scoped.empty:
            continue
        for aggregation in ROLLING_AGGREGATIONS:
            for stat in ROLLING_STATS:
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
                        "test_season": fold.test_season,
                        "training_seasons": ",".join(fold.training_seasons),
                        "model_type": model_type,
                        "scope": scope,
                        "aggregation": aggregation,
                        "stat": stat,
                        "training_rows": int(len(fold.training)),
                        "training_players": int(
                            fold.training["player_id"].nunique()
                        ),
                        "test_rows": int(v2["n"]),
                        "test_players": int(v2["players"]),
                        "trust_tuning_seconds": float(trust_tuning_seconds),
                        "model_tuning_seconds": float(model_tuning_seconds),
                        "fit_predict_seconds": float(fit_predict_seconds),
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


def tune_model_parameters(
    training: pd.DataFrame,
    *,
    model_type: str,
    n_iter: int,
    n_splits: int,
    min_rows: int,
    random_state: int,
    parameter_space: Mapping[str, Sequence[Any]] | None,
    neural_tuning_max_epochs: int,
    neural_tuning_patience: int,
    neural_final_ensemble_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, pd.DataFrame]]:
    """Tune one model family and return fit parameters plus trial details."""
    if model_type == "pytorch_multitask":
        results = tune_multitask_residual_models(
            training,
            parameter_space=parameter_space,
            n_iter=n_iter,
            n_splits=n_splits,
            min_rows=min_rows,
            random_state=random_state,
            tuning_max_epochs=neural_tuning_max_epochs,
            tuning_patience=neural_tuning_patience,
            final_ensemble_size=neural_final_ensemble_size,
        )
        shared = dict(results["fga"].best_parameters)
        summary = {
            "shared": shared,
            "inner_validation": {
                target: {
                    "v1_mae": result.mean_v1_mae,
                    "v2_mae": result.best_v2_mae,
                }
                for target, result in results.items()
            },
        }
        return (
            {"shared": shared},
            summary,
            {"multitask": results["fga"].trials},
        )

    results = {
        target: tune_residual_model(
            training,
            target,
            model_type=model_type,
            parameter_space=parameter_space,
            n_iter=n_iter,
            n_splits=n_splits,
            min_rows=min_rows,
            random_state=random_state,
        )
        for target in TARGET_SPECS
    }
    parameters = {
        target: dict(result.best_parameters) for target, result in results.items()
    }
    summary = {
        "targets": {
            target: {
                "parameters": parameters[target],
                "v1_mae": result.mean_v1_mae,
                "v2_mae": result.best_v2_mae,
            }
            for target, result in results.items()
        }
    }
    return parameters, summary, {
        target: result.trials for target, result in results.items()
    }


def run_rolling_season_experiment(
    frame: pd.DataFrame,
    *,
    test_seasons: Sequence[str],
    model_types: Sequence[str] = ROLLING_MODEL_TYPES,
    trust_n_iter: int = 20,
    model_n_iter: Mapping[str, int] | None = None,
    n_splits: int = 3,
    min_training_seasons: int = 2,
    min_rows: int = 30,
    random_state: int = 42,
    parameter_spaces: Mapping[
        str, Mapping[str, Sequence[Any]]
    ] | None = None,
    neural_tuning_max_epochs: int = 80,
    neural_tuning_patience: int = 10,
    neural_final_ensemble_size: int = 5,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RollingSeasonExperimentResult:
    """Retune and score each model inside expanding, untouched season folds."""
    requested_models = list(dict.fromkeys(str(value) for value in model_types))
    unknown = sorted(set(requested_models) - set(ROLLING_MODEL_TYPES))
    if unknown:
        raise ValueError(
            f"Unsupported rolling-experiment models: {unknown}. "
            f"Choose from {list(ROLLING_MODEL_TYPES)}."
        )
    if not requested_models:
        raise ValueError("At least one model type is required.")
    if trust_n_iter < 1:
        raise ValueError("trust_n_iter must be at least 1.")

    iterations = {
        "elastic_net": 20,
        "pytorch_multitask": 6,
        **dict(model_n_iter or {}),
    }
    spaces = dict(parameter_spaces or {})
    folds = build_rolling_season_folds(
        frame,
        test_seasons=test_seasons,
        min_training_seasons=min_training_seasons,
    )

    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    parameters: dict[str, Any] = {}
    trials: dict[str, pd.DataFrame] = {}
    for fold_number, fold in enumerate(folds):
        if progress:
            progress(
                f"{fold.test_season}: tuning trust on {len(fold.training)} rows "
                f"from {', '.join(fold.training_seasons)}"
            )
        trust_started = perf_counter()
        trust_result = tune_trust_parameters(
            fold.training,
            n_iter=trust_n_iter,
            n_splits=n_splits,
            random_state=random_state + fold_number,
            multiplier_bounds=multiplier_bounds,
        )
        trust_seconds = perf_counter() - trust_started
        trust = trust_result.best_parameters
        rebased_training = rebase_training_frame_with_trust(
            fold.training,
            trust,
            multiplier_bounds,
        )
        rebased_test = rebase_feature_frame_with_trust(
            fold.test,
            trust,
            multiplier_bounds,
        )
        trials[f"{fold.test_season}/trust"] = trust_result.trials
        parameters[fold.test_season] = {
            "training_seasons": list(fold.training_seasons),
            "training_rows": int(len(fold.training)),
            "training_players": int(fold.training["player_id"].nunique()),
            "test_rows": int(len(fold.test)),
            "test_players": int(fold.test["player_id"].nunique()),
            "trust": {
                "parameters": asdict(trust),
                "objective": trust_result.best_objective,
                "tuning_seconds": trust_seconds,
            },
            "multiplier_bounds": MultiplierBounds.from_value(
                multiplier_bounds
            ).to_dict(),
            "models": {},
        }

        for model_number, model_type in enumerate(requested_models):
            n_iter = int(iterations[model_type])
            if n_iter < 1:
                raise ValueError(f"n_iter for {model_type} must be at least 1.")
            if progress:
                progress(
                    f"{fold.test_season}, {model_type}: tuning {n_iter} candidates"
                )
            tuning_started = perf_counter()
            fit_parameters, tuning_summary, model_trials = tune_model_parameters(
                rebased_training,
                model_type=model_type,
                n_iter=n_iter,
                n_splits=n_splits,
                min_rows=min_rows,
                random_state=random_state + fold_number * 100 + model_number,
                parameter_space=spaces.get(
                    model_type, MODEL_PARAMETER_SPACES[model_type]
                ),
                neural_tuning_max_epochs=neural_tuning_max_epochs,
                neural_tuning_patience=neural_tuning_patience,
                neural_final_ensemble_size=neural_final_ensemble_size,
            )
            model_tuning_seconds = perf_counter() - tuning_started
            for trial_name, trial_frame in model_trials.items():
                trials[
                    f"{fold.test_season}/{model_type}/{trial_name}"
                ] = trial_frame

            if progress:
                progress(f"{fold.test_season}, {model_type}: fitting final model")
            fit_started = perf_counter()
            bundle = fit_residual_models(
                rebased_training,
                model_type=model_type,
                parameters_by_target=fit_parameters,
                min_rows=min_rows,
                random_state=random_state + fold_number * 100 + model_number,
            )
            predictions = bundle.predict(rebased_test)
            fit_predict_seconds = perf_counter() - fit_started
            predictions.insert(0, "model_type", model_type)
            predictions.insert(0, "test_season", fold.test_season)
            training_ids = set(fold.training["player_id"].astype(int))
            predictions["target_seen_in_training"] = predictions[
                "player_id"
            ].astype(int).isin(training_ids)
            predictions["training_seasons"] = ",".join(fold.training_seasons)
            prediction_frames.append(predictions)
            summary_rows.extend(
                _evaluation_rows(
                    predictions,
                    fold=fold,
                    model_type=model_type,
                    trust_tuning_seconds=trust_seconds,
                    model_tuning_seconds=model_tuning_seconds,
                    fit_predict_seconds=fit_predict_seconds,
                )
            )
            parameters[fold.test_season]["models"][model_type] = {
                **tuning_summary,
                "tuning_seconds": model_tuning_seconds,
                "fit_predict_seconds": fit_predict_seconds,
            }
            if progress:
                progress(
                    f"{fold.test_season}, {model_type}: scored {len(predictions)} "
                    f"rows in {fit_predict_seconds:.1f} seconds"
                )

    return RollingSeasonExperimentResult(
        summary=pd.DataFrame(summary_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        parameters=parameters,
        trials=trials,
    )

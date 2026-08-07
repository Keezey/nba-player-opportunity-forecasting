"""Conditional evaluation when actual minutes stay near pregame expectations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..trust import MultiplierBounds
from .evaluate import regression_metrics_for_aggregation
from .models import fit_residual_models
from .rolling_experiment import (
    ROLLING_MODEL_TYPES,
    ROLLING_STATS,
    build_rolling_season_folds,
    tune_model_parameters,
)
from .tuning import (
    MODEL_PARAMETER_SPACES,
    rebase_feature_frame_with_trust,
    rebase_training_frame_with_trust,
    tune_trust_parameters,
)


TRAINING_COHORTS = ("all", "regular")
EVALUATION_COHORTS = ("all", "regular", "irregular")
AGGREGATIONS = ("game_weighted", "player_weighted")


@dataclass(frozen=True)
class RegularMinutesExperimentResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    parameters: dict[str, Any]
    trials: dict[str, pd.DataFrame]


def annotate_regular_minutes(
    frame: pd.DataFrame,
    *,
    absolute_tolerance: float = 3.0,
    relative_tolerance: float = 0.15,
) -> pd.DataFrame:
    """Mark games whose actual minutes landed near the pregame V1 expectation."""
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance cannot be negative.")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance cannot be negative.")
    required = ["actual_minutes", "v1_pred_minutes"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Minutes cohort requires columns: {missing}")

    result = frame.copy()
    actual = pd.to_numeric(result["actual_minutes"], errors="coerce")
    expected = pd.to_numeric(result["v1_pred_minutes"], errors="coerce")
    tolerance = np.maximum(
        float(absolute_tolerance),
        float(relative_tolerance) * expected,
    )
    minute_difference = actual - expected
    valid = (
        actual.notna()
        & expected.notna()
        & np.isfinite(actual)
        & np.isfinite(expected)
        & (actual >= 0)
        & (expected > 0)
    )
    result["minute_difference"] = minute_difference
    result["absolute_minute_error"] = minute_difference.abs()
    result["regular_minutes_tolerance"] = tolerance
    result["regular_minutes_flag"] = (
        valid & (minute_difference.abs() <= tolerance)
    )
    return result


def _evaluation_rows(
    predictions: pd.DataFrame,
    *,
    test_season: str,
    training_cohort: str,
    model_type: str,
    full_training_rows: int,
    training_rows: int,
    full_test_rows: int,
    trust_tuning_seconds: float,
    model_tuning_seconds: float,
    fit_predict_seconds: float,
) -> list[dict[str, Any]]:
    regular = predictions["regular_minutes_flag"].astype(bool)
    cohort_masks = {
        "all": pd.Series(True, index=predictions.index),
        "regular": regular,
        "irregular": ~regular,
    }
    rows: list[dict[str, Any]] = []
    for evaluation_cohort in EVALUATION_COHORTS:
        scoped = predictions.loc[cohort_masks[evaluation_cohort]].copy()
        if scoped.empty:
            continue
        for aggregation in AGGREGATIONS:
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
                frozen = regression_metrics_for_aggregation(
                    scoped,
                    actual_column=actual_column,
                    prediction_column=frozen_column,
                    aggregation=aggregation,
                )
                improvement = float(v1["mae"] - v2["mae"])
                improvement_pct = (
                    100 * improvement / float(v1["mae"])
                    if np.isfinite(v1["mae"]) and float(v1["mae"]) > 0
                    else np.nan
                )
                rows.append(
                    {
                        "test_season": test_season,
                        "training_cohort": training_cohort,
                        "evaluation_cohort": evaluation_cohort,
                        "model_type": model_type,
                        "aggregation": aggregation,
                        "stat": stat,
                        "full_training_rows": int(full_training_rows),
                        "training_rows": int(training_rows),
                        "training_coverage_pct": float(
                            100 * training_rows / full_training_rows
                        ),
                        "full_test_rows": int(full_test_rows),
                        "test_rows": int(v2["n"]),
                        "test_players": int(v2["players"]),
                        "evaluation_coverage_pct": float(
                            100 * len(scoped) / full_test_rows
                        ),
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


def run_regular_minutes_experiment(
    frame: pd.DataFrame,
    *,
    test_seasons: Sequence[str],
    model_types: Sequence[str] = ("elastic_net",),
    training_cohorts: Sequence[str] = TRAINING_COHORTS,
    absolute_tolerance: float = 3.0,
    relative_tolerance: float = 0.15,
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
) -> RegularMinutesExperimentResult:
    """Compare all-game and regular-minute training on future-season cohorts."""
    requested_models = list(dict.fromkeys(str(value) for value in model_types))
    unknown_models = sorted(set(requested_models) - set(ROLLING_MODEL_TYPES))
    if unknown_models:
        raise ValueError(f"Unsupported model types: {unknown_models}")
    requested_training = list(
        dict.fromkeys(str(value) for value in training_cohorts)
    )
    unknown_cohorts = sorted(set(requested_training) - set(TRAINING_COHORTS))
    if unknown_cohorts:
        raise ValueError(f"Unsupported training cohorts: {unknown_cohorts}")
    if not requested_models or not requested_training:
        raise ValueError("At least one model and training cohort are required.")

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
        training_all = annotate_regular_minutes(
            fold.training,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        test_all = annotate_regular_minutes(
            fold.test,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        parameters[fold.test_season] = {
            "training_seasons": list(fold.training_seasons),
            "full_training_rows": int(len(training_all)),
            "regular_training_rows": int(training_all["regular_minutes_flag"].sum()),
            "full_test_rows": int(len(test_all)),
            "regular_test_rows": int(test_all["regular_minutes_flag"].sum()),
            "training_cohorts": {},
        }

        for training_cohort in requested_training:
            training = (
                training_all
                if training_cohort == "all"
                else training_all[training_all["regular_minutes_flag"]].copy()
            )
            if progress:
                progress(
                    f"{fold.test_season}, {training_cohort} training: "
                    f"{len(training)}/{len(training_all)} rows"
                )
            trust_started = perf_counter()
            trust_result = tune_trust_parameters(
                training,
                n_iter=trust_n_iter,
                n_splits=n_splits,
                random_state=random_state + fold_number,
                multiplier_bounds=multiplier_bounds,
            )
            trust_seconds = perf_counter() - trust_started
            trust = trust_result.best_parameters
            rebased_training = rebase_training_frame_with_trust(
                training,
                trust,
                multiplier_bounds,
            )
            rebased_test = rebase_feature_frame_with_trust(
                test_all,
                trust,
                multiplier_bounds,
            )
            trials[f"{fold.test_season}/{training_cohort}/trust"] = (
                trust_result.trials
            )
            parameters[fold.test_season]["training_cohorts"][training_cohort] = {
                "training_rows": int(len(training)),
                "training_coverage_pct": float(
                    100 * len(training) / len(training_all)
                ),
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
                if progress:
                    progress(
                        f"{fold.test_season}, {training_cohort}, {model_type}: "
                        f"tuning {n_iter} candidates"
                    )
                tuning_started = perf_counter()
                fit_parameters, tuning_summary, model_trials = (
                    tune_model_parameters(
                        rebased_training,
                        model_type=model_type,
                        n_iter=n_iter,
                        n_splits=n_splits,
                        min_rows=min_rows,
                        random_state=(
                            random_state + fold_number * 100 + model_number
                        ),
                        parameter_space=spaces.get(
                            model_type, MODEL_PARAMETER_SPACES[model_type]
                        ),
                        neural_tuning_max_epochs=neural_tuning_max_epochs,
                        neural_tuning_patience=neural_tuning_patience,
                        neural_final_ensemble_size=neural_final_ensemble_size,
                    )
                )
                model_tuning_seconds = perf_counter() - tuning_started
                for trial_name, trial_frame in model_trials.items():
                    trials[
                        f"{fold.test_season}/{training_cohort}/"
                        f"{model_type}/{trial_name}"
                    ] = trial_frame

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
                predictions.insert(0, "training_cohort", training_cohort)
                predictions.insert(0, "model_type", model_type)
                predictions.insert(0, "test_season", fold.test_season)
                for column in [
                    "actual_minutes",
                    "v1_pred_minutes",
                    "minute_difference",
                    "absolute_minute_error",
                    "regular_minutes_tolerance",
                    "regular_minutes_flag",
                ]:
                    predictions[column] = test_all[column].to_numpy()
                prediction_frames.append(predictions)
                summary_rows.extend(
                    _evaluation_rows(
                        predictions,
                        test_season=fold.test_season,
                        training_cohort=training_cohort,
                        model_type=model_type,
                        full_training_rows=len(training_all),
                        training_rows=len(training),
                        full_test_rows=len(test_all),
                        trust_tuning_seconds=trust_seconds,
                        model_tuning_seconds=model_tuning_seconds,
                        fit_predict_seconds=fit_predict_seconds,
                    )
                )
                parameters[fold.test_season]["training_cohorts"][
                    training_cohort
                ]["models"][model_type] = {
                    **tuning_summary,
                    "tuning_seconds": model_tuning_seconds,
                    "fit_predict_seconds": fit_predict_seconds,
                }
                if progress:
                    progress(
                        f"{fold.test_season}, {training_cohort}, {model_type}: "
                        f"scored {len(predictions)} rows"
                    )

    return RegularMinutesExperimentResult(
        summary=pd.DataFrame(summary_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        parameters=parameters,
        trials=trials,
    )

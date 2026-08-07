"""Rolling evaluation of robust matchup-multiplier bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..trust import MultiplierBounds
from .evaluate import regression_metrics_for_aggregation
from .models import fit_residual_models
from .regular_minutes_experiment import annotate_regular_minutes
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


BOUND_STRATEGIES = (
    "unbounded",
    "fixed_0.50_1.50",
    "fixed_0.67_1.50",
    "training_p01_p99",
)
TRAINING_COHORTS = ("all", "regular")
EVALUATION_COHORTS = ("all", "regular", "irregular")
AGGREGATIONS = ("game_weighted", "player_weighted")


@dataclass(frozen=True)
class MultiplierBoundsExperimentResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    parameters: dict[str, Any]
    trials: dict[str, pd.DataFrame]


def derive_multiplier_bounds(
    training: pd.DataFrame,
    strategy: str,
) -> MultiplierBounds:
    """Resolve fixed or training-only bounds for one outer fold."""
    if strategy == "unbounded":
        return MultiplierBounds()
    if strategy == "fixed_0.50_1.50":
        return MultiplierBounds.fixed(0.50, 1.50)
    if strategy == "fixed_0.67_1.50":
        return MultiplierBounds.fixed(0.67, 1.50)
    if strategy != "training_p01_p99":
        raise ValueError(
            f"Unknown bound strategy {strategy!r}. Choose from {BOUND_STRATEGIES}."
        )

    resolved: dict[str, float] = {}
    columns = {
        "fga": "raw_fga_multiplier",
        "reb_chances": "raw_reb_chances_multiplier",
    }
    for metric, column in columns.items():
        values = pd.to_numeric(training[column], errors="coerce")
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) < 10:
            raise ValueError(
                f"Cannot derive {metric} percentile bounds from only {len(values)} rows."
            )
        lower = min(float(values.quantile(0.01)), 1.0)
        upper = max(float(values.quantile(0.99)), 1.0)
        resolved[f"{metric}_lower"] = lower
        resolved[f"{metric}_upper"] = upper
    return MultiplierBounds(**resolved)


def _tail_metrics(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
    aggregation: str,
) -> dict[str, float]:
    actual = pd.to_numeric(frame[actual_column], errors="coerce")
    predicted = pd.to_numeric(frame[prediction_column], errors="coerce")
    valid = actual.notna() & predicted.notna()
    if not valid.any():
        return {"p95": np.nan, "p99": np.nan, "max": np.nan}

    errors = (predicted[valid] - actual[valid]).abs()
    if aggregation == "player_weighted":
        values = pd.DataFrame(
            {
                "player_id": frame.loc[valid, "player_id"].to_numpy(),
                "absolute_error": errors.to_numpy(),
            }
        ).groupby("player_id")["absolute_error"].mean()
    else:
        values = errors
    return {
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _clipped_rows(
    frame: pd.DataFrame,
    *,
    metric: str,
) -> tuple[int, float]:
    flag_column = f"{metric}_multiplier_was_bounded"
    if flag_column not in frame:
        return 0, 0.0
    flags = frame[flag_column].fillna(False).astype(bool)
    count = int(flags.sum())
    return count, float(100 * count / len(frame)) if len(frame) else 0.0


def _evaluation_rows(
    predictions: pd.DataFrame,
    *,
    test_season: str,
    training_cohort: str,
    bound_strategy: str,
    model_type: str,
    bounds: MultiplierBounds,
    full_training_rows: int,
    training_rows: int,
    full_test_rows: int,
    trust_tuning_seconds: float,
    model_tuning_seconds: float,
    fit_predict_seconds: float,
) -> list[dict[str, Any]]:
    regular = predictions["regular_minutes_flag"].astype(bool)
    masks = {
        "all": pd.Series(True, index=predictions.index),
        "regular": regular,
        "irregular": ~regular,
    }
    rows: list[dict[str, Any]] = []
    for evaluation_cohort in EVALUATION_COHORTS:
        scoped = predictions.loc[masks[evaluation_cohort]].copy()
        if scoped.empty:
            continue
        for aggregation in AGGREGATIONS:
            for stat in ROLLING_STATS:
                actual_column = f"actual_{stat}"
                v1_column = f"v1_pred_{stat}"
                v2_column = f"v2_pred_{stat}"
                frozen_column = f"frozen_v1_pred_{stat}"
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
                frozen = regression_metrics_for_aggregation(
                    scoped,
                    actual_column=actual_column,
                    prediction_column=frozen_column,
                    aggregation=aggregation,
                )
                v1_tail = _tail_metrics(
                    scoped,
                    actual_column=actual_column,
                    prediction_column=v1_column,
                    aggregation=aggregation,
                )
                v2_tail = _tail_metrics(
                    scoped,
                    actual_column=actual_column,
                    prediction_column=v2_column,
                    aggregation=aggregation,
                )
                bound_metric = "fga" if stat == "fga" else "reb_chances"
                clipped_count, clipped_pct = _clipped_rows(
                    scoped,
                    metric=bound_metric,
                )
                lower, upper = bounds.for_metric(bound_metric)
                improvement = float(v1["mae"] - v2["mae"])
                rows.append(
                    {
                        "test_season": test_season,
                        "training_cohort": training_cohort,
                        "evaluation_cohort": evaluation_cohort,
                        "bound_strategy": bound_strategy,
                        "model_type": model_type,
                        "aggregation": aggregation,
                        "stat": stat,
                        "multiplier_lower_bound": lower,
                        "multiplier_upper_bound": upper,
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
                        "bounded_test_rows": clipped_count,
                        "bounded_test_pct": clipped_pct,
                        "trust_tuning_seconds": float(trust_tuning_seconds),
                        "model_tuning_seconds": float(model_tuning_seconds),
                        "fit_predict_seconds": float(fit_predict_seconds),
                        "frozen_v1_mae": float(frozen["mae"]),
                        "v1_mae": float(v1["mae"]),
                        "v2_mae": float(v2["mae"]),
                        "mae_improvement": improvement,
                        "mae_improvement_pct": (
                            100 * improvement / float(v1["mae"])
                            if np.isfinite(v1["mae"]) and float(v1["mae"]) > 0
                            else np.nan
                        ),
                        "v1_rmse": float(v1["rmse"]),
                        "v2_rmse": float(v2["rmse"]),
                        "v1_bias": float(v1["bias"]),
                        "v2_bias": float(v2["bias"]),
                        "v1_p95_abs_error": v1_tail["p95"],
                        "v2_p95_abs_error": v2_tail["p95"],
                        "v1_p99_abs_error": v1_tail["p99"],
                        "v2_p99_abs_error": v2_tail["p99"],
                        "v1_max_abs_error": v1_tail["max"],
                        "v2_max_abs_error": v2_tail["max"],
                        "v1_max_prediction": float(
                            pd.to_numeric(scoped[v1_column], errors="coerce").max()
                        ),
                        "v2_max_prediction": float(
                            pd.to_numeric(scoped[v2_column], errors="coerce").max()
                        ),
                    }
                )
    return rows


def run_multiplier_bounds_experiment(
    frame: pd.DataFrame,
    *,
    test_seasons: Sequence[str],
    bound_strategies: Sequence[str] = BOUND_STRATEGIES,
    training_cohorts: Sequence[str] = TRAINING_COHORTS,
    model_types: Sequence[str] = ("elastic_net",),
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
    progress: Callable[[str], None] | None = None,
) -> MultiplierBoundsExperimentResult:
    """Compare robust bound strategies in leakage-safe rolling season folds."""
    strategies = list(dict.fromkeys(str(value) for value in bound_strategies))
    unknown_strategies = sorted(set(strategies) - set(BOUND_STRATEGIES))
    if unknown_strategies:
        raise ValueError(f"Unsupported bound strategies: {unknown_strategies}")
    cohorts = list(dict.fromkeys(str(value) for value in training_cohorts))
    unknown_cohorts = sorted(set(cohorts) - set(TRAINING_COHORTS))
    if unknown_cohorts:
        raise ValueError(f"Unsupported training cohorts: {unknown_cohorts}")
    models = list(dict.fromkeys(str(value) for value in model_types))
    unknown_models = sorted(set(models) - set(ROLLING_MODEL_TYPES))
    if unknown_models:
        raise ValueError(f"Unsupported model types: {unknown_models}")
    if not strategies or not cohorts or not models:
        raise ValueError("At least one bound strategy, cohort, and model are required.")

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
            "full_test_rows": int(len(test_all)),
            "training_cohorts": {},
        }

        for training_cohort in cohorts:
            training = (
                training_all
                if training_cohort == "all"
                else training_all[training_all["regular_minutes_flag"]].copy()
            )
            cohort_parameters: dict[str, Any] = {}
            parameters[fold.test_season]["training_cohorts"][
                training_cohort
            ] = cohort_parameters

            for strategy_number, strategy in enumerate(strategies):
                bounds = derive_multiplier_bounds(training, strategy)
                if progress:
                    progress(
                        f"{fold.test_season}, {training_cohort}, {strategy}: "
                        f"{len(training)}/{len(training_all)} training rows"
                    )
                trust_started = perf_counter()
                trust_result = tune_trust_parameters(
                    training,
                    n_iter=trust_n_iter,
                    n_splits=n_splits,
                    random_state=random_state + fold_number,
                    multiplier_bounds=bounds,
                )
                trust_seconds = perf_counter() - trust_started
                trust = trust_result.best_parameters
                rebased_training = rebase_training_frame_with_trust(
                    training,
                    trust,
                    bounds,
                )
                rebased_test = rebase_feature_frame_with_trust(
                    test_all,
                    trust,
                    bounds,
                )
                trial_prefix = f"{fold.test_season}/{training_cohort}/{strategy}"
                trials[f"{trial_prefix}/trust"] = trust_result.trials
                strategy_parameters = {
                    "training_rows": int(len(training)),
                    "multiplier_bounds": bounds.to_dict(),
                    "trust": {
                        "parameters": asdict(trust),
                        "objective": trust_result.best_objective,
                        "tuning_seconds": trust_seconds,
                    },
                    "models": {},
                }
                cohort_parameters[strategy] = strategy_parameters

                for model_number, model_type in enumerate(models):
                    n_iter = int(iterations[model_type])
                    tuning_started = perf_counter()
                    fit_parameters, tuning_summary, model_trials = (
                        tune_model_parameters(
                            rebased_training,
                            model_type=model_type,
                            n_iter=n_iter,
                            n_splits=n_splits,
                            min_rows=min_rows,
                            random_state=(
                                random_state
                                + fold_number * 100
                                + model_number
                            ),
                            parameter_space=spaces.get(
                                model_type,
                                MODEL_PARAMETER_SPACES[model_type],
                            ),
                            neural_tuning_max_epochs=neural_tuning_max_epochs,
                            neural_tuning_patience=neural_tuning_patience,
                            neural_final_ensemble_size=(
                                neural_final_ensemble_size
                            ),
                        )
                    )
                    model_tuning_seconds = perf_counter() - tuning_started
                    for trial_name, trial_frame in model_trials.items():
                        trials[
                            f"{trial_prefix}/{model_type}/{trial_name}"
                        ] = trial_frame

                    fit_started = perf_counter()
                    bundle = fit_residual_models(
                        rebased_training,
                        model_type=model_type,
                        parameters_by_target=fit_parameters,
                        min_rows=min_rows,
                        random_state=(
                            random_state + fold_number * 100 + model_number
                        ),
                    )
                    predictions = bundle.predict(rebased_test)
                    fit_predict_seconds = perf_counter() - fit_started
                    predictions.insert(0, "bound_strategy", strategy)
                    predictions.insert(0, "training_cohort", training_cohort)
                    predictions.insert(0, "model_type", model_type)
                    predictions.insert(0, "test_season", fold.test_season)
                    diagnostic_columns = [
                        "actual_minutes",
                        "v1_pred_minutes",
                        "minute_difference",
                        "absolute_minute_error",
                        "regular_minutes_tolerance",
                        "regular_minutes_flag",
                        "baseline_fga",
                        "baseline_reb_chances",
                        "raw_fga_multiplier",
                        "bounded_fga_multiplier",
                        "fga_multiplier_was_bounded",
                        "fga_multiplier",
                        "raw_reb_chances_multiplier",
                        "bounded_reb_chances_multiplier",
                        "reb_chances_multiplier_was_bounded",
                        "reb_chances_multiplier",
                    ]
                    for column in diagnostic_columns:
                        predictions[column] = rebased_test[column].to_numpy()
                    prediction_frames.append(predictions)
                    summary_rows.extend(
                        _evaluation_rows(
                            predictions,
                            test_season=fold.test_season,
                            training_cohort=training_cohort,
                            bound_strategy=strategy,
                            model_type=model_type,
                            bounds=bounds,
                            full_training_rows=len(training_all),
                            training_rows=len(training),
                            full_test_rows=len(test_all),
                            trust_tuning_seconds=trust_seconds,
                            model_tuning_seconds=model_tuning_seconds,
                            fit_predict_seconds=fit_predict_seconds,
                        )
                    )
                    strategy_parameters["models"][model_type] = {
                        **tuning_summary,
                        "tuning_seconds": model_tuning_seconds,
                        "fit_predict_seconds": fit_predict_seconds,
                    }
                    if progress:
                        progress(
                            f"{fold.test_season}, {training_cohort}, "
                            f"{strategy}, {model_type}: scored "
                            f"{len(predictions)} rows"
                        )

    return MultiplierBoundsExperimentResult(
        summary=pd.DataFrame(summary_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        parameters=parameters,
        trials=trials,
    )

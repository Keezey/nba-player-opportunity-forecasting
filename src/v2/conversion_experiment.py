"""Rolling evaluation of rebound-conversion estimators.

The opportunity models remain frozen. This experiment changes only the rate
used to translate held-out V2 rebound-chance predictions into rebounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .conversion import (
    ConversionAdjustmentModel,
    ConversionPrior,
    fit_conversion_adjustment_model,
    fit_conversion_prior,
    predict_shrunken_conversion,
    prepare_conversion_frame,
)
from .evaluate import regression_metrics_for_aggregation, walk_forward_splits
from .regular_minutes_experiment import annotate_regular_minutes
from .rolling_experiment import build_rolling_season_folds


CONVERSION_METHODS = ("current", "shrunk", "elastic_net")
TRAINING_COHORTS = ("all", "regular")
EVALUATION_COHORTS = ("all", "regular", "irregular")
AGGREGATIONS = ("game_weighted", "player_weighted")

DEFAULT_SHRINKAGE_STRENGTHS = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
DEFAULT_ELASTIC_PARAMETER_SPACE = {
    "alpha": (0.001, 0.003, 0.01, 0.03, 0.1),
    "l1_ratio": (0.1, 0.25, 0.5, 0.75, 1.0),
}


@dataclass(frozen=True)
class ConversionExperimentResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    bootstrap: pd.DataFrame
    parameters: dict[str, Any]
    trials: dict[str, pd.DataFrame]


def _oracle_chance_mae(
    frame: pd.DataFrame,
    predicted_conversion: pd.Series,
) -> float:
    chances = pd.to_numeric(frame["actual_reb_chances"], errors="coerce")
    rebounds = pd.to_numeric(frame["actual_reb"], errors="coerce")
    rate = pd.to_numeric(predicted_conversion, errors="coerce")
    valid = chances.notna() & rebounds.notna() & rate.notna() & (chances > 0)
    if not valid.any():
        return np.nan
    return float((chances[valid] * rate[valid] - rebounds[valid]).abs().mean())


def tune_shrinkage_strength(
    frame: pd.DataFrame,
    *,
    strengths: Sequence[float] = DEFAULT_SHRINKAGE_STRENGTHS,
    n_splits: int = 3,
) -> tuple[float, pd.DataFrame]:
    """Choose prior strength using expanding inner-date validation."""
    candidates = [float(value) for value in strengths]
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("Shrinkage strengths must be a non-empty nonnegative list.")
    prepared = prepare_conversion_frame(frame).reset_index(drop=True)
    splits = walk_forward_splits(prepared, n_splits=n_splits)
    rows: list[dict[str, float | int]] = []
    for strength in candidates:
        fold_scores: list[float] = []
        for train_index, validation_index in splits:
            inner_train = prepared.loc[train_index]
            validation = prepared.loc[validation_index]
            prior = fit_conversion_prior(inner_train)
            predicted = predict_shrunken_conversion(
                validation,
                prior,
                prior_strength=strength,
            )
            score = _oracle_chance_mae(validation, predicted)
            if np.isfinite(score):
                fold_scores.append(score)
        rows.append(
            {
                "prior_strength": strength,
                "folds": len(fold_scores),
                "mean_oracle_chance_mae": (
                    float(np.mean(fold_scores)) if fold_scores else np.nan
                ),
            }
        )
    trials = pd.DataFrame(rows).sort_values(
        ["mean_oracle_chance_mae", "prior_strength"], kind="stable"
    )
    usable = trials[trials["mean_oracle_chance_mae"].notna()]
    if usable.empty:
        raise ValueError("Shrinkage tuning produced no usable validation scores.")
    return float(usable.iloc[0]["prior_strength"]), trials.reset_index(drop=True)


def _elastic_candidates(
    parameter_space: Mapping[str, Sequence[float]],
    *,
    n_iter: int,
    random_state: int,
) -> list[dict[str, float]]:
    alphas = [float(value) for value in parameter_space["alpha"]]
    ratios = [float(value) for value in parameter_space["l1_ratio"]]
    candidates = [
        {"alpha": alpha, "l1_ratio": ratio}
        for alpha, ratio in product(alphas, ratios)
    ]
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if n_iter >= len(candidates):
        return candidates
    rng = np.random.default_rng(random_state)
    selected = rng.choice(len(candidates), size=n_iter, replace=False)
    return [candidates[int(index)] for index in selected]


def tune_conversion_adjustment_model(
    frame: pd.DataFrame,
    *,
    parameter_space: Mapping[str, Sequence[float]] = DEFAULT_ELASTIC_PARAMETER_SPACE,
    n_iter: int = 20,
    n_splits: int = 3,
    min_rows: int = 30,
    random_state: int = 42,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune the conversion adjustment on expanding inner-date folds."""
    prepared = prepare_conversion_frame(frame).reset_index(drop=True)
    splits = walk_forward_splits(prepared, n_splits=n_splits)
    candidates = _elastic_candidates(
        parameter_space,
        n_iter=n_iter,
        random_state=random_state,
    )
    rows: list[dict[str, float | int]] = []
    for candidate_number, parameters in enumerate(candidates):
        fold_scores: list[float] = []
        for split_number, (train_index, validation_index) in enumerate(splits):
            inner_train = prepared.loc[train_index]
            validation = prepared.loc[validation_index]
            try:
                model = fit_conversion_adjustment_model(
                    inner_train,
                    parameters=parameters,
                    min_rows=min_rows,
                    random_state=random_state + candidate_number * 100 + split_number,
                )
            except ValueError:
                continue
            predicted = model.predict_rate(validation)
            score = _oracle_chance_mae(validation, predicted)
            if np.isfinite(score):
                fold_scores.append(score)
        rows.append(
            {
                **parameters,
                "folds": len(fold_scores),
                "mean_oracle_chance_mae": (
                    float(np.mean(fold_scores)) if fold_scores else np.nan
                ),
            }
        )
    trials = pd.DataFrame(rows).sort_values(
        ["mean_oracle_chance_mae", "alpha", "l1_ratio"], kind="stable"
    )
    usable = trials[trials["mean_oracle_chance_mae"].notna()]
    if usable.empty:
        raise ValueError("Conversion tuning produced no usable validation scores.")
    best = usable.iloc[0]
    return {
        "alpha": float(best["alpha"]),
        "l1_ratio": float(best["l1_ratio"]),
    }, trials.reset_index(drop=True)


def _select_base_predictions(
    predictions: pd.DataFrame,
    *,
    test_season: str,
    training_cohort: str,
    model_type: str,
    bound_strategy: str,
) -> pd.DataFrame:
    required = [
        "test_season",
        "training_cohort",
        "model_type",
        "bound_strategy",
        "game_id",
        "player_id",
        "v2_pred_reb_chances",
    ]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"Base predictions are missing columns: {missing}")
    selected = predictions[
        (predictions["test_season"].astype(str) == str(test_season))
        & (predictions["training_cohort"].astype(str) == training_cohort)
        & (predictions["model_type"].astype(str) == model_type)
        & (predictions["bound_strategy"].astype(str) == bound_strategy)
    ][["game_id", "player_id", "v2_pred_reb_chances"]].copy()
    if selected.empty:
        raise ValueError(
            "No base predictions match "
            f"{test_season}/{training_cohort}/{model_type}/{bound_strategy}."
        )
    if selected.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Selected base predictions contain duplicate game/player rows.")
    return selected


def _tail_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = pd.to_numeric(actual, errors="coerce")
    predicted_values = pd.to_numeric(predicted, errors="coerce")
    valid = actual_values.notna() & predicted_values.notna()
    errors = (predicted_values[valid] - actual_values[valid]).abs()
    if errors.empty:
        return {"p95": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "p95": float(errors.quantile(0.95)),
        "p99": float(errors.quantile(0.99)),
        "max": float(errors.max()),
    }


def _weighted_conversion_mae(
    frame: pd.DataFrame,
    conversion_column: str,
) -> float:
    chances = pd.to_numeric(frame["actual_reb_chances"], errors="coerce")
    actual = pd.to_numeric(frame["actual_reb"], errors="coerce")
    predicted = pd.to_numeric(frame[conversion_column], errors="coerce")
    valid = chances.notna() & actual.notna() & predicted.notna() & (chances > 0)
    if not valid.any():
        return np.nan
    actual_rate = actual[valid] / chances[valid]
    return float(
        np.average(
            np.abs(predicted[valid] - actual_rate),
            weights=chances[valid],
        )
    )


def _evaluation_rows(
    predictions: pd.DataFrame,
    *,
    test_season: str,
    training_cohort: str,
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
            current = regression_metrics_for_aggregation(
                scoped,
                actual_column="actual_reb",
                prediction_column="current_pred_reb",
                aggregation=aggregation,
            )
            for method in CONVERSION_METHODS:
                prediction_column = f"{method}_pred_reb"
                conversion_column = f"{method}_conversion"
                metrics = regression_metrics_for_aggregation(
                    scoped,
                    actual_column="actual_reb",
                    prediction_column=prediction_column,
                    aggregation=aggregation,
                )
                tails = _tail_metrics(scoped["actual_reb"], scoped[prediction_column])
                improvement = float(current["mae"] - metrics["mae"])
                rows.append(
                    {
                        "test_season": test_season,
                        "training_cohort": training_cohort,
                        "evaluation_cohort": evaluation_cohort,
                        "aggregation": aggregation,
                        "method": method,
                        "test_rows": int(metrics["n"]),
                        "test_players": int(metrics["players"]),
                        "mae": float(metrics["mae"]),
                        "rmse": float(metrics["rmse"]),
                        "bias": float(metrics["bias"]),
                        "p95_abs_error": tails["p95"],
                        "p99_abs_error": tails["p99"],
                        "max_abs_error": tails["max"],
                        "weighted_conversion_mae": _weighted_conversion_mae(
                            scoped, conversion_column
                        ),
                        "mae_improvement_vs_current": improvement,
                        "mae_improvement_pct_vs_current": (
                            100 * improvement / float(current["mae"])
                            if float(current["mae"]) > 0
                            else np.nan
                        ),
                    }
                )
    return rows


def bootstrap_conversion_improvements(
    predictions: pd.DataFrame,
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap paired MAE gains by resampling players within each season."""
    if samples < 1:
        raise ValueError("Bootstrap samples must be at least 1.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    required = [
        "test_season",
        "training_cohort",
        "player_id",
        "regular_minutes_flag",
        "actual_reb",
        "current_pred_reb",
        "shrunk_pred_reb",
        "elastic_net_pred_reb",
    ]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"Bootstrap predictions are missing columns: {missing}")

    rng = np.random.default_rng(random_state)
    alpha = (1 - confidence) / 2
    rows: list[dict[str, Any]] = []
    for training_cohort in sorted(predictions["training_cohort"].unique()):
        cohort = predictions[
            predictions["training_cohort"] == training_cohort
        ].copy()
        regular = cohort["regular_minutes_flag"].astype(bool)
        masks = {
            "all": pd.Series(True, index=cohort.index),
            "regular": regular,
            "irregular": ~regular,
        }
        for evaluation_cohort, mask in masks.items():
            scoped = cohort.loc[mask].copy()
            if scoped.empty:
                continue
            actual = pd.to_numeric(scoped["actual_reb"], errors="coerce")
            current = pd.to_numeric(
                scoped["current_pred_reb"], errors="coerce"
            )
            current_abs = (current - actual).abs()
            for method in ("shrunk", "elastic_net"):
                method_predicted = pd.to_numeric(
                    scoped[f"{method}_pred_reb"], errors="coerce"
                )
                valid = actual.notna() & current.notna() & method_predicted.notna()
                paired = scoped.loc[
                    valid, ["test_season", "player_id"]
                ].copy()
                paired["gain"] = (
                    current_abs[valid]
                    - (method_predicted[valid] - actual[valid]).abs()
                )
                season_clusters: list[tuple[np.ndarray, np.ndarray]] = []
                observed_by_season: list[float] = []
                for _, season in paired.groupby("test_season", sort=True):
                    clusters = season.groupby("player_id", sort=False).agg(
                        gain_sum=("gain", "sum"),
                        rows=("gain", "size"),
                    )
                    gain_sum = clusters["gain_sum"].to_numpy(dtype=float)
                    counts = clusters["rows"].to_numpy(dtype=float)
                    if len(gain_sum) == 0:
                        continue
                    season_clusters.append((gain_sum, counts))
                    observed_by_season.append(float(gain_sum.sum() / counts.sum()))
                if not season_clusters:
                    continue

                draws = np.empty(samples, dtype=float)
                for draw in range(samples):
                    season_gains: list[float] = []
                    for gain_sum, counts in season_clusters:
                        sampled = rng.integers(0, len(gain_sum), size=len(gain_sum))
                        season_gains.append(
                            float(
                                gain_sum[sampled].sum()
                                / counts[sampled].sum()
                            )
                        )
                    draws[draw] = float(np.mean(season_gains))
                rows.append(
                    {
                        "training_cohort": training_cohort,
                        "evaluation_cohort": evaluation_cohort,
                        "method": method,
                        "seasons": len(season_clusters),
                        "players": int(paired["player_id"].nunique()),
                        "rows": int(len(paired)),
                        "bootstrap_samples": samples,
                        "observed_mae_improvement": float(
                            np.mean(observed_by_season)
                        ),
                        "bootstrap_mean_improvement": float(draws.mean()),
                        "ci_lower": float(np.quantile(draws, alpha)),
                        "ci_upper": float(np.quantile(draws, 1 - alpha)),
                        "probability_improvement": float((draws > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def run_conversion_experiment(
    frame: pd.DataFrame,
    base_predictions: pd.DataFrame,
    *,
    test_seasons: Sequence[str],
    training_cohorts: Sequence[str] = TRAINING_COHORTS,
    base_model_type: str = "elastic_net",
    base_bound_strategy: str = "fixed_0.50_1.50",
    absolute_tolerance: float = 3.0,
    relative_tolerance: float = 0.15,
    shrinkage_strengths: Sequence[float] = DEFAULT_SHRINKAGE_STRENGTHS,
    elastic_parameter_space: Mapping[
        str, Sequence[float]
    ] = DEFAULT_ELASTIC_PARAMETER_SPACE,
    elastic_n_iter: int = 20,
    n_splits: int = 3,
    min_training_seasons: int = 2,
    min_rows: int = 30,
    bootstrap_samples: int = 2000,
    random_state: int = 42,
    progress: Callable[[str], None] | None = None,
) -> ConversionExperimentResult:
    """Evaluate current, shrunken, and learned conversion on future seasons."""
    cohorts = list(dict.fromkeys(str(value) for value in training_cohorts))
    unknown = sorted(set(cohorts) - set(TRAINING_COHORTS))
    if unknown:
        raise ValueError(f"Unsupported conversion training cohorts: {unknown}")
    if not cohorts:
        raise ValueError("At least one training cohort is required.")

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
        for cohort_number, training_cohort in enumerate(cohorts):
            training = (
                training_all
                if training_cohort == "all"
                else training_all[training_all["regular_minutes_flag"]].copy()
            )
            if progress:
                progress(
                    f"{fold.test_season}, {training_cohort}: tuning conversion "
                    f"on {len(training)} rows"
                )
            best_strength, shrinkage_trials = tune_shrinkage_strength(
                training,
                strengths=shrinkage_strengths,
                n_splits=n_splits,
            )
            seed = random_state + fold_number * 100 + cohort_number
            best_elastic, elastic_trials = tune_conversion_adjustment_model(
                training,
                parameter_space=elastic_parameter_space,
                n_iter=elastic_n_iter,
                n_splits=n_splits,
                min_rows=min_rows,
                random_state=seed,
            )
            prior = fit_conversion_prior(training)
            model = fit_conversion_adjustment_model(
                training,
                parameters=best_elastic,
                min_rows=min_rows,
                random_state=seed,
            )
            trials[f"{fold.test_season}/{training_cohort}/shrinkage"] = (
                shrinkage_trials
            )
            trials[f"{fold.test_season}/{training_cohort}/elastic_net"] = (
                elastic_trials
            )

            base = _select_base_predictions(
                base_predictions,
                test_season=fold.test_season,
                training_cohort=training_cohort,
                model_type=base_model_type,
                bound_strategy=base_bound_strategy,
            )
            test = test_all.merge(
                base,
                on=["game_id", "player_id"],
                how="inner",
                validate="one_to_one",
            )
            if len(test) != len(test_all):
                raise ValueError(
                    f"Base predictions cover {len(test)}/{len(test_all)} rows for "
                    f"{fold.test_season}/{training_cohort}."
                )
            test = prepare_conversion_frame(test)
            test["current_conversion"] = test[
                "baseline_reb_conversion"
            ].clip(lower=0, upper=1)
            test["shrunk_conversion"] = predict_shrunken_conversion(
                test,
                prior,
                prior_strength=best_strength,
            )
            test["elastic_net_conversion"] = model.predict_rate(test)
            for method in CONVERSION_METHODS:
                test[f"{method}_pred_reb"] = (
                    test["v2_pred_reb_chances"] * test[f"{method}_conversion"]
                )
            test.insert(0, "training_cohort", training_cohort)
            test.insert(0, "test_season", fold.test_season)
            prediction_frames.append(
                test[
                    [
                        "test_season",
                        "training_cohort",
                        "game_id",
                        "game_date",
                        "season",
                        "player_id",
                        "opp_abbr",
                        "actual_minutes",
                        "v1_pred_minutes",
                        "regular_minutes_flag",
                        "actual_reb_chances",
                        "actual_reb",
                        "v2_pred_reb_chances",
                        "baseline_reb_conversion",
                        "conversion_exposure",
                        "current_conversion",
                        "shrunk_conversion",
                        "elastic_net_conversion",
                        "current_pred_reb",
                        "shrunk_pred_reb",
                        "elastic_net_pred_reb",
                    ]
                ].copy()
            )
            summary_rows.extend(
                _evaluation_rows(
                    test,
                    test_season=fold.test_season,
                    training_cohort=training_cohort,
                )
            )
            parameters[fold.test_season]["training_cohorts"][training_cohort] = {
                "training_rows": int(len(training)),
                "training_players": int(training["player_id"].nunique()),
                "prior_strength": best_strength,
                "global_prior": prior.global_rate,
                "position_priors": prior.position_rates,
                "elastic_net": best_elastic,
                "conversion_lower_bound": model.lower_bound,
                "conversion_upper_bound": model.upper_bound,
                "conversion_model_rows": model.training_rows,
            }
            if progress:
                progress(
                    f"{fold.test_season}, {training_cohort}: scored {len(test)} rows"
                )

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    return ConversionExperimentResult(
        summary=pd.DataFrame(summary_rows),
        predictions=combined_predictions,
        bootstrap=bootstrap_conversion_improvements(
            combined_predictions,
            samples=bootstrap_samples,
            random_state=random_state,
        ),
        parameters=parameters,
        trials=trials,
    )

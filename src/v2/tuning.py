"""Numerical tuning for V2 models, trust anchors, and the V1 workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..trust import MultiplierBounds
from .evaluate import regression_metrics, walk_forward_splits
from .models import (
    DEFAULT_MODEL_PARAMETERS,
    TARGET_SPECS,
    fit_residual_models,
    fit_residual_model,
    get_target_spec,
)
from .training_data import TrainingDatasetResult, build_training_dataset


MODEL_PARAMETER_SPACES: dict[str, dict[str, Sequence[Any]]] = {
    "elastic_net": {
        "alpha": [0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
        "l1_ratio": [0.1, 0.25, 0.5, 0.75, 1.0],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "max_iter": [150, 250, 400],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [10, 20, 30, 40],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
    },
    "pytorch_multitask": {
        "hidden_size_1": [32, 64, 96],
        "hidden_size_2": [16, 32, 48],
        "dropout": [0.05, 0.15, 0.25],
        "learning_rate": [0.0005, 0.001, 0.002],
        "weight_decay": [0.0, 0.0001, 0.001],
        "batch_size": [128, 256],
    },
}

WORKFLOW_PARAMETER_SPACE: dict[str, Sequence[Any]] = {
    "top_n_similar": [5, 8, 10, 12, 15],
    "lookback_days": [21, 30, 45],
    "min_games": [4, 5, 6],
    "min_minutes_ratio": [0.65, 0.75, 0.85],
    "position_weight": [0.0, 0.2, 0.35, 0.5, 0.75],
    "require_same_starter_flag": [False, True],
    "fga_per_min_weight": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "reb_chances_per_min_weight": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "usage_rate_weight": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "touches_per_min_weight": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "starter_rate_weight": [0.25, 0.5, 0.75, 1.0, 1.25],
}


@dataclass(frozen=True)
class ModelTuningResult:
    target: str
    model_type: str
    best_parameters: dict[str, Any]
    best_v2_mae: float
    mean_v1_mae: float
    trials: pd.DataFrame


@dataclass(frozen=True)
class TrustParameters:
    low_sample_threshold: int = 5
    high_sample_threshold: int = 10
    max_sample_threshold: int = 15
    low_trust_weight: float = 0.35
    high_trust_weight: float = 0.80
    max_trust_weight: float = 0.95


@dataclass(frozen=True)
class TrustTuningResult:
    best_parameters: TrustParameters
    best_objective: float
    trials: pd.DataFrame


@dataclass(frozen=True)
class WorkflowTuningResult:
    best_parameters: dict[str, Any]
    best_objective: float
    reference_rows: int
    trials: pd.DataFrame


def _plain_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _sample_candidates(
    space: Mapping[str, Sequence[Any]],
    n_iter: int,
    random_state: int,
    *,
    include_default: bool = True,
) -> list[dict[str, Any]]:
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if any(not values for values in space.values()):
        raise ValueError("Every search-space parameter must have at least one value.")

    rng = np.random.default_rng(random_state)
    candidates: list[dict[str, Any]] = [{}] if include_default else []
    seen = {"{}"} if include_default else set()
    attempts = 0
    while len(candidates) < n_iter and attempts < n_iter * 200:
        candidate = {
            key: _plain_value(rng.choice(list(values))) for key, values in space.items()
        }
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
        attempts += 1
    return candidates


def tune_residual_model(
    training_df: pd.DataFrame,
    target: str,
    *,
    model_type: str = "hist_gradient_boosting",
    parameter_space: Mapping[str, Sequence[Any]] | None = None,
    n_iter: int = 20,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    gap_days: int = 0,
    min_rows: int = 30,
    random_state: int = 42,
) -> ModelTuningResult:
    """Choose residual-model parameters with expanding date-based folds."""
    spec = get_target_spec(target)
    space = dict(parameter_space or MODEL_PARAMETER_SPACES[model_type])
    candidates = _sample_candidates(space, n_iter, random_state)
    splits = walk_forward_splits(
        training_df,
        n_splits=n_splits,
        min_train_fraction=min_train_fraction,
        gap_days=gap_days,
    )

    trial_rows: list[dict[str, Any]] = []
    for trial_number, parameters in enumerate(candidates):
        v1_errors: list[pd.Series] = []
        v2_errors: list[pd.Series] = []
        folds_used = 0
        error_message = ""
        try:
            for train_index, validation_index in splits:
                train = training_df.loc[train_index]
                validation = training_df.loc[validation_index]
                model = fit_residual_model(
                    train,
                    target,
                    model_type=model_type,
                    parameters=parameters,
                    min_rows=min_rows,
                    random_state=random_state,
                )
                actual = pd.to_numeric(
                    validation[spec.actual_column], errors="coerce"
                )
                v1 = pd.to_numeric(
                    validation[spec.v1_prediction_column], errors="coerce"
                )
                v2 = (v1 + model.predict_correction(validation)).clip(lower=0)
                valid = actual.notna() & v1.notna() & v2.notna()
                if valid.any():
                    v1_errors.append(v1[valid] - actual[valid])
                    v2_errors.append(v2[valid] - actual[valid])
                    folds_used += 1
        except (KeyError, ValueError) as exc:
            error_message = str(exc)

        if v2_errors:
            combined_v1 = pd.concat(v1_errors, ignore_index=True)
            combined_v2 = pd.concat(v2_errors, ignore_index=True)
            v1_mae = float(combined_v1.abs().mean())
            v2_mae = float(combined_v2.abs().mean())
        else:
            v1_mae = np.nan
            v2_mae = np.inf
        trial_rows.append(
            {
                "trial": trial_number,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "folds_used": folds_used,
                "v1_mae": v1_mae,
                "v2_mae": v2_mae,
                "error": error_message,
            }
        )

    trials = pd.DataFrame(trial_rows).sort_values(["v2_mae", "trial"])
    usable = trials[np.isfinite(trials["v2_mae"])]
    if usable.empty:
        raise ValueError("No parameter trial produced a usable validation score.")
    best = usable.iloc[0]
    return ModelTuningResult(
        target=target,
        model_type=model_type,
        best_parameters=json.loads(best["parameters_json"]),
        best_v2_mae=float(best["v2_mae"]),
        mean_v1_mae=float(best["v1_mae"]),
        trials=trials.reset_index(drop=True),
    )


def tune_multitask_residual_models(
    training_df: pd.DataFrame,
    *,
    parameter_space: Mapping[str, Sequence[Any]] | None = None,
    n_iter: int = 8,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    gap_days: int = 0,
    min_rows: int = 30,
    random_state: int = 42,
    tuning_max_epochs: int = 80,
    tuning_patience: int = 10,
    final_ensemble_size: int = 5,
) -> dict[str, ModelTuningResult]:
    """Tune one shared network using the mean normalized opportunity MAE."""
    space = dict(parameter_space or MODEL_PARAMETER_SPACES["pytorch_multitask"])
    candidates = _sample_candidates(space, n_iter, random_state)
    splits = walk_forward_splits(
        training_df,
        n_splits=n_splits,
        min_train_fraction=min_train_fraction,
        gap_days=gap_days,
    )

    trial_rows = []
    for trial_number, candidate in enumerate(candidates):
        fit_parameters = {
            "max_epochs": int(tuning_max_epochs),
            "patience": int(tuning_patience),
            "ensemble_size": 1,
            **candidate,
        }
        final_parameters = {
            **fit_parameters,
            "ensemble_size": int(final_ensemble_size),
        }
        errors = {
            target: {"v1": [], "v2": []}
            for target in TARGET_SPECS
        }
        folds_used = 0
        error_message = ""
        try:
            for train_index, validation_index in splits:
                train = training_df.loc[train_index]
                validation = training_df.loc[validation_index]
                bundle = fit_residual_models(
                    train,
                    model_type="pytorch_multitask",
                    parameters_by_target={"shared": fit_parameters},
                    min_rows=min_rows,
                    random_state=random_state,
                )
                predictions = bundle.predict(validation)
                fold_used = False
                for target, spec in TARGET_SPECS.items():
                    actual = pd.to_numeric(
                        predictions[spec.actual_column], errors="coerce"
                    )
                    v1 = pd.to_numeric(
                        predictions[spec.v1_prediction_column], errors="coerce"
                    )
                    v2 = pd.to_numeric(
                        predictions[spec.v2_prediction_column], errors="coerce"
                    )
                    valid = actual.notna() & v1.notna() & v2.notna()
                    if valid.any():
                        errors[target]["v1"].append(v1[valid] - actual[valid])
                        errors[target]["v2"].append(v2[valid] - actual[valid])
                        fold_used = True
                folds_used += int(fold_used)
        except (KeyError, RuntimeError, ValueError) as exc:
            error_message = str(exc)

        scores = {}
        normalized_scores = []
        for target in TARGET_SPECS:
            if errors[target]["v2"]:
                v1_error = pd.concat(errors[target]["v1"], ignore_index=True)
                v2_error = pd.concat(errors[target]["v2"], ignore_index=True)
                v1_mae = float(v1_error.abs().mean())
                v2_mae = float(v2_error.abs().mean())
                normalized_scores.append(v2_mae / max(v1_mae, 1e-9))
            else:
                v1_mae = np.nan
                v2_mae = np.inf
            scores[target] = (v1_mae, v2_mae)
        objective = (
            float(np.mean(normalized_scores)) if normalized_scores else np.inf
        )
        trial_rows.append(
            {
                "trial": trial_number,
                "parameters_json": json.dumps(final_parameters, sort_keys=True),
                "folds_used": folds_used,
                "objective": objective,
                "fga_v1_mae": scores["fga"][0],
                "fga_v2_mae": scores["fga"][1],
                "reb_chances_v1_mae": scores["reb_chances"][0],
                "reb_chances_v2_mae": scores["reb_chances"][1],
                "tuning_ensemble_size": 1,
                "error": error_message,
            }
        )

    trials = pd.DataFrame(trial_rows).sort_values(["objective", "trial"])
    usable = trials[np.isfinite(trials["objective"])]
    if usable.empty:
        raise ValueError("No neural parameter trial produced a usable validation score.")
    best = usable.iloc[0]
    best_parameters = json.loads(best["parameters_json"])
    output = {}
    for target in TARGET_SPECS:
        output[target] = ModelTuningResult(
            target=target,
            model_type="pytorch_multitask",
            best_parameters=best_parameters,
            best_v2_mae=float(best[f"{target}_v2_mae"]),
            mean_v1_mae=float(best[f"{target}_v1_mae"]),
            trials=trials.reset_index(drop=True),
        )
    return output


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3 - 2 * clipped)


def trust_weights(sample_counts: pd.Series, parameters: TrustParameters) -> pd.Series:
    """Vectorized monotonic trust curve with learnable anchor weights."""
    if not (
        0 <= parameters.low_trust_weight
        <= parameters.high_trust_weight
        <= parameters.max_trust_weight
        <= 1
    ):
        raise ValueError("Trust weights must be monotonic and between 0 and 1.")

    counts = pd.to_numeric(sample_counts, errors="coerce").fillna(0).clip(lower=0)
    values = np.zeros(len(counts), dtype=float)
    n = counts.to_numpy(dtype=float)

    low = (n > 0) & (n <= parameters.low_sample_threshold)
    progress = n[low] / parameters.low_sample_threshold
    values[low] = parameters.low_trust_weight * _smoothstep(progress)

    middle = (n > parameters.low_sample_threshold) & (
        n <= parameters.high_sample_threshold
    )
    progress = (n[middle] - parameters.low_sample_threshold) / (
        parameters.high_sample_threshold - parameters.low_sample_threshold
    )
    values[middle] = parameters.low_trust_weight + (
        parameters.high_trust_weight - parameters.low_trust_weight
    ) * _smoothstep(progress)

    high = (n > parameters.high_sample_threshold) & (
        n <= parameters.max_sample_threshold
    )
    progress = (n[high] - parameters.high_sample_threshold) / (
        parameters.max_sample_threshold - parameters.high_sample_threshold
    )
    values[high] = parameters.high_trust_weight + (
        parameters.max_trust_weight - parameters.high_trust_weight
    ) * _smoothstep(progress)
    values[n > parameters.max_sample_threshold] = parameters.max_trust_weight
    return pd.Series(values, index=sample_counts.index)


def apply_trust_parameters(
    frame: pd.DataFrame,
    parameters: TrustParameters,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Bound raw ratios and recalculate V1 projections using trust anchors."""
    bounds = MultiplierBounds.from_value(multiplier_bounds)
    output = pd.DataFrame(index=frame.index)
    metric_columns = {
        "fga": (
            "raw_fga_multiplier",
            "fga_n_samples",
            "baseline_fga",
            "tuned_pred_fga",
        ),
        "reb_chances": (
            "raw_reb_chances_multiplier",
            "reb_chances_n_samples",
            "baseline_reb_chances",
            "tuned_pred_reb_chances",
        ),
    }
    for metric, (raw_column, sample_column, baseline_column, output_column) in metric_columns.items():
        raw = pd.to_numeric(frame[raw_column], errors="coerce")
        valid = raw.notna() & np.isfinite(raw)
        lower, upper = bounds.for_metric(metric)
        bounded = raw.copy()
        if lower is not None:
            bounded = bounded.clip(lower=lower)
        if upper is not None:
            bounded = bounded.clip(upper=upper)
        bounded = bounded.where(valid)
        trust = trust_weights(frame[sample_column], parameters)
        effective = trust * bounded + (1 - trust)
        effective = effective.where(valid, 1.0)
        output[f"{metric}_trust_weight"] = trust
        output[f"{metric}_bounded_raw_multiplier"] = bounded
        output[f"{metric}_multiplier_was_bounded"] = (
            valid & ~np.isclose(raw, bounded, equal_nan=True)
        )
        output[f"{metric}_multiplier_lower_bound"] = lower
        output[f"{metric}_multiplier_upper_bound"] = upper
        output[f"{metric}_effective_multiplier"] = effective
        output[output_column] = pd.to_numeric(
            frame[baseline_column], errors="coerce"
        ) * effective
    return output


def rebase_feature_frame_with_trust(
    frame: pd.DataFrame,
    parameters: TrustParameters,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply tuned trust to pregame features while retaining frozen V1."""
    result = frame.copy()
    tuned = apply_trust_parameters(result, parameters, multiplier_bounds)
    for stat in ["fga", "reb_chances", "reb"]:
        source = f"v1_pred_{stat}"
        if source in result.columns and f"frozen_{source}" not in result.columns:
            result[f"frozen_{source}"] = result[source]

    result["fga_trust_weight"] = tuned["fga_trust_weight"]
    result["bounded_fga_multiplier"] = tuned[
        "fga_bounded_raw_multiplier"
    ]
    result["fga_multiplier_was_bounded"] = tuned[
        "fga_multiplier_was_bounded"
    ]
    result["fga_multiplier"] = tuned["fga_effective_multiplier"]
    result["reb_chances_trust_weight"] = tuned["reb_chances_trust_weight"]
    result["bounded_reb_chances_multiplier"] = tuned[
        "reb_chances_bounded_raw_multiplier"
    ]
    result["reb_chances_multiplier_was_bounded"] = tuned[
        "reb_chances_multiplier_was_bounded"
    ]
    result["reb_chances_multiplier"] = tuned[
        "reb_chances_effective_multiplier"
    ]
    result["v1_pred_fga"] = tuned["tuned_pred_fga"]
    result["v1_pred_reb_chances"] = tuned["tuned_pred_reb_chances"]
    conversion = pd.to_numeric(
        result["baseline_reb_conversion"], errors="coerce"
    ).clip(lower=0, upper=1)
    result["v1_pred_reb"] = result["v1_pred_reb_chances"] * conversion
    return result


def rebase_training_frame_with_trust(
    frame: pd.DataFrame,
    parameters: TrustParameters,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply tuned trust and recalculate residual labels for model training."""
    result = rebase_feature_frame_with_trust(
        frame,
        parameters,
        multiplier_bounds,
    )
    result["fga_residual"] = pd.to_numeric(
        result["actual_fga"], errors="coerce"
    ) - result["v1_pred_fga"]
    result["reb_chances_residual"] = pd.to_numeric(
        result["actual_reb_chances"], errors="coerce"
    ) - result["v1_pred_reb_chances"]
    result["reb_residual"] = pd.to_numeric(
        result["actual_reb"], errors="coerce"
    ) - result["v1_pred_reb"]
    return result


def tune_trust_parameters(
    training_df: pd.DataFrame,
    *,
    n_iter: int = 30,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
    random_state: int = 42,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = None,
) -> TrustTuningResult:
    """Tune the 5/10/15-sample trust anchors on later validation dates."""
    space = {
        "low_trust_weight": [0.15, 0.25, 0.35, 0.45, 0.55],
        "high_trust_weight": [0.65, 0.75, 0.80, 0.85, 0.90, 0.95],
        "max_trust_weight": [0.85, 0.90, 0.95, 0.98, 1.0],
    }
    default = asdict(TrustParameters())
    default_candidate = {
        "low_trust_weight": default["low_trust_weight"],
        "high_trust_weight": default["high_trust_weight"],
        "max_trust_weight": default["max_trust_weight"],
    }
    valid_candidates = [
        {
            "low_trust_weight": low,
            "high_trust_weight": high,
            "max_trust_weight": maximum,
        }
        for low, high, maximum in product(
            space["low_trust_weight"],
            space["high_trust_weight"],
            space["max_trust_weight"],
        )
        if low <= high <= maximum
        and (low, high, maximum)
        != (
            default_candidate["low_trust_weight"],
            default_candidate["high_trust_weight"],
            default_candidate["max_trust_weight"],
        )
    ]
    rng = np.random.default_rng(random_state)
    rng.shuffle(valid_candidates)
    candidate_dicts = [default_candidate] + valid_candidates[: max(n_iter - 1, 0)]
    validation_indices = pd.Index([])
    for _, test_index in walk_forward_splits(
        training_df,
        n_splits=n_splits,
        min_train_fraction=min_train_fraction,
    ):
        validation_indices = validation_indices.union(test_index)
    validation = training_df.loc[validation_indices]

    v1_fga = regression_metrics(validation["actual_fga"], validation["v1_pred_fga"])["mae"]
    v1_reb = regression_metrics(
        validation["actual_reb_chances"], validation["v1_pred_reb_chances"]
    )["mae"]
    scales = [max(v1_fga, 1e-9), max(v1_reb, 1e-9)]

    rows = []
    for trial_number, candidate in enumerate(candidate_dicts):
        parameters = TrustParameters(**candidate)
        tuned = apply_trust_parameters(
            validation,
            parameters,
            multiplier_bounds,
        )
        fga_mae = regression_metrics(
            validation["actual_fga"], tuned["tuned_pred_fga"]
        )["mae"]
        reb_mae = regression_metrics(
            validation["actual_reb_chances"], tuned["tuned_pred_reb_chances"]
        )["mae"]
        objective = np.nanmean([fga_mae / scales[0], reb_mae / scales[1]])
        rows.append(
            {
                "trial": trial_number,
                **candidate,
                "fga_mae": fga_mae,
                "reb_chances_mae": reb_mae,
                "objective": objective,
            }
        )
    trials = pd.DataFrame(rows).sort_values(["objective", "trial"]).reset_index(drop=True)
    best = trials.iloc[0]
    return TrustTuningResult(
        best_parameters=TrustParameters(
            low_trust_weight=float(best["low_trust_weight"]),
            high_trust_weight=float(best["high_trust_weight"]),
            max_trust_weight=float(best["max_trust_weight"]),
        ),
        best_objective=float(best["objective"]),
        trials=trials,
    )


def _workflow_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "top_n_similar": int(parameters.get("top_n_similar", 10)),
        "lookback_days": int(parameters.get("lookback_days", 30)),
        "min_games": int(parameters.get("min_games", 5)),
        "min_minutes_ratio": float(parameters.get("min_minutes_ratio", 0.75)),
        "position_weight": float(parameters.get("position_weight", 0.35)),
        "require_same_starter_flag": bool(
            parameters.get("require_same_starter_flag", False)
        ),
        "feature_weights": {
            "minutes": 1.0,
            "fga_per_min": float(parameters.get("fga_per_min_weight", 1.25)),
            "reb_chances_per_min": float(
                parameters.get("reb_chances_per_min_weight", 1.5)
            ),
            "usage_rate": float(parameters.get("usage_rate_weight", 1.25)),
            "touches_per_min": float(
                parameters.get("touches_per_min_weight", 1.25)
            ),
            "starter_rate": float(parameters.get("starter_rate_weight", 0.75)),
        },
    }


def _workflow_score(
    candidate: TrainingDatasetResult,
    reference_rows: pd.DataFrame,
    *,
    coverage_penalty: float,
) -> dict[str, float]:
    keys = ["player_id", "game_id"]
    candidate_rows = candidate.rows.merge(
        reference_rows[keys], on=keys, how="inner"
    )
    reference_common = reference_rows.merge(
        candidate_rows[keys], on=keys, how="inner"
    )
    coverage = len(candidate_rows) / len(reference_rows) if len(reference_rows) else 0.0
    fga_mae = regression_metrics(
        candidate_rows["actual_fga"], candidate_rows["v1_pred_fga"]
    )["mae"]
    reb_mae = regression_metrics(
        candidate_rows["actual_reb_chances"],
        candidate_rows["v1_pred_reb_chances"],
    )["mae"]
    reference_fga = regression_metrics(
        reference_common["actual_fga"], reference_common["v1_pred_fga"]
    )["mae"]
    reference_reb = regression_metrics(
        reference_common["actual_reb_chances"],
        reference_common["v1_pred_reb_chances"],
    )["mae"]
    if pd.isna(fga_mae) or pd.isna(reb_mae):
        objective = np.inf
    else:
        normalized = np.mean(
            [fga_mae / max(reference_fga, 1e-9), reb_mae / max(reference_reb, 1e-9)]
        )
        objective = normalized + coverage_penalty * (1 - coverage)
    return {
        "coverage": coverage,
        "fga_mae": fga_mae,
        "reb_chances_mae": reb_mae,
        "objective": objective,
    }


def tune_workflow_parameters(
    player_queries: Iterable[str | int],
    start_date,
    end_date,
    *,
    parameter_space: Mapping[str, Sequence[Any]] | None = None,
    n_iter: int = 20,
    coverage_penalty: float = 2.0,
    random_state: int = 42,
    build_kwargs: Mapping[str, Any] | None = None,
) -> WorkflowTuningResult:
    """Rerun V1 to tune neighbor selection and profile weights.

    This is the expensive tuning stage because every candidate changes which
    comparison players are selected. Cached NBA responses make later trials
    substantially cheaper than the first one.
    """
    players = list(player_queries)
    if not players:
        raise ValueError("At least one player is required for workflow tuning.")
    build_kwargs = dict(build_kwargs or {})
    reference = build_training_dataset(players, start_date, end_date, **build_kwargs)
    if reference.rows.empty:
        raise ValueError("The default workflow did not produce any reference rows.")

    candidates = _sample_candidates(
        parameter_space or WORKFLOW_PARAMETER_SPACE,
        n_iter,
        random_state,
        include_default=True,
    )
    rows: list[dict[str, Any]] = []
    for trial_number, parameters in enumerate(candidates):
        if trial_number == 0:
            result = reference
            resolved = _workflow_kwargs({})
        else:
            resolved = _workflow_kwargs(parameters)
            trial_kwargs = dict(build_kwargs)
            trial_kwargs.update(resolved)
            result = build_training_dataset(
                players,
                start_date,
                end_date,
                **trial_kwargs,
            )
        score = _workflow_score(
            result,
            reference.rows,
            coverage_penalty=coverage_penalty,
        )
        rows.append(
            {
                "trial": trial_number,
                "parameters_json": json.dumps(resolved, sort_keys=True),
                "rows_built": len(result.rows),
                "rows_skipped": len(result.skipped),
                **score,
            }
        )

    trials = pd.DataFrame(rows).sort_values(["objective", "trial"]).reset_index(drop=True)
    usable = trials[np.isfinite(trials["objective"])]
    if usable.empty:
        raise ValueError("No workflow trial produced a usable score.")
    best = usable.iloc[0]
    return WorkflowTuningResult(
        best_parameters=json.loads(best["parameters_json"]),
        best_objective=float(best["objective"]),
        reference_rows=len(reference.rows),
        trials=trials,
    )


def write_tuning_summary(
    model_results: Mapping[str, ModelTuningResult],
    output_path: str | Path,
    *,
    trust_result: TrustTuningResult | None = None,
    workflow_result: WorkflowTuningResult | None = None,
    workflow_parameters: Mapping[str, Any] | None = None,
    multiplier_bounds: MultiplierBounds | Mapping[str, Any] | None = None,
) -> Path:
    """Write reusable best parameters and validation scores to JSON."""
    payload: dict[str, Any] = {
        "model_type": next(iter(model_results.values())).model_type
        if model_results
        else None,
        "model_parameters": {
            target: result.best_parameters for target, result in model_results.items()
        },
        "model_scores": {
            target: {
                "v1_mae": result.mean_v1_mae,
                "v2_mae": result.best_v2_mae,
            }
            for target, result in model_results.items()
        },
    }
    if trust_result is not None:
        payload["trust_parameters"] = asdict(trust_result.best_parameters)
        payload["trust_objective"] = trust_result.best_objective
    if multiplier_bounds is not None:
        payload["multiplier_bounds"] = MultiplierBounds.from_value(
            multiplier_bounds
        ).to_dict()
    if workflow_result is not None:
        payload["workflow_parameters"] = workflow_result.best_parameters
        payload["workflow_objective"] = workflow_result.best_objective
    elif workflow_parameters is not None:
        payload["workflow_parameters"] = dict(workflow_parameters)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output

"""Residual-regression models that correct, rather than replace, V1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    CATEGORICAL_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    residual_column: str
    actual_column: str
    v1_prediction_column: str
    correction_column: str
    v2_prediction_column: str


TARGET_SPECS = {
    "fga": TargetSpec(
        name="fga",
        residual_column="fga_residual",
        actual_column="actual_fga",
        v1_prediction_column="v1_pred_fga",
        correction_column="fga_correction",
        v2_prediction_column="v2_pred_fga",
    ),
    "reb_chances": TargetSpec(
        name="reb_chances",
        residual_column="reb_chances_residual",
        actual_column="actual_reb_chances",
        v1_prediction_column="v1_pred_reb_chances",
        correction_column="reb_chances_correction",
        v2_prediction_column="v2_pred_reb_chances",
    ),
}

DEFAULT_MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "elastic_net": {
        "alpha": 0.05,
        "l1_ratio": 0.25,
        "max_iter": 20000,
        "tol": 0.001,
    },
    "hist_gradient_boosting": {
        "loss": "absolute_error",
        "learning_rate": 0.05,
        "max_iter": 250,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "early_stopping": False,
    },
}

SUPPORTED_MODEL_TYPES = tuple(DEFAULT_MODEL_PARAMETERS) + ("pytorch_multitask",)


@dataclass
class ResidualModel:
    """One fitted correction model and the schema used to train it."""

    target: str
    model_type: str
    estimator: Pipeline
    feature_columns: list[str]
    parameters: dict[str, Any]
    training_rows: int
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None

    def predict_correction(self, frame: pd.DataFrame) -> pd.Series:
        features = prepare_model_frame(frame, self.feature_columns)
        values = self.estimator.predict(features)
        return pd.Series(values, index=frame.index, dtype="float64")


@dataclass
class V2ModelBundle:
    """The fitted FGA and rebound-chance correction models."""

    models: dict[str, ResidualModel]
    model_type: str
    trust_parameters: dict[str, Any] | None = None
    multiplier_bounds: dict[str, Any] | None = None
    workflow_parameters: dict[str, Any] | None = None
    multitask_model: Any | None = None
    conversion_model: Any | None = None
    release_metadata: dict[str, Any] | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return predict_v2(self, frame)


def get_target_spec(target: str) -> TargetSpec:
    try:
        return TARGET_SPECS[target]
    except KeyError as exc:
        raise ValueError(
            f"Unknown target {target!r}. Choose from {sorted(TARGET_SPECS)}."
        ) from exc


def prepare_model_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return model inputs with stable numeric and categorical dtypes."""
    columns = list(feature_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Training data is missing model features: {missing}")

    result = frame[columns].copy()
    for column in set(columns).intersection(NUMERIC_FEATURE_COLUMNS):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in set(columns).intersection(CATEGORICAL_FEATURE_COLUMNS):
        result[column] = (
            result[column].astype("string").fillna("UNKNOWN").astype(object)
        )
    return result


def _build_preprocessor(model_type: str, feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric = [column for column in feature_columns if column in NUMERIC_FEATURE_COLUMNS]
    categorical = [
        column for column in feature_columns if column in CATEGORICAL_FEATURE_COLUMNS
    ]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if model_type == "elastic_net":
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("At least one model feature is required.")
    return ColumnTransformer(transformers, remainder="drop")


def build_estimator(
    model_type: str = "hist_gradient_boosting",
    *,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    parameters: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted preprocessing and residual-regression pipeline."""
    if model_type not in DEFAULT_MODEL_PARAMETERS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choose from {sorted(DEFAULT_MODEL_PARAMETERS)}."
        )

    resolved = dict(DEFAULT_MODEL_PARAMETERS[model_type])
    if parameters:
        resolved.update(parameters)

    if model_type == "elastic_net":
        model = ElasticNet(random_state=random_state, **resolved)
    else:
        model = HistGradientBoostingRegressor(random_state=random_state, **resolved)

    return Pipeline(
        [
            ("preprocess", _build_preprocessor(model_type, feature_columns)),
            ("model", model),
        ]
    )


def fit_residual_model(
    training_df: pd.DataFrame,
    target: str,
    *,
    model_type: str = "hist_gradient_boosting",
    parameters: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    min_rows: int = 30,
    random_state: int = 42,
) -> ResidualModel:
    """Fit one model to the historical difference between actual and V1."""
    spec = get_target_spec(target)
    required = [spec.residual_column, spec.v1_prediction_column]
    missing = [column for column in required if column not in training_df.columns]
    if missing:
        raise KeyError(f"Training data is missing target columns: {missing}")

    residual = pd.to_numeric(training_df[spec.residual_column], errors="coerce")
    v1_prediction = pd.to_numeric(
        training_df[spec.v1_prediction_column], errors="coerce"
    )
    valid = residual.notna() & v1_prediction.notna()
    fit_frame = training_df.loc[valid].copy()
    y = residual.loc[valid]
    if len(fit_frame) < min_rows:
        raise ValueError(
            f"Target {target!r} has {len(fit_frame)} usable rows; at least {min_rows} are required."
        )

    columns = list(feature_columns)
    estimator = build_estimator(
        model_type,
        feature_columns=columns,
        parameters=parameters,
        random_state=random_state,
    )
    estimator.fit(prepare_model_frame(fit_frame, columns), y)

    dates = pd.to_datetime(fit_frame.get("game_date"), errors="coerce")
    resolved_parameters = dict(DEFAULT_MODEL_PARAMETERS[model_type])
    if parameters:
        resolved_parameters.update(parameters)
    return ResidualModel(
        target=target,
        model_type=model_type,
        estimator=estimator,
        feature_columns=columns,
        parameters=resolved_parameters,
        training_rows=len(fit_frame),
        train_start=dates.min() if dates.notna().any() else None,
        train_end=dates.max() if dates.notna().any() else None,
    )


def fit_residual_models(
    training_df: pd.DataFrame,
    *,
    model_type: str = "hist_gradient_boosting",
    parameters_by_target: Mapping[str, Mapping[str, Any]] | None = None,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    min_rows: int = 30,
    random_state: int = 42,
) -> V2ModelBundle:
    """Fit both opportunity correction models."""
    parameters_by_target = parameters_by_target or {}
    if model_type == "pytorch_multitask":
        from .neural import fit_multitask_residual_model

        if "shared" in parameters_by_target:
            shared_parameters = parameters_by_target["shared"]
        else:
            target_parameters = [
                dict(parameters_by_target[target])
                for target in TARGET_SPECS
                if target in parameters_by_target
            ]
            if target_parameters and any(
                parameters != target_parameters[0]
                for parameters in target_parameters[1:]
            ):
                raise ValueError(
                    "PyTorch multi-task parameters must be shared by both targets."
                )
            shared_parameters = target_parameters[0] if target_parameters else None
        multitask_model = fit_multitask_residual_model(
            training_df,
            parameters=shared_parameters,
            feature_columns=feature_columns,
            min_rows=min_rows,
            random_state=random_state,
        )
        return V2ModelBundle(
            models={},
            model_type=model_type,
            multitask_model=multitask_model,
        )
    if model_type not in DEFAULT_MODEL_PARAMETERS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choose from {sorted(SUPPORTED_MODEL_TYPES)}."
        )
    models = {
        target: fit_residual_model(
            training_df,
            target,
            model_type=model_type,
            parameters=parameters_by_target.get(target),
            feature_columns=feature_columns,
            min_rows=min_rows,
            random_state=random_state,
        )
        for target in TARGET_SPECS
    }
    return V2ModelBundle(models=models, model_type=model_type)


def fit_model_bundle(
    training_df: pd.DataFrame,
    *,
    model_type: str = "hist_gradient_boosting",
    parameters_by_target: Mapping[str, Mapping[str, Any]] | None = None,
    conversion_parameters: Mapping[str, float] | None = None,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    min_rows: int = 30,
    random_state: int = 42,
) -> V2ModelBundle:
    """Fit opportunity residuals and an optional rebound-conversion model."""
    bundle = fit_residual_models(
        training_df,
        model_type=model_type,
        parameters_by_target=parameters_by_target,
        feature_columns=feature_columns,
        min_rows=min_rows,
        random_state=random_state,
    )
    if conversion_parameters is not None:
        from .conversion import fit_conversion_adjustment_model

        bundle.conversion_model = fit_conversion_adjustment_model(
            training_df,
            parameters=conversion_parameters,
            min_rows=min_rows,
            random_state=random_state,
        )
    return bundle


def predict_v2(bundle: V2ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply learned corrections and derive rebounds from corrected chances."""
    if frame.empty:
        return pd.DataFrame()

    identifiers = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
    result = frame[identifiers].copy()
    multitask_corrections = None
    if bundle.multitask_model is not None:
        multitask_corrections = bundle.multitask_model.predict_corrections(frame)
    for target, spec in TARGET_SPECS.items():
        v1 = pd.to_numeric(frame[spec.v1_prediction_column], errors="coerce")
        if multitask_corrections is not None:
            correction = multitask_corrections[target]
        else:
            if target not in bundle.models:
                raise KeyError(f"Model bundle does not contain target {target!r}.")
            correction = bundle.models[target].predict_correction(frame)
        result[spec.v1_prediction_column] = v1
        result[spec.correction_column] = correction
        result[spec.v2_prediction_column] = (v1 + correction).clip(lower=0)

    baseline_conversion = pd.to_numeric(
        frame["baseline_reb_conversion"], errors="coerce"
    ).clip(lower=0, upper=1)
    conversion_model = getattr(bundle, "conversion_model", None)
    if conversion_model is None:
        predicted_conversion = baseline_conversion
    else:
        predicted_conversion = pd.to_numeric(
            conversion_model.predict_rate(frame), errors="coerce"
        ).clip(lower=0, upper=1)
    result["baseline_reb_conversion"] = baseline_conversion
    result["pred_reb_conversion"] = predicted_conversion
    result["reb_conversion_correction"] = (
        predicted_conversion - baseline_conversion
    )
    result["v1_pred_reb"] = pd.to_numeric(frame["v1_pred_reb"], errors="coerce")
    result["v2_pred_reb"] = (
        result["v2_pred_reb_chances"] * predicted_conversion
    )

    for column in [
        "frozen_v1_pred_fga",
        "frozen_v1_pred_reb_chances",
        "frozen_v1_pred_reb",
    ]:
        if column in frame.columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["actual_fga", "actual_reb_chances", "actual_reb"]:
        if column in frame.columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
    return result


def save_model_bundle(bundle: V2ModelBundle, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_model_bundle(path: str | Path) -> V2ModelBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, V2ModelBundle):
        raise TypeError("The model artifact is not a V2ModelBundle.")
    return bundle

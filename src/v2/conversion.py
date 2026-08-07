"""Leakage-safe rebound-conversion estimation for V2 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONVERSION_NUMERIC_FEATURES = [
    "baseline_reb_conversion",
    "conversion_exposure",
    "tracking_games_used",
    "baseline_minutes",
    "baseline_reb",
    "baseline_reb_chances",
    "baseline_reb_chances_per_min",
    "profile_minutes",
    "profile_reb_chances_per_min",
    "profile_usage_rate",
    "profile_touches_per_min",
    "profile_starter_rate",
    "n_similar_players",
    "similarity_score_mean",
    "similarity_score_std",
    "numeric_distance_mean",
    "position_penalty_mean",
    "raw_reb_chances_multiplier",
    "reb_chances_n_samples",
    "v1_pred_minutes",
]
CONVERSION_CATEGORICAL_FEATURES = ["profile_listed_position"]
CONVERSION_FEATURES = (
    CONVERSION_NUMERIC_FEATURES + CONVERSION_CATEGORICAL_FEATURES
)


@dataclass(frozen=True)
class ConversionPrior:
    global_rate: float
    position_rates: dict[str, float]


@dataclass
class ConversionAdjustmentModel:
    estimator: Pipeline
    parameters: dict[str, float]
    lower_bound: float
    upper_bound: float
    training_rows: int

    def predict_rate(self, frame: pd.DataFrame) -> pd.Series:
        prepared = prepare_conversion_frame(frame)
        correction = self.estimator.predict(
            prepare_conversion_features(prepared, CONVERSION_FEATURES)
        )
        baseline = prepared["baseline_reb_conversion"].to_numpy(dtype=float)
        rate = np.clip(
            baseline + correction,
            self.lower_bound,
            self.upper_bound,
        )
        return pd.Series(rate, index=frame.index, dtype="float64")


def prepare_conversion_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add recent chance exposure and optional postgame conversion targets."""
    required = [
        "baseline_reb_conversion",
        "baseline_reb_chances",
        "tracking_games_used",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Conversion data is missing columns: {missing}")

    result = frame.copy()
    numeric = set(CONVERSION_NUMERIC_FEATURES).intersection(result.columns)
    numeric.update(
        column
        for column in ["actual_reb", "actual_reb_chances"]
        if column in result
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["conversion_exposure"] = (
        result["baseline_reb_chances"] * result["tracking_games_used"]
    ).clip(lower=0)
    if {"actual_reb", "actual_reb_chances"}.issubset(result.columns):
        positive = result["actual_reb_chances"] > 0
        result["actual_reb_conversion"] = np.where(
            positive,
            result["actual_reb"] / result["actual_reb_chances"],
            np.nan,
        )
        result["conversion_target"] = (
            result["actual_reb_conversion"].clip(lower=0, upper=1)
            - result["baseline_reb_conversion"]
        )
    return result


def fit_conversion_prior(frame: pd.DataFrame) -> ConversionPrior:
    """Estimate global and listed-position conversion priors from prior rows."""
    prepared = prepare_conversion_frame(frame)
    required = ["actual_reb", "actual_reb_chances"]
    missing = [column for column in required if column not in prepared]
    if missing:
        raise KeyError(f"Conversion prior requires columns: {missing}")

    valid = prepared[
        prepared["actual_reb"].notna()
        & prepared["actual_reb_chances"].notna()
        & (prepared["actual_reb_chances"] > 0)
    ].copy()
    if valid.empty:
        raise ValueError("No positive rebound-chance rows are available for a prior.")

    global_rate = float(
        valid["actual_reb"].sum() / valid["actual_reb_chances"].sum()
    )
    global_rate = float(np.clip(global_rate, 0, 1))
    positions = (
        valid["profile_listed_position"]
        .astype("string")
        .fillna("UNKNOWN")
    )
    grouped = valid.assign(_position=positions).groupby("_position", sort=True)
    position_rates = {
        str(position): float(
            np.clip(
                group["actual_reb"].sum()
                / group["actual_reb_chances"].sum(),
                0,
                1,
            )
        )
        for position, group in grouped
        if group["actual_reb_chances"].sum() > 0
    }
    return ConversionPrior(global_rate=global_rate, position_rates=position_rates)


def predict_shrunken_conversion(
    frame: pd.DataFrame,
    prior: ConversionPrior,
    *,
    prior_strength: float,
) -> pd.Series:
    """Shrink a recent player rate toward a position prior by chance exposure."""
    if prior_strength < 0:
        raise ValueError("prior_strength cannot be negative.")
    prepared = prepare_conversion_frame(frame)
    baseline = prepared["baseline_reb_conversion"].clip(lower=0, upper=1)
    exposure = prepared["conversion_exposure"].fillna(0).clip(lower=0)
    if "profile_listed_position" in prepared:
        positions = (
            prepared["profile_listed_position"]
            .astype("string")
            .fillna("UNKNOWN")
        )
    else:
        positions = pd.Series("UNKNOWN", index=prepared.index, dtype="string")
    prior_rate = positions.map(prior.position_rates).fillna(prior.global_rate)
    denominator = exposure + float(prior_strength)
    shrunken = np.where(
        denominator > 0,
        (
            baseline * exposure
            + prior_rate.astype(float) * float(prior_strength)
        )
        / denominator,
        baseline,
    )
    return pd.Series(np.clip(shrunken, 0, 1), index=frame.index, dtype="float64")


def prepare_conversion_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] = CONVERSION_FEATURES,
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise KeyError(f"Conversion model is missing features: {missing}")
    result = frame[list(feature_columns)].copy()
    for column in set(feature_columns).intersection(CONVERSION_NUMERIC_FEATURES):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in set(feature_columns).intersection(
        CONVERSION_CATEGORICAL_FEATURES
    ):
        result[column] = (
            result[column].astype("string").fillna("UNKNOWN").astype(object)
        )
    return result


def conversion_bounds(frame: pd.DataFrame) -> tuple[float, float]:
    """Use training-only player-rate percentiles as prediction bounds."""
    values = pd.to_numeric(
        frame["baseline_reb_conversion"], errors="coerce"
    ).dropna()
    if values.empty:
        return 0.0, 1.0
    lower = float(np.clip(values.quantile(0.01), 0, 1))
    upper = float(np.clip(values.quantile(0.99), 0, 1))
    if lower >= upper:
        return 0.0, 1.0
    return lower, upper


def _build_conversion_estimator(
    parameters: Mapping[str, float],
    *,
    random_state: int,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, CONVERSION_NUMERIC_FEATURES),
            ("categorical", categorical, CONVERSION_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = ElasticNet(
        alpha=float(parameters["alpha"]),
        l1_ratio=float(parameters["l1_ratio"]),
        max_iter=20000,
        tol=0.001,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_conversion_adjustment_model(
    frame: pd.DataFrame,
    *,
    parameters: Mapping[str, float],
    min_rows: int = 30,
    random_state: int = 42,
) -> ConversionAdjustmentModel:
    """Fit a chance-weighted Elastic Net adjustment to the recent player rate."""
    prepared = prepare_conversion_frame(frame)
    valid = (
        prepared["conversion_target"].notna()
        & prepared["actual_reb_chances"].notna()
        & (prepared["actual_reb_chances"] > 0)
    )
    fit_frame = prepared.loc[valid].copy()
    if len(fit_frame) < min_rows:
        raise ValueError(
            f"Conversion model has {len(fit_frame)} usable rows; "
            f"at least {min_rows} are required."
        )
    estimator = _build_conversion_estimator(parameters, random_state=random_state)
    weights = fit_frame["actual_reb_chances"].clip(lower=1).to_numpy(dtype=float)
    estimator.fit(
        prepare_conversion_features(fit_frame),
        fit_frame["conversion_target"],
        model__sample_weight=weights,
    )
    lower, upper = conversion_bounds(fit_frame)
    return ConversionAdjustmentModel(
        estimator=estimator,
        parameters={
            "alpha": float(parameters["alpha"]),
            "l1_ratio": float(parameters["l1_ratio"]),
        },
        lower_bound=lower,
        upper_bound=upper,
        training_rows=len(fit_frame),
    )

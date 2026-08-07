"""PyTorch multi-task residual correction for FGA and rebound chances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)


NEURAL_TARGETS = ("fga", "reb_chances")
NEURAL_RESIDUAL_COLUMNS = ("fga_residual", "reb_chances_residual")

DEFAULT_NEURAL_PARAMETERS: dict[str, Any] = {
    "hidden_size_1": 64,
    "hidden_size_2": 32,
    "dropout": 0.15,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "max_epochs": 120,
    "patience": 15,
    "validation_fraction": 0.15,
    "huber_delta": 1.0,
    "min_delta": 0.0001,
    "gradient_clip": 5.0,
    "ensemble_size": 5,
}


class MultiTaskResidualNetwork(nn.Module):
    """Shared representation with one linear output head per residual."""

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.LayerNorm(hidden_size_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.LayerNorm(hidden_size_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fga_head = nn.Linear(hidden_size_2, 1)
        self.reb_chances_head = nn.Linear(hidden_size_2, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.shared(features)
        return torch.cat(
            [self.fga_head(shared), self.reb_chances_head(shared)],
            dim=1,
        )


@dataclass
class TorchMultiTaskResidualModel:
    """Fitted preprocessing, network ensemble, and residual scaling metadata."""

    preprocessor: ColumnTransformer
    networks: list[MultiTaskResidualNetwork]
    feature_columns: list[str]
    parameters: dict[str, Any]
    target_means: np.ndarray
    target_scales: np.ndarray
    training_rows: int
    target_training_rows: dict[str, int]
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    selected_epochs: int
    validation_loss: float | None

    def predict_corrections(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(index=frame.index, columns=NEURAL_TARGETS)
        prepared = _prepare_feature_frame(frame, self.feature_columns)
        transformed = np.asarray(
            self.preprocessor.transform(prepared),
            dtype=np.float32,
        )
        features = torch.from_numpy(transformed)
        predictions = []
        with torch.no_grad():
            for network in self.networks:
                network.eval()
                predictions.append(network(features).cpu().numpy())
        standardized = np.mean(predictions, axis=0)
        corrections = standardized * self.target_scales + self.target_means
        return pd.DataFrame(corrections, index=frame.index, columns=NEURAL_TARGETS)


def resolve_neural_parameters(
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(DEFAULT_NEURAL_PARAMETERS)
    if parameters:
        resolved.update(parameters)

    integer_keys = [
        "hidden_size_1",
        "hidden_size_2",
        "batch_size",
        "max_epochs",
        "patience",
        "ensemble_size",
    ]
    for key in integer_keys:
        resolved[key] = int(resolved[key])
        if resolved[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    float_keys = [
        "dropout",
        "learning_rate",
        "weight_decay",
        "validation_fraction",
        "huber_delta",
        "min_delta",
        "gradient_clip",
    ]
    for key in float_keys:
        resolved[key] = float(resolved[key])
    if not 0 <= resolved["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1).")
    if not 0 <= resolved["validation_fraction"] < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5).")
    for key in ["learning_rate", "huber_delta", "gradient_clip"]:
        if resolved[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if resolved["weight_decay"] < 0 or resolved["min_delta"] < 0:
        raise ValueError("weight_decay and min_delta cannot be negative.")
    return resolved


def _prepare_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
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


def _build_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric = [
        column for column in feature_columns if column in NUMERIC_FEATURE_COLUMNS
    ]
    categorical = [
        column for column in feature_columns if column in CATEGORICAL_FEATURE_COLUMNS
    ]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                keep_empty_features=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
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


def _target_array(frame: pd.DataFrame) -> np.ndarray:
    missing = [
        column for column in NEURAL_RESIDUAL_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Training data is missing target columns: {missing}")
    return np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            for column in NEURAL_RESIDUAL_COLUMNS
        ]
    )


def _target_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(values, axis=0)
    scales = np.nanstd(values, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)
    means = np.where(np.isfinite(means), means, 0.0)
    return means.astype(np.float32), scales.astype(np.float32)


def _scaled_targets(
    values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values)
    scaled = np.where(mask, (values - means) / scales, 0.0)
    return scaled.astype(np.float32), mask.astype(np.float32)


def _chronological_masks(
    frame: pd.DataFrame,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if validation_fraction <= 0:
        return np.ones(len(frame), dtype=bool), np.zeros(len(frame), dtype=bool)
    dates = pd.to_datetime(frame["game_date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("game_date contains missing or invalid dates.")
    unique_dates = pd.Index(dates.unique()).sort_values()
    if len(unique_dates) < 2:
        raise ValueError("At least two dates are required for neural early stopping.")
    validation_dates = min(
        max(1, int(np.ceil(len(unique_dates) * validation_fraction))),
        len(unique_dates) - 1,
    )
    validation_start = pd.Timestamp(unique_dates[-validation_dates])
    train_mask = (dates < validation_start).to_numpy()
    validation_mask = (dates >= validation_start).to_numpy()
    return train_mask, validation_mask


def _loader(
    features: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
        torch.from_numpy(target_mask.astype(np.float32, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    losses = nn.functional.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=delta,
    )
    denominator = target_mask.sum().clamp_min(1.0)
    return (losses * target_mask).sum() / denominator


def _train_network(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_target_mask: np.ndarray,
    *,
    parameters: Mapping[str, Any],
    seed: int,
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    epochs: int | None = None,
) -> tuple[MultiTaskResidualNetwork, int, float | None]:
    torch.manual_seed(seed)
    network = MultiTaskResidualNetwork(
        train_features.shape[1],
        int(parameters["hidden_size_1"]),
        int(parameters["hidden_size_2"]),
        float(parameters["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    train_loader = _loader(
        train_features,
        train_targets,
        train_target_mask,
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    validation_loader = None
    if validation_data is not None:
        validation_loader = _loader(
            *validation_data,
            batch_size=int(parameters["batch_size"]),
            shuffle=False,
            seed=seed,
        )

    epoch_limit = int(epochs or parameters["max_epochs"])
    best_epoch = epoch_limit
    best_loss = np.inf
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epoch_limit + 1):
        network.train()
        for batch_features, batch_targets, batch_mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = network(batch_features)
            loss = _masked_huber_loss(
                prediction,
                batch_targets,
                batch_mask,
                delta=float(parameters["huber_delta"]),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                network.parameters(),
                float(parameters["gradient_clip"]),
            )
            optimizer.step()

        if validation_loader is None:
            continue
        network.eval()
        validation_loss = 0.0
        validation_weight = 0.0
        with torch.no_grad():
            for batch_features, batch_targets, batch_mask in validation_loader:
                prediction = network(batch_features)
                batch_loss = _masked_huber_loss(
                    prediction,
                    batch_targets,
                    batch_mask,
                    delta=float(parameters["huber_delta"]),
                )
                weight = float(batch_mask.sum().item())
                validation_loss += float(batch_loss.item()) * weight
                validation_weight += weight
        validation_loss /= max(validation_weight, 1.0)
        if validation_loss < best_loss - float(parameters["min_delta"]):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(parameters["patience"]):
                break

    if best_state is not None:
        network.load_state_dict(best_state)
    network.cpu().eval()
    return network, best_epoch, None if not np.isfinite(best_loss) else float(best_loss)


def fit_multitask_residual_model(
    training_df: pd.DataFrame,
    *,
    parameters: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    min_rows: int = 30,
    random_state: int = 42,
) -> TorchMultiTaskResidualModel:
    """Fit a chronologically early-stopped multi-seed residual ensemble."""
    resolved = resolve_neural_parameters(parameters)
    targets = _target_array(training_df)
    target_counts = np.isfinite(targets).sum(axis=0)
    for target, count in zip(NEURAL_TARGETS, target_counts):
        if count < min_rows:
            raise ValueError(
                f"Target {target!r} has {count} usable rows; at least {min_rows} are required."
            )
    usable = np.isfinite(targets).any(axis=1)
    fit_frame = training_df.loc[usable].copy()
    targets = targets[usable]
    if len(fit_frame) < min_rows:
        raise ValueError(
            f"The multi-task model has {len(fit_frame)} usable rows; "
            f"at least {min_rows} are required."
        )

    columns = list(feature_columns)
    train_mask, validation_mask = _chronological_masks(
        fit_frame,
        resolved["validation_fraction"],
    )
    selected_epochs = int(resolved["max_epochs"])
    validation_loss = None
    if validation_mask.any():
        pilot_preprocessor = _build_preprocessor(columns)
        pilot_train_frame = fit_frame.iloc[np.flatnonzero(train_mask)]
        pilot_validation_frame = fit_frame.iloc[np.flatnonzero(validation_mask)]
        pilot_preprocessor.fit(_prepare_feature_frame(pilot_train_frame, columns))
        pilot_train_features = np.asarray(
            pilot_preprocessor.transform(
                _prepare_feature_frame(pilot_train_frame, columns)
            ),
            dtype=np.float32,
        )
        pilot_validation_features = np.asarray(
            pilot_preprocessor.transform(
                _prepare_feature_frame(pilot_validation_frame, columns)
            ),
            dtype=np.float32,
        )
        pilot_means, pilot_scales = _target_stats(targets[train_mask])
        pilot_train_targets, pilot_train_target_mask = _scaled_targets(
            targets[train_mask],
            pilot_means,
            pilot_scales,
        )
        pilot_validation_targets, pilot_validation_target_mask = _scaled_targets(
            targets[validation_mask],
            pilot_means,
            pilot_scales,
        )
        _, selected_epochs, validation_loss = _train_network(
            pilot_train_features,
            pilot_train_targets,
            pilot_train_target_mask,
            parameters=resolved,
            seed=random_state,
            validation_data=(
                pilot_validation_features,
                pilot_validation_targets,
                pilot_validation_target_mask,
            ),
        )

    preprocessor = _build_preprocessor(columns)
    prepared = _prepare_feature_frame(fit_frame, columns)
    transformed = np.asarray(
        preprocessor.fit_transform(prepared),
        dtype=np.float32,
    )
    target_means, target_scales = _target_stats(targets)
    scaled_targets, target_mask = _scaled_targets(
        targets,
        target_means,
        target_scales,
    )
    networks = []
    for ensemble_index in range(int(resolved["ensemble_size"])):
        network, _, _ = _train_network(
            transformed,
            scaled_targets,
            target_mask,
            parameters=resolved,
            seed=random_state + 1009 * ensemble_index,
            epochs=selected_epochs,
        )
        networks.append(network)

    dates = pd.to_datetime(fit_frame.get("game_date"), errors="coerce")
    return TorchMultiTaskResidualModel(
        preprocessor=preprocessor,
        networks=networks,
        feature_columns=columns,
        parameters=resolved,
        target_means=target_means,
        target_scales=target_scales,
        training_rows=len(fit_frame),
        target_training_rows={
            target: int(count) for target, count in zip(NEURAL_TARGETS, target_counts)
        },
        train_start=dates.min() if dates.notna().any() else None,
        train_end=dates.max() if dates.notna().any() else None,
        selected_epochs=selected_epochs,
        validation_loss=validation_loss,
    )

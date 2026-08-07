"""Held-out feature importance and feature-group ablation for V2."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .evaluate import evaluate_model_bundle
from .features import MODEL_FEATURE_COLUMNS
from .models import (
    TARGET_SPECS,
    V2ModelBundle,
    fit_residual_models,
    get_target_spec,
    prepare_model_frame,
)


FEATURE_GROUPS = {
    "baseline": [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column.startswith("baseline_")
        or column
        in {
            "tracking_games_used",
            "usual_minutes",
            "min_minutes_threshold",
        }
    ],
    "profile": [column for column in MODEL_FEATURE_COLUMNS if column.startswith("profile_")],
    "similarity": [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column.startswith("similarity_")
        or column
        in {
            "n_similar_players",
            "numeric_distance_mean",
            "position_penalty_mean",
        }
    ],
    "matchup": [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column.startswith("raw_")
        or column.endswith("_multiplier")
        or column.endswith("_n_samples")
        or column.endswith("_trust_weight")
    ],
    "v1_prediction": [
        column for column in MODEL_FEATURE_COLUMNS if column.startswith("v1_pred_")
    ],
}


def permutation_importance_table(
    bundle: V2ModelBundle,
    validation_df: pd.DataFrame,
    *,
    targets: Sequence[str] = tuple(TARGET_SPECS),
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure raw-feature importance on dates unseen during model fitting."""
    if bundle.multitask_model is not None:
        return _multitask_permutation_importance(
            bundle,
            validation_df,
            targets=targets,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    rows: list[dict] = []
    for target in targets:
        if target not in bundle.models:
            raise KeyError(f"Model bundle does not contain target {target!r}.")
        model = bundle.models[target]
        spec = get_target_spec(target)
        y = pd.to_numeric(validation_df[spec.residual_column], errors="coerce")
        valid = y.notna()
        if valid.sum() < 2:
            continue
        frame = validation_df.loc[valid]
        result = permutation_importance(
            model.estimator,
            prepare_model_frame(frame, model.feature_columns),
            y.loc[valid],
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats,
            random_state=random_state,
        )
        for feature, mean, std in zip(
            model.feature_columns,
            result.importances_mean,
            result.importances_std,
        ):
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["target", "feature", "importance_mean", "importance_std", "rank"]
        )
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("target")["importance_mean"].rank(
        method="dense", ascending=False
    ).astype(int)
    return table.sort_values(["target", "rank", "feature"]).reset_index(drop=True)


def _multitask_permutation_importance(
    bundle: V2ModelBundle,
    validation_df: pd.DataFrame,
    *,
    targets: Sequence[str],
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Permutation importance using residual MAE for a shared-output model."""
    model = bundle.multitask_model
    rng = np.random.default_rng(random_state)
    base_corrections = model.predict_corrections(validation_df)
    rows = []
    for target in targets:
        spec = get_target_spec(target)
        residual = pd.to_numeric(
            validation_df[spec.residual_column], errors="coerce"
        )
        valid = residual.notna()
        if valid.sum() < 2:
            continue
        baseline_mae = float(
            (residual.loc[valid] - base_corrections.loc[valid, target]).abs().mean()
        )
        for feature in model.feature_columns:
            increases = []
            for _ in range(n_repeats):
                permuted = validation_df.copy()
                permuted[feature] = rng.permutation(
                    permuted[feature].to_numpy(copy=True)
                )
                correction = model.predict_corrections(permuted)[target]
                permuted_mae = float(
                    (residual.loc[valid] - correction.loc[valid]).abs().mean()
                )
                increases.append(permuted_mae - baseline_mae)
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "importance_mean": float(np.mean(increases)),
                    "importance_std": float(np.std(increases)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["target", "feature", "importance_mean", "importance_std", "rank"]
        )
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("target")["importance_mean"].rank(
        method="dense", ascending=False
    ).astype(int)
    return table.sort_values(["target", "rank", "feature"]).reset_index(drop=True)


def group_ablation_table(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    model_type: str = "hist_gradient_boosting",
    parameters_by_target: Mapping[str, Mapping] | None = None,
    feature_groups: Mapping[str, Sequence[str]] = FEATURE_GROUPS,
    min_rows: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Retrain after removing each feature group and measure held-out damage."""
    full_bundle = fit_residual_models(
        train_df,
        model_type=model_type,
        parameters_by_target=parameters_by_target,
        min_rows=min_rows,
        random_state=random_state,
    )
    full_summary = evaluate_model_bundle(full_bundle, validation_df).summary.set_index("stat")
    rows: list[dict] = []

    for group_name, group_columns in feature_groups.items():
        removed = set(group_columns)
        remaining = [column for column in MODEL_FEATURE_COLUMNS if column not in removed]
        if not remaining:
            continue
        ablated_bundle = fit_residual_models(
            train_df,
            model_type=model_type,
            parameters_by_target=parameters_by_target,
            feature_columns=remaining,
            min_rows=min_rows,
            random_state=random_state,
        )
        ablated_summary = evaluate_model_bundle(
            ablated_bundle, validation_df
        ).summary.set_index("stat")
        for target in TARGET_SPECS:
            full_mae = float(full_summary.loc[target, "v2_mae"])
            ablated_mae = float(ablated_summary.loc[target, "v2_mae"])
            rows.append(
                {
                    "feature_group": group_name,
                    "target": target,
                    "features_removed": len(removed.intersection(MODEL_FEATURE_COLUMNS)),
                    "full_v2_mae": full_mae,
                    "ablated_v2_mae": ablated_mae,
                    "mae_increase": ablated_mae - full_mae,
                    "mae_increase_pct": (
                        100 * (ablated_mae - full_mae) / full_mae
                        if full_mae > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target", "mae_increase"], ascending=[True, False]
    ).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No results."
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_importance_report(
    permutation_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 15,
) -> Path:
    """Write individual and grouped importance results to Markdown."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if permutation_table.empty:
        top_features = pd.DataFrame(
            columns=["target", "feature", "importance_mean", "importance_std", "rank"]
        )
    else:
        top_features = (
            permutation_table.sort_values(["target", "rank"])
            .groupby("target", group_keys=False)
            .head(top_n)
        )
    lines = [
        "# V2 Feature Importance",
        "",
        "Permutation importance is measured on held-out dates. Positive values mean",
        "shuffling the feature made residual predictions worse.",
        "",
        "## Individual Features",
        "",
        _markdown_table(top_features),
        "",
        "## Feature-Group Ablation",
        "",
        "Positive MAE increase means removing the group hurt V2 performance.",
        "",
        _markdown_table(ablation_table),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output

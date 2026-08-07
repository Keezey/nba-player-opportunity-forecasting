import pandas as pd
import pytest

import src.v2.tuning as tuning
from src.trust import MultiplierBounds
from src.v2.training_data import TrainingDatasetResult


def test_residual_model_tuning_uses_walk_forward_dates(v2_training_frame):
    result = tuning.tune_residual_model(
        v2_training_frame,
        "fga",
        model_type="elastic_net",
        parameter_space={"alpha": [0.005, 0.05], "l1_ratio": [0.1, 0.5]},
        n_iter=3,
        n_splits=2,
        min_rows=20,
    )

    assert result.target == "fga"
    assert result.best_v2_mae < result.mean_v1_mae
    assert len(result.trials) == 3


def test_trust_tuning_returns_monotonic_weights(v2_training_frame):
    result = tuning.tune_trust_parameters(
        v2_training_frame,
        n_iter=5,
        n_splits=2,
    )
    parameters = result.best_parameters

    assert parameters.low_trust_weight <= parameters.high_trust_weight
    assert parameters.high_trust_weight <= parameters.max_trust_weight
    weights = tuning.trust_weights(pd.Series([0, 5, 10, 15]), parameters)
    assert weights.is_monotonic_increasing

    rebased = tuning.rebase_training_frame_with_trust(
        v2_training_frame, parameters
    )
    assert "frozen_v1_pred_fga" in rebased.columns
    assert rebased["fga_residual"].equals(
        rebased["actual_fga"] - rebased["v1_pred_fga"]
    )


def test_trust_rebase_applies_bounds_before_shrinkage(v2_training_frame):
    frame = v2_training_frame.iloc[:2].copy()
    frame["raw_fga_multiplier"] = [10.0, 0.1]
    frame["fga_n_samples"] = 15
    bounds = MultiplierBounds.fixed(0.5, 1.5)

    tuned = tuning.apply_trust_parameters(
        frame,
        tuning.TrustParameters(),
        bounds,
    )

    assert tuned["fga_bounded_raw_multiplier"].tolist() == [1.5, 0.5]
    assert tuned["fga_multiplier_was_bounded"].tolist() == [True, True]
    assert tuned["fga_effective_multiplier"].tolist() == pytest.approx(
        [1.475, 0.525]
    )


def test_workflow_tuning_penalizes_missing_coverage(monkeypatch, v2_training_frame):
    reference_rows = v2_training_frame.iloc[:20].copy()
    better_rows = reference_rows.copy()
    better_rows["v1_pred_fga"] = better_rows["actual_fga"]
    better_rows["v1_pred_reb_chances"] = better_rows["actual_reb_chances"]
    calls = []

    def fake_build(*args, **kwargs):
        calls.append(kwargs)
        rows = reference_rows if len(calls) == 1 else better_rows
        return TrainingDatasetResult(rows=rows, skipped=pd.DataFrame())

    monkeypatch.setattr(tuning, "build_training_dataset", fake_build)
    result = tuning.tune_workflow_parameters(
        ["Target"],
        "2026-01-01",
        "2026-02-01",
        parameter_space={
            "top_n_similar": [5],
            "lookback_days": [30],
        },
        n_iter=2,
    )

    assert result.best_parameters["top_n_similar"] == 5
    assert result.best_objective == 0.0

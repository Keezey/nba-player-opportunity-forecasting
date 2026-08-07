import pandas as pd

from src.v2.evaluate import chronological_split, evaluate_model_bundle
from src.v2.models import (
    fit_residual_models,
    load_model_bundle,
    save_model_bundle,
)
from src.v2.neural import resolve_neural_parameters
from src.v2.tuning import tune_multitask_residual_models


SMALL_NEURAL_PARAMETERS = {
    "hidden_size_1": 16,
    "hidden_size_2": 8,
    "dropout": 0.05,
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "max_epochs": 6,
    "patience": 2,
    "validation_fraction": 0.20,
    "ensemble_size": 2,
}


def test_neural_parameter_validation():
    parameters = resolve_neural_parameters(SMALL_NEURAL_PARAMETERS)
    assert parameters["hidden_size_1"] == 16
    assert parameters["ensemble_size"] == 2

    try:
        resolve_neural_parameters({"dropout": 1.0})
    except ValueError as exc:
        assert "dropout" in str(exc)
    else:
        raise AssertionError("Expected invalid dropout to be rejected.")


def test_multitask_model_predicts_and_round_trips(v2_training_frame, tmp_path):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    bundle = fit_residual_models(
        train,
        model_type="pytorch_multitask",
        parameters_by_target={"shared": SMALL_NEURAL_PARAMETERS},
        min_rows=20,
        random_state=7,
    )

    assert not bundle.models
    assert bundle.multitask_model is not None
    assert len(bundle.multitask_model.networks) == 2
    predictions = bundle.predict(test)
    assert predictions[["v2_pred_fga", "v2_pred_reb_chances"]].notna().all().all()
    assert (predictions[["v2_pred_fga", "v2_pred_reb_chances"]] >= 0).all().all()

    evaluation = evaluate_model_bundle(bundle, test)
    assert evaluation.summary["stat"].tolist() == ["fga", "reb_chances", "reb"]

    path = save_model_bundle(bundle, tmp_path / "neural.joblib")
    loaded = load_model_bundle(path)
    loaded_predictions = loaded.predict(test)
    pd.testing.assert_series_equal(
        predictions["v2_pred_fga"],
        loaded_predictions["v2_pred_fga"],
    )
    pd.testing.assert_series_equal(
        predictions["v2_pred_reb_chances"],
        loaded_predictions["v2_pred_reb_chances"],
    )


def test_multitask_tuning_returns_one_shared_parameter_set(v2_training_frame):
    results = tune_multitask_residual_models(
        v2_training_frame,
        parameter_space={
            "hidden_size_1": [16],
            "hidden_size_2": [8],
            "dropout": [0.05],
            "learning_rate": [0.003],
            "weight_decay": [0.0001],
            "batch_size": [32],
        },
        n_iter=1,
        n_splits=2,
        min_rows=20,
        random_state=7,
        tuning_max_epochs=3,
        tuning_patience=1,
        final_ensemble_size=2,
    )

    assert set(results) == {"fga", "reb_chances"}
    assert results["fga"].best_parameters == results["reb_chances"].best_parameters
    assert results["fga"].best_parameters["ensemble_size"] == 2
    assert len(results["fga"].trials) == 1

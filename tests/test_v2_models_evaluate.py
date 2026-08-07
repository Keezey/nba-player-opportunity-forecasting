import pandas as pd

from src.v2.evaluate import chronological_split, evaluate_model_bundle
from src.v2.models import (
    fit_model_bundle,
    fit_residual_models,
    load_model_bundle,
    save_model_bundle,
)


def test_chronological_split_keeps_whole_dates_and_future_test(v2_training_frame):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)

    assert train["game_date"].max() < test["game_date"].min()
    assert set(train["game_date"]).isdisjoint(set(test["game_date"]))


def test_residual_models_predict_and_round_trip(v2_training_frame, tmp_path):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    bundle = fit_residual_models(
        train,
        model_type="elastic_net",
        min_rows=20,
    )
    bundle.multiplier_bounds = {
        "fga_lower": 0.5,
        "fga_upper": 1.5,
        "reb_chances_lower": 0.5,
        "reb_chances_upper": 1.5,
    }

    predictions = bundle.predict(test)
    assert len(predictions) == len(test)
    assert (predictions["v2_pred_fga"] >= 0).all()
    assert (predictions["v2_pred_reb_chances"] >= 0).all()

    path = save_model_bundle(bundle, tmp_path / "v2.joblib")
    loaded = load_model_bundle(path)
    loaded_predictions = loaded.predict(test)
    pd.testing.assert_series_equal(
        predictions["v2_pred_fga"],
        loaded_predictions["v2_pred_fga"],
    )
    assert loaded.multiplier_bounds == bundle.multiplier_bounds


def test_evaluation_compares_v1_and_v2(v2_training_frame):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    bundle = fit_residual_models(
        train,
        model_type="elastic_net",
        min_rows=20,
    )
    evaluation = evaluate_model_bundle(bundle, test)

    assert evaluation.summary["stat"].tolist() == ["fga", "reb_chances", "reb"]
    assert set(["v1_mae", "v2_mae", "mae_improvement_pct"]).issubset(
        evaluation.summary.columns
    )
    assert evaluation.summary.set_index("stat").loc["fga", "v2_mae"] < evaluation.summary.set_index("stat").loc["fga", "v1_mae"]


def test_hist_gradient_boosting_pipeline_fits(v2_training_frame):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    bundle = fit_residual_models(
        train,
        model_type="hist_gradient_boosting",
        parameters_by_target={
            "fga": {"max_iter": 20, "min_samples_leaf": 5},
            "reb_chances": {"max_iter": 20, "min_samples_leaf": 5},
        },
        min_rows=20,
    )

    predictions = bundle.predict(test)
    assert predictions[["v2_pred_fga", "v2_pred_reb_chances"]].notna().all().all()


def test_bundle_can_fit_serialize_and_apply_conversion_model(
    v2_training_frame, tmp_path
):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    bundle = fit_model_bundle(
        train,
        model_type="elastic_net",
        conversion_parameters={"alpha": 0.003, "l1_ratio": 0.25},
        min_rows=20,
    )

    predictions = bundle.predict(test)
    assert bundle.conversion_model is not None
    assert predictions["pred_reb_conversion"].notna().all()
    assert predictions["pred_reb_conversion"].between(
        bundle.conversion_model.lower_bound,
        bundle.conversion_model.upper_bound,
    ).all()
    assert (
        predictions["v2_pred_reb"]
        == predictions["v2_pred_reb_chances"]
        * predictions["pred_reb_conversion"]
    ).all()

    path = save_model_bundle(bundle, tmp_path / "v2-conversion.joblib")
    loaded = load_model_bundle(path)
    loaded_predictions = loaded.predict(test)
    assert loaded.conversion_model is not None
    pd.testing.assert_series_equal(
        predictions["v2_pred_reb"],
        loaded_predictions["v2_pred_reb"],
    )

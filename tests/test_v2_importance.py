from src.v2.evaluate import chronological_split
from src.v2.importance import group_ablation_table, permutation_importance_table
from src.v2.models import fit_residual_models


def test_permutation_importance_uses_raw_feature_names(v2_training_frame):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    bundle = fit_residual_models(
        train,
        model_type="elastic_net",
        min_rows=20,
    )
    importance = permutation_importance_table(
        bundle,
        test,
        n_repeats=2,
    )

    assert set(importance["target"]) == {"fga", "reb_chances"}
    assert "profile_usage_rate" in importance["feature"].values
    assert importance.groupby("target")["rank"].min().eq(1).all()


def test_group_ablation_retrains_without_selected_group(v2_training_frame):
    train, test = chronological_split(v2_training_frame, test_fraction=0.25)
    ablation = group_ablation_table(
        train,
        test,
        model_type="elastic_net",
        feature_groups={
            "profile": ["profile_usage_rate", "profile_touches_per_min"],
            "similarity": ["similarity_score_mean", "numeric_distance_mean"],
        },
        min_rows=20,
    )

    assert len(ablation) == 4
    assert set(ablation["feature_group"]) == {"profile", "similarity"}
    assert set(ablation["target"]) == {"fga", "reb_chances"}

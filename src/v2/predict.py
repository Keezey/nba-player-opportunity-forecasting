"""Automatic single-game inference using a saved V2 model bundle."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..predict import predict_player_game
from .features import build_pregame_feature_row
from .models import V2ModelBundle
from .tuning import TrustParameters, rebase_feature_frame_with_trust


def predict_player_game_v2(
    bundle: V2ModelBundle,
    player_query: str | int,
    game_date,
    *,
    prediction_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run V1 pregame lookup, then apply the fitted V2 residual corrections."""
    prediction_parameters = dict(bundle.workflow_parameters or {})
    bounds_payload = getattr(bundle, "multiplier_bounds", None)
    prediction_parameters.setdefault("multiplier_bounds", bounds_payload)
    if prediction_overrides:
        prediction_parameters.update(prediction_overrides)
    resolved_bounds = prediction_parameters.get("multiplier_bounds")

    v1_result = predict_player_game(
        player_query,
        game_date,
        **prediction_parameters,
    )
    feature_row = build_pregame_feature_row(v1_result)
    feature_frame = pd.DataFrame([feature_row])
    if bundle.trust_parameters:
        feature_frame = rebase_feature_frame_with_trust(
            feature_frame,
            TrustParameters(**bundle.trust_parameters),
            resolved_bounds,
        )
    v2_projection = bundle.predict(feature_frame).iloc[0]
    return {
        "target_player_id": v1_result["target_player_id"],
        "target_player_name": v1_result["target_player_name"],
        "game": v1_result["game"],
        "v1_projection": v1_result["projection"],
        "v2_projection": v2_projection,
        "similar_players": v1_result["similar_players"],
        "feature_row": feature_frame.iloc[0],
    }

import pandas as pd
import pytest

from src.v2.player_pool import (
    PlayerPoolResult,
    build_nested_player_pools,
    build_player_season_profiles,
    freeze_eligible_player_profiles,
    read_player_pool,
    select_stratified_players,
    write_player_pool,
)


def _synthetic_player_games(season="2022-23"):
    rows = []
    player_id = 100
    usage_values = [0.12, 0.15, 0.19, 0.22, 0.27, 0.32]
    for position_index, position in enumerate(["G", "F", "C"]):
        for usage_index, usage_rate in enumerate(usage_values):
            player_id += 1
            for game_index in range(4):
                minutes = 28 + usage_index
                rows.append(
                    {
                        "season": season,
                        "game_id": f"{season}-{player_id}-{game_index}",
                        "game_date": pd.Timestamp(f"{season[:4]}-10-01")
                        + pd.Timedelta(days=game_index),
                        "player_id": player_id,
                        "player_name": f"{position} Player {usage_index}",
                        "minutes": minutes,
                        "fga": minutes * (0.25 + usage_rate),
                        "reb_chances": minutes * (0.30 + 0.02 * position_index),
                        "touches": minutes * (1.0 + usage_rate),
                        "usage_rate": usage_rate,
                        "listed_position": position,
                        "starter_flag": 1,
                    }
                )
    return pd.DataFrame(rows)


def _eligible_profiles(games=None):
    profiles = build_player_season_profiles(
        _synthetic_player_games() if games is None else games,
        min_games=4,
        min_average_minutes=20,
        min_starter_rate=0.5,
        min_data_coverage=0.95,
    )
    return profiles, freeze_eligible_player_profiles(profiles)


def test_player_season_profiles_apply_eligibility_and_weighted_rates():
    games = _synthetic_player_games()
    low_minutes = games[games["player_id"] == 101].copy()
    low_minutes["player_id"] = 999
    low_minutes["player_name"] = "Low Minutes"
    low_minutes["minutes"] = 10
    games = pd.concat([games, low_minutes], ignore_index=True)

    profiles, _ = _eligible_profiles(games)

    assert profiles["eligible"].sum() == 18
    first = profiles[profiles["player_id"] == 101].iloc[0]
    assert first["avg_minutes"] == pytest.approx(28)
    assert first["usage_rate"] == pytest.approx(0.12)
    assert first["fga_per_min"] == pytest.approx(0.37)
    assert first["required_data_coverage"] == pytest.approx(1.0)

    rejected = profiles[profiles["player_id"] == 999].iloc[0]
    assert not rejected["eligible"]
    assert "minutes" in rejected["eligibility_reason"]


def test_freeze_profiles_assigns_balanced_usage_terciles_and_latest_season():
    first_season = _synthetic_player_games("2022-23")
    second_season = _synthetic_player_games("2023-24")
    second_season = second_season[second_season["player_id"] == 101].copy()
    second_season["usage_rate"] = 0.18
    games = pd.concat([first_season, second_season], ignore_index=True)

    profiles, eligible = _eligible_profiles(games)

    assert len(profiles) == 19
    assert len(eligible) == 18
    latest = eligible[eligible["player_id"] == 101].iloc[0]
    assert latest["profile_season"] == "2023-24"
    assert latest["eligible_season_count"] == 2
    assert latest["eligible_seasons"] == "2022-23,2023-24"

    counts = eligible.groupby(["position_group", "usage_tier"]).size()
    assert set(counts.index.get_level_values("position_group")) == {"G", "F", "C"}
    assert counts.eq(2).all()


def test_stratified_selection_is_balanced_and_reproducible():
    _, eligible = _eligible_profiles()

    selected_one, summary_one = select_stratified_players(
        eligible,
        players_per_stratum=1,
        random_state=42,
    )
    selected_two, _ = select_stratified_players(
        eligible,
        players_per_stratum=1,
        random_state=42,
    )

    assert len(selected_one) == 9
    assert selected_one["player_id"].tolist() == selected_two["player_id"].tolist()
    assert selected_one.groupby("stratum").size().eq(1).all()
    assert summary_one["eligible_players"].eq(2).all()
    assert summary_one["selected_players"].eq(1).all()


def test_empty_stratified_selection_returns_nine_zero_summary_rows():
    selected, summary = select_stratified_players(
        pd.DataFrame(),
        players_per_stratum=1,
    )

    assert selected.empty
    assert len(summary) == 9
    assert summary["eligible_players"].eq(0).all()
    assert summary["selected_players"].eq(0).all()


def test_player_pool_csv_round_trip(tmp_path):
    profiles, eligible = _eligible_profiles()
    selected, summary = select_stratified_players(
        eligible,
        players_per_stratum=1,
        random_state=7,
    )
    result = PlayerPoolResult(
        selected=selected,
        eligible_players=eligible,
        player_seasons=profiles,
        strata_summary=summary,
    )

    selected_path, eligible_path = write_player_pool(
        result,
        tmp_path / "target_pool.csv",
    )

    assert selected_path.exists()
    assert eligible_path.exists()
    assert read_player_pool(selected_path) == selected["player_id"].tolist()


def test_nested_player_pools_preserve_anchor_and_are_reproducible():
    _, eligible = _eligible_profiles()
    anchor, _ = select_stratified_players(
        eligible,
        players_per_stratum=1,
        random_state=42,
    )

    pools_one = build_nested_player_pools(
        eligible,
        pool_sizes=[5, 9, 14, 18],
        anchor_players=anchor,
        random_state=42,
    )
    pools_two = build_nested_player_pools(
        eligible,
        pool_sizes=[5, 9, 14, 18],
        anchor_players=anchor,
        random_state=42,
    )

    prior_ids = set()
    for size in [5, 9, 14, 18]:
        ids = set(pools_one[size]["player_id"])
        assert len(ids) == size
        assert prior_ids.issubset(ids)
        assert pools_one[size]["player_id"].tolist() == pools_two[size][
            "player_id"
        ].tolist()
        prior_ids = ids

    assert set(pools_one[9]["player_id"]) == set(anchor["player_id"])
    assert set(pools_one[18]["player_id"]) == set(eligible["player_id"])

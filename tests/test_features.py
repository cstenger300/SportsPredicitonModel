import pandas as pd
from nfl_model.features import build_features, TEAM_METRICS


def team_games():
    rows = []
    for index, date in enumerate(["2022-09-01", "2022-09-08", "2022-09-15"]):
        for home, team, opponent in [(1, "A", "B"), (0, "B", "A")]:
            rows.append({"game_id": f"g{index}", "season": 2022, "week": index + 1,
                         "game_date": pd.Timestamp(date), "team": team, "opponent": opponent,
                         "is_home": home, "points": 20 + index,
                         **{name: float(index + home) for name in TEAM_METRICS}})
    return pd.DataFrame(rows)


def test_current_and_future_outcomes_cannot_change_pregame_features():
    original = team_games()
    before, names = build_features(original)
    changed = original.copy()
    mask = changed.game_date.ge(pd.Timestamp("2022-09-08"))
    changed.loc[mask, TEAM_METRICS + ["points"]] = 9999
    after, other_names = build_features(changed)
    assert names == other_names
    selected = before.game_date.le(pd.Timestamp("2022-09-08"))
    pd.testing.assert_frame_equal(before.loc[selected, names], after.loc[selected, names])
    assert "points" not in names
    assert "points_allowed" not in names


def test_opponent_features_and_first_game_missing_history():
    frame, _ = build_features(team_games())
    first = frame.loc[frame.game_id.eq("g0")]
    assert first.recent_off_epa.isna().all()
    second_home = frame.loc[frame.game_id.eq("g1") & frame.is_home.eq(1)].iloc[0]
    assert second_home.recent_off_epa == 1
    assert second_home.opponent_recent_off_epa == 0
    assert second_home.rest_days == 7


def quarterback_games():
    frame = team_games()
    frame["primary_qb_id"] = frame.team + "_qb"
    frame["primary_qb_epa"] = frame.off_pass_epa
    frame["primary_qb_dropbacks"] = 30.0
    return frame


def test_enhanced_features_do_not_use_current_or_future_outcomes():
    original = quarterback_games()
    before, names = build_features(original, "enhanced")
    changed = original.copy()
    mask = changed.game_date.ge(pd.Timestamp("2022-09-08"))
    changed.loc[mask, TEAM_METRICS + ["points", "primary_qb_epa", "primary_qb_dropbacks"]] = 9999
    changed.loc[mask, "primary_qb_id"] = "future_starter"
    after, _ = build_features(changed, "enhanced")
    selected = before.game_date.le(pd.Timestamp("2022-09-08"))
    pd.testing.assert_frame_equal(before.loc[selected, names], after.loc[selected, names])
    assert "primary_qb_id" not in names
    assert "primary_qb_epa" not in names


def test_quarterback_change_is_not_known_until_after_the_game():
    original = quarterback_games()
    original.loc[original.game_id.eq("g1") & original.team.eq("A"), "primary_qb_id"] = "new_qb"
    original.loc[original.game_id.eq("g1") & original.team.eq("A"), "primary_qb_epa"] = -2.0
    frame, _ = build_features(original, "enhanced")
    home = frame.loc[frame.team.eq("A")].set_index("game_id")
    assert home.loc["g1", "previous_qb_recent_epa"] == 1.0
    assert home.loc["g2", "previous_qb_recent_epa"] == -2.0
    assert home.loc["g2", "previous_qb_prior_dropbacks"] == 30.0


def test_opponent_ratings_freeze_before_first_game_of_week():
    original = quarterback_games()
    original.loc[original.game_id.eq("g2"), "week"] = 2
    before, _ = build_features(original, "enhanced")
    changed = original.copy()
    changed.loc[changed.game_id.eq("g1"), TEAM_METRICS] = 9999
    after, names = build_features(changed, "enhanced")
    columns = [name for name in names if "adjusted_" in name]
    selected = before.game_id.eq("g2")
    pd.testing.assert_frame_equal(before.loc[selected, columns], after.loc[selected, columns])


def test_adjusted_ratings_recover_relative_offense_and_defense_strength():
    from nfl_model.ratings import opponent_adjusted_ratings
    offense = {"A": 0.4, "B": -0.3, "C": 0.1, "D": -0.2}
    defense = {"A": -0.2, "B": 0.2, "C": -0.1, "D": 0.1}
    rotations = [[("A", "B"), ("C", "D")], [("A", "C"), ("B", "D")], [("A", "D"), ("B", "C")]]
    rows = []
    for week in range(1, 10):
        for home, away in rotations[(week - 1) % 3]:
            for team, opponent in [(home, away), (away, home)]:
                value = offense[team] + defense[opponent]
                rows.append({"season": 2020, "week": week, "game_date": pd.Timestamp("2020-09-01") + pd.Timedelta(days=7 * week),
                             "team": team, "opponent": opponent, "is_home": int(team == home),
                             "off_epa": value, "off_pass_epa": value, "off_rush_epa": value, "off_plays": 60})
    frame = pd.DataFrame(rows)
    ratings = opponent_adjusted_ratings(frame)
    last = pd.concat([frame[["team", "week"]], ratings], axis=1).query("week == 9").set_index("team")
    assert last.loc["A", "adjusted_off_epa"] > last.loc["B", "adjusted_off_epa"]
    assert last.loc["A", "adjusted_def_allowed_epa"] < last.loc["B", "adjusted_def_allowed_epa"]

import pandas as pd
import pytest
from nfl_model.features import TEAM_METRICS
from nfl_model.prediction import forecast_game


def history():
    dates = ["2020-09-01", "2021-09-01", "2022-09-01", "2022-09-08", "2022-09-15"]
    rows = []
    for index, date in enumerate(dates):
        for home, team, opponent in [(1, "A", "B"), (0, "B", "A")]:
            rows.append({"game_id": "target" if index == 3 else f"g{index}",
                         "season": int(date[:4]), "week": max(index - 1, 1),
                         "game_date": pd.Timestamp(date), "team": team, "opponent": opponent,
                         "is_home": home, "points": 20 + index + home,
                         "primary_qb_id": team + "_qb", "primary_qb_epa": index * 0.1,
                         "primary_qb_dropbacks": 30.0,
                         **{metric: float(index + home) for metric in TEAM_METRICS}})
    return pd.DataFrame(rows)


def game():
    return {"game_id": "target", "season": 2022, "week": 2, "gameday": "2022-09-08", "home_team": "A", "away_team": "B"}


def test_forecast_excludes_target_and_future_results(tmp_path):
    data = history()
    before = forecast_game(data, game(), tmp_path / "before")
    data.loc[data.game_date.ge(pd.Timestamp("2022-09-08")), TEAM_METRICS + ["points", "primary_qb_epa"]] = 9999
    after = forecast_game(data, game(), tmp_path / "after")
    assert before == after
    assert before["trained_through_season"] == 2021
    assert before["training_games"] == 2
    assert before["latest_observed_game_date"] == "2022-09-01"
    assert before["current_season_games_observed"] == 1
    a = pd.read_csv(tmp_path / "before" / "pregame_inputs.csv")
    b = pd.read_csv(tmp_path / "after" / "pregame_inputs.csv")
    pd.testing.assert_frame_equal(a, b)
    assert all(model["home_points"] > 0 for model in after["models"])


def test_current_season_score_labels_are_not_fitted():
    data = history()
    before = forecast_game(data, game())
    data.loc[data.game_date.eq(pd.Timestamp("2022-09-01")), "points"] = 9999
    after = forecast_game(data, game())
    assert before["models"] == after["models"]


def test_missing_team_history_is_rejected():
    selected = {**game(), "away_team": "UNKNOWN"}
    with pytest.raises(ValueError, match="No pregame history"):
        forecast_game(history(), selected)

import pandas as pd
from nfl_model.data import aggregate_games


def test_offense_defense_and_excluded_plays():
    base = {"game_id": "2020_01_A_B", "season": 2020, "week": 1,
            "game_date": "2020-09-01", "season_type": "REG", "home_team": "A",
            "away_team": "B", "home_score": 21, "away_score": 14, "posteam": "A",
            "defteam": "B", "epa": 1.0, "pass_attempt": 1, "rush_attempt": 0,
            "sack": 0, "yards_gained": 25, "qb_kneel": 0, "qb_spike": 0, "down": 1}
    plays = pd.DataFrame([base, {**base, "qb_spike": 1, "epa": -99},
                          {**base, "posteam": "B", "defteam": "A", "epa": -1,
                           "yards_gained": -5, "sack": 1}])
    frame = aggregate_games(plays).set_index("team")
    assert len(frame) == 2
    assert frame.loc["A", "off_epa"] == 1
    assert frame.loc["B", "def_epa"] == 1
    assert frame.loc["A", "off_plays"] == 1
    assert frame.loc["A", "off_explosive"] == 1
    assert frame.loc["B", "off_sack_rate"] == 1
    assert frame.loc["A", "points"] == 21
    assert frame.loc["B", "points"] == 14


def test_relocation_aliases_share_one_franchise_and_qb_is_primary_passer():
    base = {"game_id": "2019_01_OAK_DEN", "season": 2019, "week": 1,
            "game_date": "2019-09-09", "season_type": "REG", "home_team": "OAK",
            "away_team": "DEN", "home_score": 24, "away_score": 16, "posteam": "OAK",
            "defteam": "DEN", "epa": 0.2, "pass_attempt": 1, "rush_attempt": 0,
            "sack": 0, "yards_gained": 5, "qb_kneel": 0, "qb_spike": 0, "down": 1,
            "passer_player_id": "starter"}
    plays = pd.DataFrame([base, base, {**base, "passer_player_id": "backup", "epa": -2.0}])
    frame = aggregate_games(plays).set_index("team")
    assert "LV" in frame.index
    assert "OAK" not in frame.index
    assert frame.loc["DEN", "opponent"] == "LV"
    assert frame.loc["LV", "primary_qb_id"] == "starter"
    assert frame.loc["LV", "primary_qb_epa"] == 0.2
    assert frame.loc["LV", "primary_qb_dropbacks"] == 2

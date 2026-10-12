import numpy as np
import pandas as pd
from nfl_model.research import prepare_research


def test_calibration_never_uses_future_season_labels(tmp_path):
    rows, schedule = [], []
    for season in range(2020, 2024):
        for index in range(60):
            game_id = f"{season}_{index}"
            schedule.append({"game_id": game_id, "spread_line": 3.5, "total_line": 45.5})
            for model in ["ridge", "poisson"]:
                for home in [0, 1]:
                    rows.append({"game_id": game_id, "season": season, "model": model, "is_home": home,
                                 "game_date": f"{season}-09-01", "points": (28 if index % 3 else 14) if home else 21,
                                 "predicted_points": (24 + np.sin(index) * 4) if home else 21})
    predictions = pd.DataFrame(rows)
    prediction_file = tmp_path / "predictions.csv"
    schedule_file = tmp_path / "schedule.csv"
    pd.DataFrame(schedule).to_csv(schedule_file, index=False)
    predictions.to_csv(prediction_file, index=False)
    prepare_research(prediction_file, schedule_file, tmp_path / "before")
    predictions.loc[predictions.season.eq(2023), "points"] = 90
    predictions.to_csv(prediction_file, index=False)
    prepare_research(prediction_file, schedule_file, tmp_path / "after")
    before = pd.read_csv(tmp_path / "before/validation.csv")
    after = pd.read_csv(tmp_path / "after/validation.csv")
    early = before.season.eq(2022)
    np.testing.assert_allclose(before.loc[early, "calibrated_conditional_win"], after.loc[early, "calibrated_conditional_win"])
    assert (before.calibration_latest_season < before.season).all()

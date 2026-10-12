import pandas as pd
import numpy as np
from nfl_model.evaluation import evaluate


def fixture():
    rows = []
    for season in [2020, 2021, 2022]:
        for week in range(1, 7):
            for home in [0, 1]:
                rows.append({"game_id": f"{season}_{week}", "season": season, "week": week,
                             "game_date": f"{season}-09-{week:02}", "team": "A" if home else "B",
                             "opponent": "B" if home else "A", "is_home": home,
                             "points": 14 + week + home * 3, "rating": week * 0.1 + home})
    return pd.DataFrame(rows)


def test_future_labels_do_not_change_earlier_model_predictions(tmp_path):
    data = fixture()
    metrics = evaluate(data, ["rating", "is_home"], 2021, 2022, tmp_path / "before")
    data.loc[data.season.eq(2022), "points"] = 80
    evaluate(data, ["rating", "is_home"], 2021, 2022, tmp_path / "after")
    before = pd.read_csv(tmp_path / "before" / "predictions.csv")
    after = pd.read_csv(tmp_path / "after" / "predictions.csv")
    np.testing.assert_allclose(before.predicted_points, after.predicted_points)
    assert len(metrics) == 6
    assert metrics.games.eq(6).all()
    assert (tmp_path / "before" / "report.md").exists()
    assert before.loc[before.model.eq("poisson"), "predicted_points"].gt(0).all()

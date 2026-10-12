from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest
from test_betting import forecast, quotes, bundle


def test_dashboard_renders_forecast_prices_and_disables_unreviewed_bet(monkeypatch, tmp_path):
    import nfl_model.workflow
    monkeypatch.setattr(nfl_model.workflow, "ROOT", tmp_path)
    schedules = tmp_path / "data/schedules"
    schedules.mkdir(parents=True)
    date = str(datetime.now(ZoneInfo("America/Chicago")).date())
    pd.DataFrame([{"game_id": "2026_05_BUF_LA", "gameday": date, "home_team": "LA", "away_team": "BUF",
                   "spread_line": 3.5, "total_line": 54.5}]).to_csv(schedules / "games.csv", index=False)
    run = tmp_path / "artifacts/upcoming/2026_05_BUF_LA"
    run.mkdir(parents=True)
    f = forecast()
    f.update({"latest_observed_game_date": "2026-10-11", "trained_through_season": 2025})
    for model in f["models"]:
        model["total_points"] = model["home_points"] + model["away_points"]
        model["away_margin"] = model["away_points"] - model["home_points"]
    (run / "prediction.json").write_text(json.dumps(f))
    quotes().to_csv(run / "quotes.csv", index=False)
    calibration = tmp_path / "artifacts/probability"
    calibration.mkdir()
    joblib.dump(bundle(), calibration / "calibration.joblib")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/nfl_model/dashboard.py")).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 5
    assert any(metric.label == "BUF expected points" and metric.value == "24.0" for metric in app.metric)
    paper_button = next(button for button in app.button if button.label == "Log top paper candidate")
    assert paper_button.disabled
    assert len(app.dataframe) >= 2

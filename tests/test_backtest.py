import numpy as np
import pandas as pd
from nfl_model.backtest import backtest


class Calibrator:
    def predict_proba(self, x):
        return np.tile([.2, .8], (len(x), 1))


def test_timestamped_backtest_uses_latest_line_at_cutoff_not_old_better_line(monkeypatch, tmp_path):
    predictions = pd.DataFrame([{"game_id": "2021_01_A_B", "season": 2021, "game_date": "2021-09-01",
                                  "model": model, "is_home": home, "points": 27 if home else 21,
                                  "predicted_points": 27 if home else 21}
                                 for model in ["ridge", "poisson"] for home in [0, 1]])
    quotes = pd.DataFrame([{"game_id": "2021_01_A_B", "bookmaker": "book", "market": "spread", "side": "home",
                            "point": point, "american_odds": -110, "observed_at": observed,
                            "source_updated_at": observed, "commence_time": "2021-09-01T23:15:00Z", "source": "fixture"}
                           for point, observed in [(-3, "2021-09-01T22:10:00Z"), (-7, "2021-09-01T22:15:00Z"),
                                                   (-2, "2021-09-01T22:20:00Z")]])
    residuals = pd.DataFrame({"home_residual": [-1, 1], "away_residual": [1, -1],
                             "actual_home": [20, 23], "actual_away": [21, 20]})
    bundle = {"residual_season": 2019, "residuals": {kind: residuals for kind in ["ridge", "poisson"]},
              "observations": pd.DataFrame({"season": [2020]})}
    monkeypatch.setattr("nfl_model.backtest.calibrators_from", lambda _: {(kind, "spread"): Calibrator() for kind in ["ridge", "poisson"]})
    summary = backtest(predictions, quotes, bundle, tmp_path)
    bets = pd.read_csv(tmp_path / "bets.csv")
    assert summary["mode"] == "timestamped historical odds"
    assert summary["bets"] == 1
    assert bets.point.iloc[0] == -7
    assert bets.result.iloc[0] == "loss"
    assert summary["profit_units"] == -1
    assert summary["quotes_after_cutoff"] == 1

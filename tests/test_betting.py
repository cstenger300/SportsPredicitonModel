import numpy as np
import pandas as pd
import pytest
from nfl_model.betting import decimal_odds, expected_value, settle, validate_quotes, price_quotes
from nfl_model.probability import score_distribution, apply_calibrator, fit_calibrator


def quotes():
    return pd.DataFrame([{"game_id": "2026_05_BUF_LA", "bookmaker": "book", "market": "moneyline",
                          "side": side, "point": 0, "american_odds": price,
                          "observed_at": "2026-10-12T16:00:00Z", "source_updated_at": "2026-10-12T16:00:00Z",
                          "commence_time": "2026-10-13T00:15:00Z", "source": "test-fixture"}
                         for side, price in [("home", -110), ("away", -110)]])


def residuals():
    return pd.DataFrame({"home_residual": [-5, 2, 4, -2], "away_residual": [3, -2, -4, 2],
                         "actual_home": [20, 24, 28, 21], "actual_away": [21, 21, 22, 21]})


def forecast():
    return {"game_id": "2026_05_BUF_LA", "home_team": "LA", "away_team": "BUF",
            "generated_at_utc": "2026-10-12T16:00:00Z", "feature_history_before": "2026-10-12",
            "models": [{"model": kind, "home_points": 27, "away_points": 24} for kind in ["ridge", "poisson"]],
            "context": {"lineups_reviewed": False, "observed_at": "2026-10-12T16:00:00Z",
                        "teams": {"LA": {"expected_qb_gsis_id": "la-qb"}, "BUF": {"expected_qb_gsis_id": "buf-qb"}}}}


def bundle():
    calibrator = fit_calibrator(np.linspace(.1, .9, 100), [0] * 50 + [1] * 50)
    return {"residuals": {kind: residuals() for kind in ["ridge", "poisson"]}, "calibrated_through_season": 2024,
            "calibrators": {(kind, "moneyline"): calibrator for kind in ["ridge", "poisson"]}}


def test_pricing_and_push_settlement():
    assert decimal_odds(-110) == pytest.approx(1 + 100 / 110)
    assert expected_value({"win": 0.5238095238, "loss": 0.4761904762}, -110) == pytest.approx(0, abs=1e-9)
    assert expected_value({"win": .5, "loss": .4, "push": .1}, 100) == pytest.approx(.1)
    assert settle("spread", "home", -3, 24, 21) == "push"
    assert settle("spread", "away", 3.5, 24, 21) == "win"
    assert settle("total", "under", 45, 24, 21) == "push"
    assert settle("moneyline", "home", 0, 21, 21) == "push"


@pytest.mark.parametrize("kind", ["ridge", "poisson"])
def test_distribution_probability_mass_and_paired_outcomes(kind):
    distribution = score_distribution(kind, 27, 24, residuals())
    assert distribution.mass.sum() == pytest.approx(1)
    h = distribution.probabilities("moneyline", "home")
    a = distribution.probabilities("moneyline", "away")
    assert h["push"] == pytest.approx(.25)
    assert h["win"] == pytest.approx(a["loss"])
    assert sum(h.values()) == pytest.approx(1)
    over = distribution.probabilities("total", "over", 51)
    under = distribution.probabilities("total", "under", 51)
    assert over["win"] == pytest.approx(under["loss"])


def test_calibrated_opposite_sides_remain_complementary():
    calibrator = bundle()["calibrators"][("ridge", "moneyline")]
    home = apply_calibrator({"win": .6, "loss": .3, "push": .1}, calibrator)
    away = apply_calibrator({"win": .3, "loss": .6, "push": .1}, calibrator, reverse=True)
    assert home["win"] == pytest.approx(away["loss"])
    assert home["push"] == away["push"]


def test_quote_cutoffs_and_timezone_validation():
    frame = quotes()
    frame.loc[0, "observed_at"] = "2026-10-13T00:15:00Z"
    with pytest.raises(ValueError, match="before kickoff"):
        validate_quotes(frame)
    frame = quotes()
    frame.loc[0, "observed_at"] = "2026-10-12T16:00:00"
    with pytest.raises(ValueError, match="explicit timezone"):
        validate_quotes(frame)


def test_stale_or_unreviewed_inputs_cannot_be_paper_candidates():
    result = price_quotes(forecast(), quotes(), bundle(), now="2026-10-12T16:01:00Z")
    assert not result.research_candidate.any()
    assert result.status.str.contains("lineups not reviewed").all()
    assert result.no_vig_probability.sum() == pytest.approx(1)
    f = forecast()
    f["context"]["lineups_reviewed"] = True
    result = price_quotes(f, quotes(), bundle(), now="2026-10-12T17:00:00Z")
    assert not result.research_candidate.any()
    assert result.status.str.contains("stale quote").all()

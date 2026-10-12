import pandas as pd
import pytest
from nfl_model.paper import PaperLedger
from test_betting import quotes, forecast


def candidate():
    return {**quotes().iloc[0].to_dict(), "research_candidate": True, "p_win": .6, "p_push": 0.01, "estimated_ev": .15}


def test_paper_ledger_is_immutable_and_settlement_idempotent(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.sqlite")
    ledger.record_quotes(quotes())
    f = forecast()
    f["context"]["lineups_reviewed"] = True
    assert ledger.record_bet(candidate(), f, now="2026-10-12T16:01:00Z")
    changed = {**candidate(), "american_odds": 150}
    assert not ledger.record_bet(changed, f, now="2026-10-12T16:02:00Z")
    assert ledger.bets().american_odds.iloc[0] == -110
    schedule = pd.DataFrame([{"game_id": "2026_05_BUF_LA", "gameday": "2026-10-12", "home_score": 27, "away_score": 24}])
    assert ledger.settle_from_schedule(schedule, now="2026-10-13T16:00:00Z") == 1
    assert ledger.settle_from_schedule(schedule, now="2026-10-13T17:00:00Z") == 0
    assert ledger.summary()["profit_units"] == pytest.approx(100 / 110)


def test_paper_bets_reject_stale_quotes_and_started_games(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.sqlite")
    f = forecast()
    f["context"]["lineups_reviewed"] = True
    with pytest.raises(ValueError, match="stale"):
        ledger.record_bet(candidate(), f, now="2026-10-12T17:00:00Z")
    with pytest.raises(ValueError, match="before kickoff"):
        ledger.record_bet(candidate(), f, now="2026-10-13T01:00:00Z")

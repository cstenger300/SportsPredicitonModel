import pandas as pd
import pytest
from test_prediction import history, game
from sports_lab.nfl_feed import metric_rows, align_odds
from sports_lab.core import METRICS


def test_public_metrics_ignore_target_and_future_results():
    games = pd.DataFrame([{**game(),'commence_time':pd.Timestamp('2022-09-08T23:00:00Z')}])
    data = history()
    data['points_allowed'] = 17
    _, before = metric_rows(data,games,'2022-09-08T12:00:00Z')
    data.loc[data.game_date.ge('2022-09-08'),['points','off_epa','def_sack_rate']] = 9999
    _, after = metric_rows(data,games,'2022-09-08T12:00:00Z')
    pd.testing.assert_frame_equal(before[METRICS['NFL']],after[METRICS['NFL']])
    assert before.iloc[0].metrics_as_of < before.iloc[0].commence_time


def test_sportsbook_matching_requires_teams_and_kickoff():
    u = pd.DataFrame([dict(event_id='public',home='BUF',away='LA',commence_time=pd.Timestamp('2026-11-08T18:00:00Z'))])
    e = pd.DataFrame([dict(event_id='provider',home='Buffalo Bills',away='Los Angeles Rams',commence_time='2026-11-08T18:00:00Z')])
    q = pd.DataFrame([dict(event_id='provider',american_odds=-110)])
    upcoming,quotes = align_odds(u,e,q)
    assert upcoming.iloc[0].event_id == 'provider'
    assert upcoming.iloc[0].home == 'Buffalo Bills'
    e['commence_time'] = '2026-11-09T18:00:00Z'
    with pytest.raises(ValueError,match='No sportsbook'):
        align_odds(u,e,q)

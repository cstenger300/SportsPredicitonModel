import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from sports_lab.core import METRICS, fit_model, rank_quotes, Ledger
from sports_lab import provider

NOW = pd.Timestamp('2026-11-05T12:00:00Z')

def inputs(sport):
    rng = np.random.default_rng(42)
    n = 220
    dates = pd.date_range('2025-01-01', periods=n, tz='UTC')
    x = rng.normal(size=(n, len(METRICS[sport])))
    h = pd.DataFrame(x, columns=METRICS[sport])
    h['event_id'] = [f'h{i}' for i in range(n)]
    h['home'], h['away'] = 'Home', 'Away'
    h['commence_time'] = dates
    h['metrics_as_of'] = dates - pd.Timedelta(hours=1)
    h['result_available_at'] = dates + pd.Timedelta(hours=4)
    h['home_win'] = (x[:,0] > 0).astype(int)
    u = h.iloc[[0]].drop(columns=['home_win','result_available_at']).copy()
    u['event_id'] = 'next'
    u['commence_time'] = NOW + pd.Timedelta(hours=3)
    u['metrics_as_of'] = NOW - pd.Timedelta(hours=1)
    return h,u

@pytest.mark.parametrize('sport', METRICS)
def test_chronological_model_and_explanations(sport):
    h,u = inputs(sport)
    forecasts, report = fit_model(h,u,sport,NOW)
    assert 0 < forecasts.home_probability.iloc[0] < 1
    assert report['validation_events'] == 44
    assert report['beats_baseline']
    assert 'log-odds' in forecasts.why.iloc[0]
    h.loc[0,'metrics_as_of'] = h.loc[0,'commence_time']
    with pytest.raises(ValueError, match='before'):
        fit_model(h,u,sport,NOW)

def test_prices_tracking_and_settlement(tmp_path):
    h,u = inputs('NBA')
    f,r = fit_model(h,u,'NBA',NOW)
    q = pd.DataFrame([dict(event_id='next', bookmaker='Book',side='home',american_odds=5000,
        observed_at=NOW.isoformat(),source_updated_at=NOW.isoformat())])
    ranked = rank_quotes(f,q,r,'NBA',NOW)
    assert ranked.iloc[0].eligible
    ledger = Ledger(tmp_path/'paper.sqlite')
    pick = ranked.iloc[0].to_dict()
    assert ledger.record(pick,NOW)
    assert not ledger.record(pick,NOW)
    with pytest.raises(ValueError, match='stale'):
        ledger.record(pick,NOW+pd.Timedelta(minutes=16))
    results = pd.DataFrame([dict(sport='NBA',event_id='next',status='final',winner='home',completed_at=(NOW+pd.Timedelta(hours=6)).isoformat())])
    assert ledger.settle(results,NOW+pd.Timedelta(days=1)) == 1
    assert ledger.settle(results,NOW+pd.Timedelta(days=1)) == 0
    assert ledger.rows().iloc[0].profit_units == 50
    q['observed_at'] = (NOW-pd.Timedelta(minutes=20)).isoformat()
    q['source_updated_at'] = q.observed_at
    assert not rank_quotes(f,q,r,'NBA',NOW).iloc[0].eligible

def test_no_future_results_in_training():
    h,u = inputs('NBA')
    h['result_available_at'] = NOW + pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='100'):
        fit_model(h,u,'NBA',NOW)

def test_provider_two_way_and_redacted_errors(monkeypatch):
    monkeypatch.setattr(provider,'get',lambda *a,**k:[dict(id='a',home_team='H',away_team='A',commence_time='2026-11-06T12:00:00Z',bookmakers=[dict(title='Book',last_update=NOW.isoformat(),markets=[dict(key='h2h',outcomes=[dict(name='H',price=-110),dict(name='A',price=100)])])])])
    events, quotes = provider.odds('basketball_nba')
    assert len(events)==1 and len(quotes)==2
    assert set(quotes.side)=={'home','away'}

def test_dashboard_empty_state(monkeypatch,tmp_path):
    import sports_lab.core
    original = sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    app = AppTest.from_file(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src/sports_lab/dashboard.py')).run(timeout=30)
    assert not app.exception
    assert len(app.tabs)==5
    assert list(next(box for box in app.selectbox if box.label=='Sport').options)==list(METRICS)

@pytest.mark.parametrize('winner,expected,profit', [('away','lost',-1), ('void','void',0)])
def test_settlement_loss_and_void(tmp_path,winner,expected,profit):
    h,u = inputs('NBA')
    f,r = fit_model(h,u,'NBA',NOW)
    q = pd.DataFrame([dict(event_id='next',bookmaker='Book',side='home',american_odds=5000,
        observed_at=NOW.isoformat(),source_updated_at=NOW.isoformat())])
    pick = rank_quotes(f,q,r,'NBA',NOW).iloc[0].to_dict()
    ledger = Ledger(tmp_path/'paper.sqlite')
    ledger.record(pick,NOW)
    results = pd.DataFrame([dict(sport='NBA',event_id='next',status='final',winner=winner,
        completed_at=(NOW+pd.Timedelta(hours=6)).isoformat())])
    ledger.settle(results,NOW+pd.Timedelta(days=1))
    assert ledger.rows().iloc[0].result == expected
    assert ledger.rows().iloc[0].profit_units == profit

def test_latest_quote_replaces_better_old_price():
    h,u = inputs('NBA')
    f,r = fit_model(h,u,'NBA',NOW)
    quotes = pd.DataFrame([
        dict(event_id='next',bookmaker='Book',side='home',american_odds=5000,
             observed_at=(NOW-pd.Timedelta(minutes=1)).isoformat(),source_updated_at=(NOW-pd.Timedelta(minutes=1)).isoformat()),
        dict(event_id='next',bookmaker='Book',side='home',american_odds=-200,
             observed_at=NOW.isoformat(),source_updated_at=NOW.isoformat())])
    ranked = rank_quotes(f,quotes,r,'NBA',NOW)
    assert len(ranked)==1 and ranked.iloc[0].american_odds==-200

def test_prospective_forecasts_immutable_and_independent_of_bets(tmp_path):
    from sports_lab.performance import ForecastStore
    h,u = inputs('NBA')
    forecasts,report = fit_model(h,u,'NBA',NOW)
    store = ForecastStore(tmp_path/'paper.sqlite')
    assert store.record(forecasts,report,'NBA',NOW)==1
    changed = forecasts.copy()
    changed['home_probability'] = .01
    assert store.record(changed,report,'NBA',NOW)==0
    assert store.rows().iloc[0].home_probability == forecasts.iloc[0].home_probability
    results = pd.DataFrame([dict(sport='NBA',event_id='next',status='final',winner='home',completed_at=(NOW+pd.Timedelta(hours=6)).isoformat())])
    assert store.settle(results,NOW+pd.Timedelta(days=1))==1
    assert store.settle(results,NOW+pd.Timedelta(days=1))==0
    assert store.summary().iloc[0].games==1
    assert store.summary().iloc[0].status=='Collecting evidence'
    assert Ledger(tmp_path/'paper.sqlite').rows().empty
    with pytest.raises(ValueError,match='prospective'):
        store.record(forecasts,report,'NBA',NOW+pd.Timedelta(days=1))

def test_model_version_identifies_training_data():
    h,u = inputs('NBA')
    _,r1 = fit_model(h,u,'NBA',NOW)
    _,r2 = fit_model(h,u,'NBA',NOW)
    assert r1['model_version']==r2['model_version']
    h.loc[0,'net_rating'] += .1
    _,r3 = fit_model(h,u,'NBA',NOW)
    assert r1['model_version']!=r3['model_version']

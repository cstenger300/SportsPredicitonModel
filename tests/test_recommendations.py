import pandas as pd
from sports_lab.recommendations import summarize_quotes

NOW = pd.Timestamp('2026-11-05T18:49:41Z')

def quote(book='Book',odds=425,p=.3818,side='home',age=0):
    decimal = 1+(odds/100 if odds>0 else 100/abs(odds))
    return dict(event_id='event',sport='NFL',bookmaker=book,side=side,american_odds=odds,
        selection='Baltimore' if side=='home' else 'Jacksonville',probability=p,break_even=1/decimal,
        edge=p-1/decimal,expected_return=p*decimal-1,eligible=p*decimal-1>=.03,
        observed_at=NOW-pd.Timedelta(minutes=age),source_updated_at=NOW-pd.Timedelta(minutes=age),
        commence_time=NOW+pd.Timedelta(hours=6),why='EPA: -0.13 toward home',status='Pass')

def test_one_best_selection_per_event_and_separate_explanations():
    best = summarize_quotes(pd.DataFrame([quote(),quote('Other',410),quote('Away',-500,.6182,'away')]),NOW)
    assert len(best)==1
    assert best.iloc[0].bookmaker=='Book'
    assert best.iloc[0].large_disagreement
    assert 'opponent' in best.iloc[0].prediction_reason
    assert '19.0%' in best.iloc[0].price_reason

def test_stale_better_price_does_not_replace_fresh_candidate():
    best = summarize_quotes(pd.DataFrame([quote('Old',500,age=16),quote('Fresh',410)]),NOW)
    assert best.iloc[0].bookmaker=='Fresh'
    assert best.iloc[0].eligible

def test_old_snapshot_is_not_eligible_and_duplicates_keep_latest():
    best = summarize_quotes(pd.DataFrame([quote('Book',500,age=2),quote('Book',410)]),NOW+pd.Timedelta(minutes=16))
    assert not best.iloc[0].eligible
    assert best.iloc[0].american_odds==410
    assert 'stale' in best.iloc[0].status

def test_dashboard_renders_one_candidate_and_other_book_prices(tmp_path,monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import sports_lab.core
    now = pd.Timestamp.now(tz='UTC')
    quotes = pd.DataFrame([quote(),quote('Other',410)])
    for field in ['observed_at','source_updated_at']:
        quotes[field]=now
    quotes['commence_time']=now+pd.Timedelta(hours=6)
    quotes['matchup']='Jacksonville at Baltimore'
    quotes['validation']=[{},{}]
    forecasts=pd.DataFrame([dict(home='Baltimore',away='Jacksonville',home_probability=.3818,out_of_distribution=False,why='EPA contribution')])
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py'))
    app.session_state['analysis_NBA']=(forecasts,dict(brier=.238,baseline_brier=.250,validation_events=238),quotes)
    app.run(timeout=30)
    assert not app.exception
    choices=next(s for s in app.selectbox if s.label=='Track a candidate')
    assert len(choices.options)==1
    assert any('Large model' in w.value for w in app.warning)

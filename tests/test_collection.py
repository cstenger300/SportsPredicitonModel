import json
import hashlib
from pathlib import Path
import pandas as pd
import requests
from streamlit.testing.v1 import AppTest
from sports_lab.collection import CollectionStore,coverage_table,public_games
from sports_lab import collection

NOW=pd.Timestamp('2026-11-09T14:00:00Z')


def sample():
    return pd.DataFrame([dict(event_id='a',home='A',away='B',commence_time='2026-11-10T01:00:00Z',completed=False)])


def test_public_only_default_and_cooldown_preserve_archive(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(collection,'public_games',lambda *a,**k:(calls.append(a[0]) or sample()))
    monkeypatch.setattr(collection.provider,'odds',lambda *a:(_ for _ in ()).throw(AssertionError('Paid call unexpected')))
    store=CollectionStore(tmp_path)
    first=store.collect(['NBA','NHL'],'2026-11-09',now=NOW)
    assert first.status.eq('success').all()
    store.collect(['NBA'],'2026-11-09',now=NOW+pd.Timedelta(minutes=1))
    assert calls==['NBA','NHL']
    runs=store.runs()
    assert len(runs)==2
    for row in runs.itertuples():
        path=tmp_path/row.snapshot
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row.sha256
        assert json.loads(path.read_text())['data']['games'][0]['event_id']=='a'


def test_errors_keep_last_success_and_do_not_persist_credential_urls(monkeypatch,tmp_path):
    store=CollectionStore(tmp_path)
    monkeypatch.setattr(collection,'public_games',lambda *a,**k:sample())
    store.collect(['NBA'],'2026-11-09',now=NOW)
    def fail(*a,**k):raise ValueError('https://example.test/?apiKey=do-not-store')
    monkeypatch.setattr(collection,'public_games',fail)
    store.collect(['NBA'],'2026-11-09',now=NOW+pd.Timedelta(hours=2))
    status=store.status().query("sport == 'NBA' and feed == 'public'").iloc[0]
    assert status.latest_status=='error' and pd.notna(status.last_success)
    assert 'do-not-store' not in store.runs().to_csv(index=False)
    assert len(list((tmp_path/'snapshots').rglob('*.json')))==1


def test_explicit_odds_and_tennis_configuration(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(collection.provider,'odds',lambda key:(calls.append(key) or (sample(),pd.DataFrame([dict(event_id='a',side='home',american_odds=-110)]))))
    store=CollectionStore(tmp_path)
    result=store.collect(['NBA','Tennis'],'2026-11-09',include_public=False,include_odds=True,now=NOW)
    assert calls==['basketball_nba']
    assert 'not configured' in set(result.status)
    store.collect(['Tennis'],'2026-11-09',include_public=False,include_odds=True,tennis_key='tennis_atp_test',now=NOW)
    assert calls[-1]=='tennis_atp_test'
    assert set(coverage_table().sport)==set(collection.SPORTS)


def test_public_nba_parser_does_not_treat_zero_scores_as_final(monkeypatch,tmp_path):
    class Response:
        ok=True
        def json(self):return {'events':[dict(id='a',date='2026-11-10T01:00:00Z',competitions=[dict(status={'type':{'completed':False}},competitors=[
            dict(homeAway='home',team={'id':'1','displayName':'A'},score='0'),dict(homeAway='away',team={'id':'2','displayName':'B'},score='0')])])]}
    monkeypatch.setattr(collection.requests,'get',lambda *a,**k:Response())
    frame=public_games('NBA','2026-11-09',tmp_path)
    assert not frame.iloc[0].completed and frame.iloc[0].home_score is None


def test_manual_odds_snapshot_requires_no_new_request(monkeypatch,tmp_path):
    monkeypatch.setattr(collection.provider,'odds',lambda *a:(_ for _ in ()).throw(AssertionError('Extra request')))
    store=CollectionStore(tmp_path)
    store.archive_odds('NFL','americanfootball_nfl',sample(),pd.DataFrame())
    assert store.runs().iloc[0].status=='success'


def test_collection_panel_renders_without_calls(tmp_path,monkeypatch):
    import sports_lab.core
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    monkeypatch.setattr(collection,'public_games',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected request')))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py')).run(timeout=30)
    assert not app.exception
    assert any(tab.label=='Data collection' for tab in app.tabs)
    auto=next(c for c in app.checkbox if c.label=='Collect periodically while this app session is open')
    assert auto.value is False

def test_generated_metrics_archive_without_network_and_runs_are_not_overwritten(tmp_path):
    store=CollectionStore(tmp_path)
    store.archive_metrics('NFL',sample(),sample(),{'sources':[{'season':2025,'sha256':'test'}]})
    store.archive_metrics('NFL',sample(),sample(),{'sources':[{'season':2025,'sha256':'test'}]})
    runs=store.runs()
    assert len(runs)==2 and runs.snapshot.nunique()==2
    assert store.status().query("sport == 'NFL' and feed == 'metrics'").iloc[0].latest_status=='success'
    assert store.status().query("sport == 'NBA' and feed == 'metrics'").iloc[0].latest_status=='not connected'


def test_collect_button_is_functional_and_optional_sources_do_not_auto_run(tmp_path,monkeypatch):
    import sports_lab.core
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    calls=[]
    monkeypatch.setattr(collection,'public_games',lambda sport,*a,**k:(calls.append(sport) or sample()))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py')).run(timeout=30)
    next(b for b in app.button if b.label=='Collect selected feeds now').click().run(timeout=30)
    assert not app.exception
    assert set(calls)=={'NBA','NHL','NFL','NCAAB','NCAAF'}

def test_status_panel_handles_mixed_timestamp_precision(tmp_path,monkeypatch):
    import sports_lab.core
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    monkeypatch.setattr(collection,'public_games',lambda *a,**k:sample())
    store=CollectionStore(tmp_path/'collections')
    store.collect(['NBA'],'2026-11-09',now=NOW)
    store.archive_odds('NFL','americanfootball_nfl',sample(),pd.DataFrame())
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py')).run(timeout=30)
    assert not app.exception

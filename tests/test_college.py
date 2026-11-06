import pandas as pd
import pytest
from sports_lab import college

NOW=pd.Timestamp('2026-11-06T15:00:00Z')


def games():
    dates=pd.date_range('2025-10-01',periods=140,tz='UTC')
    return pd.DataFrame([dict(event_id=str(i),home='A' if i%2 else 'B',away='B' if i%2 else 'A',
        commence_time=date,completed=True,home_score=80 if i%3 else 60,away_score=70,neutral=False) for i,date in enumerate(dates)])


def upcoming():
    return pd.DataFrame([dict(event_id='tonight',home='A',away='B',commence_time=NOW+pd.Timedelta(hours=9),neutral=False)])

@pytest.mark.parametrize('sport',['NCAAB','NCAAF'])
def test_forecast_is_separate_results_only_model(sport):
    f,r=college.forecast(games(),upcoming(),sport,NOW)
    assert len(f)==1 and 0<f.iloc[0].home_probability<1
    assert 'Elo' in r['model_type']
    assert r['validation_events']==28
    assert not f.iloc[0].out_of_distribution


def test_future_and_target_results_do_not_affect_forecast():
    h=games();f,r=college.forecast(h,upcoming(),'NCAAB',NOW)
    future=h.iloc[[0]].copy();future['event_id']='future';future['commence_time']=NOW+pd.Timedelta(hours=1)
    future['home_score']=999
    after,_=college.forecast(pd.concat([h,future],ignore_index=True),upcoming(),'NCAAB',NOW)
    pd.testing.assert_frame_equal(f,after)
    unknown=upcoming().copy();unknown['away']='Unknown'
    f,_=college.forecast(h,unknown,'NCAAB',NOW)
    assert f.iloc[0].out_of_distribution


def test_alignment_refuses_guessing_and_persists_public_ids(tmp_path):
    e=pd.DataFrame([dict(event_id='book-id',home='A',away='B',commence_time=(NOW+pd.Timedelta(hours=9)).isoformat())])
    q=pd.DataFrame([dict(event_id='book-id')])
    u,prices=college.align_events(upcoming(),e,q)
    assert u.iloc[0].public_event_id=='tonight'
    college.save_mapping(tmp_path,'NCAAB',u)
    path=tmp_path/'data/sports_lab/college/NCAAB_event_mapping.csv'
    assert pd.read_csv(path).iloc[0].event_id=='book-id'
    e['home']='Unknown'
    with pytest.raises(ValueError,match='No exact'):
        college.align_events(upcoming(),e,q)


def test_fetch_only_completed_scores_and_cache(monkeypatch,tmp_path):
    calls=[]
    payload={'events':[dict(id='1',date='2026-11-06T23:00:00Z',competitions=[dict(neutralSite=True,status={'type':{'completed':False}},competitors=[
        dict(homeAway='home',team={'displayName':'A'},score='0'),dict(homeAway='away',team={'displayName':'B'},score='0')])])]}
    class Response:
        def raise_for_status(self):pass
        def json(self):return payload
    def get(*args,**kwargs):calls.append(kwargs);return Response()
    monkeypatch.setattr(college.requests,'get',get)
    f=college.fetch_games('NCAAB','2026-11-06','2026-11-06',tmp_path)
    assert pd.isna(f.iloc[0].home_score)
    college.fetch_games('NCAAB','2026-11-06','2026-11-06',tmp_path)
    assert len(calls)==1


def test_final_scores_use_mapping_and_ignore_unfinished_games(tmp_path,monkeypatch):
    h=games().iloc[:2].copy()
    h['commence_time']=NOW-pd.Timedelta(hours=10)
    h.loc[h.index[1],'completed']=False
    u=pd.DataFrame([dict(public_event_id='0',event_id='book')])
    college.save_mapping(tmp_path,'NCAAF',u)
    monkeypatch.setattr(college,'fetch_games',lambda *args,**kwargs:h)
    results=college.final_results(tmp_path,'NCAAF',NOW)
    assert set(results.event_id)=={'0','book'}
    assert set(results.winner)=={'away'}

@pytest.mark.parametrize('sport',['NCAAB','NCAAF'])
def test_college_controls_render_without_api_key(tmp_path,monkeypatch,sport):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import sports_lab.core
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py')).run(timeout=30)
    app.selectbox[0].select(sport).run(timeout=30)
    assert not app.exception
    assert any(b.label=='Download '+sport+' data and analyze' for b in app.button)
    assert any('experimental Elo' in h.value for h in app.subheader)

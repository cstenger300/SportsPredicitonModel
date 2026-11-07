from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_college_research import history
from sports_lab.shadow import predict_pair, ShadowStore
from sports_lab.core import Ledger

NOW=pd.Timestamp('2026-11-07T15:00:00Z')


def upcoming():
    return pd.DataFrame([dict(event_id='new',home='A',away='D',commence_time=NOW+pd.Timedelta(hours=8),neutral=False)])


def test_paired_cutoffs_and_future_results_do_not_change_predictions():
    h=history();before=predict_pair(h,upcoming(),'NCAAB',NOW)
    assert before.iloc[0].training_cutoff<NOW
    assert before.iloc[0].latest_training_result<before.iloc[0].training_cutoff
    assert before.iloc[0].latest_calibration_result<before.iloc[0].training_cutoff
    future=upcoming().copy();future['completed']=True;future['home_score']=999;future['away_score']=0
    after=predict_pair(pd.concat([h,future],ignore_index=True),upcoming(),'NCAAB',NOW)
    pd.testing.assert_frame_equal(before,after)


def test_record_immutable_and_settle_separate_from_picks(tmp_path):
    paired=predict_pair(history(),upcoming(),'NCAAB',NOW)
    store=ShadowStore(tmp_path/'paper.sqlite')
    assert store.record(paired,NOW)==1
    changed=paired.copy();changed['margin_home_probability']=.01
    assert store.record(changed,NOW)==0
    assert store.rows().iloc[0].margin_home_probability==paired.iloc[0].margin_home_probability
    with pytest.raises(ValueError,match='pregame'):
        store.record(paired,NOW+pd.Timedelta(days=1))
    result=pd.DataFrame([dict(sport='NCAAB',event_id='new',status='final',winner='home',completed_at=NOW+pd.Timedelta(hours=12))])
    assert store.settle(result,NOW+pd.Timedelta(days=1))==1
    assert store.settle(result,NOW+pd.Timedelta(days=1))==0
    assert set(store.summary('NCAAB').model)=={'elo','margin'}
    assert store.rows().iloc[0].margin_brier==pytest.approx((paired.iloc[0].margin_home_probability-1)**2)
    assert Ledger(tmp_path/'paper.sqlite').rows().empty


def test_unknown_team_not_saved(tmp_path):
    u=upcoming();u['away']='Unseen'
    paired=predict_pair(history(),u,'NCAAB',NOW)
    store=ShadowStore(tmp_path/'paper.sqlite')
    assert paired.iloc[0].coverage=='Unseen participant'
    assert store.record(paired,NOW)==0


def test_shadow_controls_render(tmp_path,monkeypatch):
    import sports_lab.core
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    store=ShadowStore(tmp_path/'paper.sqlite')
    store.record(predict_pair(history(),upcoming(),'NCAAB',NOW),NOW)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py')).run(timeout=30)
    app.selectbox[0].select('NCAAB').run(timeout=30)
    assert not app.exception
    assert any(b.label=='Record both models for upcoming NCAAB games' for b in app.button)
    assert any(d.label=='Export paired forecasts' for d in app.get('download_button'))

def test_numeric_and_string_source_ids_are_not_double_counted():
    h=history();h['event_id']=h.event_id.astype(object);h.loc[h.index[0],'event_id']=12345
    before=predict_pair(h,upcoming(),'NCAAB',NOW)
    duplicate=h.iloc[[0]].copy();duplicate['event_id']='12345'
    after=predict_pair(pd.concat([h,duplicate],ignore_index=True),upcoming(),'NCAAB',NOW)
    pd.testing.assert_frame_equal(before,after)


def test_unavailable_cutoff_input_rejected_and_void_excluded(tmp_path):
    paired=predict_pair(history(),upcoming(),'NCAAB',NOW)
    store=ShadowStore(tmp_path/'paper.sqlite')
    invalid=paired.copy();invalid['latest_training_result']=invalid.training_cutoff
    with pytest.raises(ValueError,match='unavailable'):
        store.record(invalid,NOW)
    assert store.rows().empty
    store.record(paired,NOW)
    result=pd.DataFrame([dict(sport='NCAAB',event_id='new',status='final',winner='void',completed_at=NOW+pd.Timedelta(hours=12))])
    store.settle(result,NOW+pd.Timedelta(days=1))
    assert store.summary('NCAAB').empty
    assert store.rows().iloc[0].winner=='void'

def test_prospective_pair_matches_the_weekly_research_protocol():
    from sports_lab.college_research import compare_models,week_cutoff
    h=history()
    later=h.loc[h.commence_time.dt.year.eq(2025)].copy()
    later['commence_time']=later.commence_time+pd.DateOffset(years=1)
    later['event_id']=later.event_id.str.replace('2025','2026')
    complete=pd.concat([h,later],ignore_index=True)
    target=later.iloc[[0]][['event_id','home','away','commence_time','neutral']]
    cutoff=week_cutoff(target.iloc[0].commence_time)
    pair=predict_pair(complete,target,'NCAAB',cutoff+pd.Timedelta(seconds=1)).iloc[0]
    replay=compare_models(complete,'NCAAB',now='2027-01-01T00:00:00Z')['predictions']
    matched=replay.loc[replay.event_id.eq(pair.event_id)].iloc[0]
    assert pair.elo_home_probability==pytest.approx(matched.probability)
    assert pair.margin_home_probability==pytest.approx(matched.calibrated_probability)

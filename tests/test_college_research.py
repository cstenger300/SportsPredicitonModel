from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from sports_lab.college_research import compare_models,save_comparison


def history():
    rng=np.random.default_rng(123)
    strength=dict(A=18.,B=8.,C=-3.,D=-15.)
    rows=[]
    for season in [2023,2024,2025]:
        for i in range(180):
            home,away=rng.choice(list(strength),size=2,replace=False)
            margin=round(strength[home]-strength[away]+3+rng.normal(0,8))
            margin=margin if margin!=0 else 1
            date=pd.Timestamp(f'{season}-11-01T20:00:00Z')+pd.Timedelta(hours=8*i)
            rows.append(dict(event_id=f'{season}_{i}',home=home,away=away,commence_time=date,
                completed=True,home_score=80+margin,away_score=80,neutral=False))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('sport',['NCAAB','NCAAF'])
def test_margin_comparison_time_order_and_outputs(tmp_path,sport):
    result=compare_models(history(),sport,now='2026-11-07T15:00:00Z')
    fold=result['seasons'].iloc[0]
    assert pd.Timestamp(fold.latest_training_result)<pd.Timestamp(fold.first_calibration_start)
    assert pd.Timestamp(fold.latest_calibration_result)<pd.Timestamp(fold.first_test_start)
    assert result['summary']['evaluated_games']==180
    assert result['predictions'].calibrated_probability.between(0,1).all()
    assert not result['predictions'].unseen_participant.any()
    save_comparison(result,tmp_path)
    assert len(pd.read_csv(tmp_path/'predictions.csv'))==180


def test_test_scores_cannot_change_candidate_predictions():
    h=history();before=compare_models(h,'NCAAB',now='2026-11-07T15:00:00Z')['predictions']
    mask=h.commence_time.dt.year.eq(2025)
    h.loc[mask,['home_score','away_score']]=h.loc[mask,['away_score','home_score']].to_numpy()
    after=compare_models(h,'NCAAB',now='2026-11-07T15:00:00Z')['predictions']
    pd.testing.assert_frame_equal(before[['event_id','predicted_margin','calibrated_probability']],after[['event_id','predicted_margin','calibrated_probability']])


def test_history_needed_and_unknown_participants_flagged():
    h=history()
    with pytest.raises(ValueError,match='three competitive seasons'):
        compare_models(h.loc[h.commence_time.dt.year<2025],'NCAAB',now='2026-11-07T15:00:00Z')
    h.loc[h.commence_time.dt.year.eq(2025),'home']='New team'
    result=compare_models(h,'NCAAB',now='2026-11-07T15:00:00Z')
    assert result['predictions'].unseen_participant.all()


def test_research_panel_renders_and_does_not_promote(tmp_path,monkeypatch):
    import sports_lab.core
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py'))
    app.session_state['college_research_NCAAB']=compare_models(history(),'NCAAB',now='2026-11-07T15:00:00Z')
    app.run(timeout=30)
    app.selectbox[0].select('NCAAB').run(timeout=30)
    assert not app.exception
    assert any(m.label=='Margin candidate Brier' for m in app.metric)
    assert 'analysis_NCAAB' not in app.session_state

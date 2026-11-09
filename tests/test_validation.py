from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from sports_lab.core import METRICS
from sports_lab.validation import validate_history, save_report


def history():
    rng=np.random.default_rng(6)
    rows=[]
    for season in [2022,2023,2024]:
        for i in range(120):
            day=pd.Timestamp(f'{season}-08-01T18:00:00Z')+pd.Timedelta(days=i)
            metrics=dict(zip(METRICS['NFL'],rng.normal(size=6)))
            rows.append(dict(event_id=f'{season}_{i}',home='A',away='B',season=season,
                commence_time=day,metrics_as_of=day-pd.Timedelta(hours=1),
                result_available_at=day+pd.Timedelta(hours=4),home_win=int(metrics['epa_per_play']>0),**metrics))
    return pd.DataFrame(rows)


def test_folds_are_time_ordered_and_reliability_covers_every_prediction(tmp_path):
    report=validate_history(history(),'NFL')
    fold=report['seasons'].iloc[0]
    assert pd.Timestamp(fold.latest_training_result)<pd.Timestamp(fold.first_calibration_input)
    assert pd.Timestamp(fold.latest_calibration_result)<pd.Timestamp(fold.first_test_input)
    assert report['summary']['evaluated_games']==120
    assert report['reliability'].groupby('model').games.sum().tolist()==[120,120]
    save_report(report,tmp_path)
    assert len(pd.read_csv(tmp_path/'predictions.csv'))==120
    assert (tmp_path/'summary.json').exists()


def test_test_labels_cannot_change_probabilities():
    h=history()
    before=validate_history(h,'NFL')['predictions']
    mask=h.season.eq(2024)
    h.loc[mask,'home_win']=1-h.loc[mask,'home_win']
    after=validate_history(h,'NFL')['predictions']
    pd.testing.assert_frame_equal(before[['probability','calibrated_probability']],after[['probability','calibrated_probability']])


def test_missing_seasons_and_future_results_are_rejected():
    h=history()
    with pytest.raises(ValueError,match='three seasons'):
        validate_history(h.loc[h.season<2024],'NFL')
    with pytest.raises(ValueError,match='three seasons'):
        validate_history(h,'NFL',now='2024-01-01T00:00:00Z')


def test_validation_ui_does_not_change_live_model(tmp_path,monkeypatch):
    import sports_lab.core
    original=sports_lab.core.Ledger
    monkeypatch.setattr(sports_lab.core,'Ledger',lambda path:original(tmp_path/'paper.sqlite'))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/sports_lab/dashboard.py'))
    app.session_state['nfl_validation']=validate_history(history(),'NFL')
    app.run(timeout=30)
    next(box for box in app.selectbox if box.label=='Sport').select('NFL').run(timeout=30)
    assert not app.exception
    assert any(m.label=='Calibration candidate Brier' for m in app.metric)
    assert 'analysis_NFL' not in app.session_state

def test_season_reliability_counts_and_exact_bin_boundaries():
    from sports_lab.validation import reliability
    report=validate_history(history(),'NFL')
    bins=report['reliability_by_season']
    for (_,model),group in bins.groupby(['season','model']):
        assert group.games.sum()==120
    rows=pd.DataFrame(dict(home_win=[0,1,1,1],probability=[.5,.6,.7,1.0],calibrated_probability=[.5,.6,.7,1.0]))
    buckets=reliability(rows)
    raw=buckets.loc[buckets.model.eq('Uncalibrated')]
    assert raw.probability_range.tolist()==['50%–60%','60%–70%','70%–80%','90%–100%']
    assert raw.games.sum()==4
    assert raw.sample_note.str.startswith('Small sample').all()

import json
import pandas as pd
from sports_lab.quality import audit_schedule, annotate_schedule
from sports_lab.shadow import ShadowStore
from test_shadow import NOW,upcoming


def fixtures():
    return pd.DataFrame([
        dict(sport='NCAAB',event_id='first',home='Southern Utah',away='Bethesda Flames',home_id='1',away_id='9',commence_time=NOW+pd.Timedelta(hours=8)),
        dict(sport='NCAAB',event_id='second',home='Pepperdine',away='Bethesda Flames',home_id='2',away_id='9',commence_time=NOW+pd.Timedelta(hours=8,minutes=30))])


def paired():
    f=fixtures()
    f['coverage']='Established history';f['generated_at']=NOW;f['training_cutoff']=NOW-pd.Timedelta(days=5)
    f['latest_training_result']=NOW-pd.Timedelta(days=6);f['latest_calibration_result']=NOW-pd.Timedelta(days=6)
    f['elo_home_probability']=.6;f['margin_home_probability']=.8
    return f


def test_overlaps_both_flagged_but_distinct_sports_and_distant_games_are_not():
    f=fixtures()
    assert set(audit_schedule(f).event_id)=={'first','second'}
    annotated=annotate_schedule(f)
    assert annotated.data_quality_review.all()
    f.loc[1,'sport']='NCAAF'
    assert audit_schedule(f).empty
    f=fixtures();f.loc[1,'commence_time']=NOW+pd.Timedelta(hours=12)
    assert audit_schedule(f).empty


def test_provider_identity_and_ambiguous_display_name_are_detected():
    f=fixtures();f.loc[1,'away']='Bethesda University'
    assert len(audit_schedule(f))==2
    f=fixtures();f.loc[1,'away_id']='other-squad'
    assert audit_schedule(f).reason.str.contains('different provider IDs').all()


def test_duplicate_snapshot_is_not_a_new_fixture_but_revision_is_flagged():
    f=fixtures().iloc[[0]]
    assert audit_schedule(pd.concat([f,f],ignore_index=True)).empty
    modified=f.copy();modified['commence_time']=modified.commence_time+pd.Timedelta(minutes=5)
    assert audit_schedule(pd.concat([f,modified],ignore_index=True)).reason.str.contains('Same event ID').all()


def test_record_blocks_conflicts_and_preserves_cross_batch_evidence(tmp_path):
    store=ShadowStore(tmp_path/'paper.sqlite')
    frame=paired()
    assert store.record(frame,NOW)==0
    assert store.rows().empty
    other=ShadowStore(tmp_path/'other.sqlite')
    assert other.record(frame.iloc[[0]],NOW)==1
    assert other.record(frame.iloc[[1]],NOW)==0
    assert len(other.rows())==1 and other.rows().iloc[0].data_quality_review


def test_legacy_records_are_audited_without_rewriting_snapshots(tmp_path):
    store=ShadowStore(tmp_path/'paper.sqlite')
    frame=paired()
    originals=[]
    with store.connect() as db:
        for row in frame.to_dict('records'):
            encoded=json.dumps(row,default=str,sort_keys=True);originals.append(encoded)
            db.execute('INSERT INTO shadow_forecasts VALUES (?,?,?,?,?)',
                (row['sport'],row['event_id'],NOW.isoformat(),encoded,'home'))
    assert store.rows().data_quality_review.all()
    assert store.summary('NCAAB').empty
    assert store.summary('NCAAB',include_review=True).games.tolist()==[2,2]
    store.audit_saved(fixtures())
    with store.connect() as db:
        actual=[x[0] for x in db.execute('SELECT snapshot FROM shadow_forecasts ORDER BY event_id')]
    assert actual==originals

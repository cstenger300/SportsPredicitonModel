"""Prospective paired college forecasts; separate from picks and live model selection."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sports_lab.college import forecast, COLUMNS
from sports_lab.college_research import compare_models, design, week_cutoff, PROTOCOL
from sports_lab.core import utc, now_utc


def predict_pair(history,upcoming,sport,now=None):
    now=utc(now) if now is not None else now_utc()
    if upcoming.empty:
        raise ValueError('No upcoming games to record.')
    u=upcoming.copy();u['event_id']=u.event_id.astype(str);u['commence_time']=u.commence_time.map(utc)
    if (u.commence_time<=now).any():
        raise ValueError('Shadow forecasts must be recorded before the game starts.')
    week_starts=u.commence_time.map(week_cutoff)
    if week_starts.nunique()!=1:
        raise ValueError('Select games from one calendar week for each paired snapshot.')
    cutoff=min(week_starts.iloc[0],now)
    h=history.copy();h['event_id']=h.event_id.astype(str);h['commence_time']=h.commence_time.map(utc)
    h=h.loc[h.completed & h.home_score.notna() & h.away_score.notna() & h.home_score.ne(h.away_score) &
        ((h.commence_time+pd.Timedelta(days=1))<cutoff)].drop_duplicates('event_id',keep='last').sort_values(['commence_time','event_id'])
    # These margins were produced using a separate fitted model at each earlier weekly cutoff.
    audit=compare_models(h,sport,now=cutoff)['calibration_predictions']
    calibration=audit.loc[(audit.available_at<cutoff) &
        (audit.commence_time>=cutoff-pd.Timedelta(days=365)) & audit.coverage.eq('Established history')]
    if len(calibration)<50 or calibration.home_win.nunique()<2:
        raise ValueError('Not enough earlier out-of-sample predictions to calibrate the paired forecast.')
    train=h.loc[h.commence_time>=cutoff-pd.Timedelta(days=730)]
    encoder=DictVectorizer(sparse=True);x=encoder.fit_transform(design(train))
    cap=40 if sport=='NCAAB' else 45
    age=(cutoff-train.commence_time).dt.total_seconds()/86400
    model=Ridge(alpha=10.,fit_intercept=False,solver='lsqr').fit(x,
        (train.home_score-train.away_score).clip(-cap,cap),sample_weight=np.exp2(-age/365.))
    calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000))
    calibrator.fit(calibration[['predicted_margin']],calibration.home_win)
    margin=model.predict(encoder.transform(design(u)))
    p=calibrator.predict_proba(pd.DataFrame({'predicted_margin':margin}))[:,1]
    elo,_=forecast(train,u,sport,now=cutoff)
    elo=elo.set_index('event_id').home_probability
    counts=pd.concat([train.home,train.away]).value_counts()
    digest=hashlib.sha256((PROTOCOL+sport+cutoff.isoformat()+train[COLUMNS].to_json(date_format='iso')+
        calibration.to_json(date_format='iso')).encode()).hexdigest()
    output=u[['event_id','home','away','commence_time']].copy()
    output['sport']=sport;output['protocol']=PROTOCOL
    output['generated_at']=now;output['training_cutoff']=cutoff
    output['latest_training_result']=train.commence_time.max()+pd.Timedelta(days=1)
    output['latest_calibration_result']=calibration.available_at.max()
    output['elo_version']='elo-'+digest[:16];output['margin_version']='margin-'+digest[:16]
    output['elo_home_probability']=u.event_id.map(elo)
    output['margin_home_probability']=p
    output['home_history_games']=u.home.map(counts).fillna(0).astype(int)
    output['away_history_games']=u.away.map(counts).fillna(0).astype(int)
    minimum=np.minimum(output.home_history_games,output.away_history_games)
    output['coverage']=np.where(minimum>=10,'Established history',np.where(minimum==0,'Unseen participant','Limited history (1–9 games)'))
    return output


class ShadowStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS shadow_forecasts (
                sport TEXT,event_id TEXT,recorded_at TEXT,snapshot TEXT,winner TEXT,
                PRIMARY KEY(sport,event_id))''')

    def connect(self):
        return sqlite3.connect(self.path)

    def record(self,frame,now=None):
        now=utc(now) if now is not None else now_utc()
        rows=[]
        for row in frame.to_dict('records'):
            if row['coverage']!='Established history':
                continue
            if utc(row['commence_time'])<=now or utc(row['generated_at'])>now or utc(row['training_cutoff'])>now:
                raise ValueError('Only pregame paired forecasts with available inputs can be recorded.')
            if any(utc(row[key])>=utc(row['training_cutoff']) for key in ['latest_training_result','latest_calibration_result']):
                raise ValueError('Paired forecasts contain results unavailable at their cutoff.')
            if any(not np.isfinite(row[key]) or not 0<row[key]<1 for key in ['elo_home_probability','margin_home_probability']):
                raise ValueError('Paired probabilities must be finite and between zero and one.')
            rows.append((row['sport'],str(row['event_id']),now.isoformat(),json.dumps(row,default=str,sort_keys=True)))
        with self.connect() as db:
            before=db.total_changes
            db.executemany('INSERT OR IGNORE INTO shadow_forecasts VALUES (?,?,?,?,NULL)',rows)
            return db.total_changes-before

    def settle(self,results,now=None):
        now=utc(now) if now is not None else now_utc()
        required={'sport','event_id','status','winner','completed_at'}
        if not required.issubset(results):
            raise ValueError('Final results require '+', '.join(sorted(required)))
        if results.duplicated(['sport','event_id']).any():
            raise ValueError('Duplicate result events.')
        updates=[]
        with self.connect() as db:
            for row in results.to_dict('records'):
                if row['status']!='final':
                    continue
                completed=utc(row['completed_at'])
                if completed>now or row['winner'] not in ['home','away','void']:
                    raise ValueError('Invalid final result or future completion.')
                old=db.execute('SELECT snapshot FROM shadow_forecasts WHERE sport=? AND event_id=? AND winner IS NULL',
                    (row['sport'],str(row['event_id']))).fetchone()
                if not old:
                    continue
                if completed<=utc(json.loads(old[0])['commence_time']):
                    raise ValueError('Completion must follow game start.')
                updates.append((row['winner'],row['sport'],str(row['event_id'])))
            before=db.total_changes
            db.executemany('UPDATE shadow_forecasts SET winner=? WHERE sport=? AND event_id=? AND winner IS NULL',updates)
            return db.total_changes-before

    def rows(self):
        with self.connect() as db:
            records=db.execute('SELECT recorded_at,snapshot,winner FROM shadow_forecasts ORDER BY recorded_at').fetchall()
        rows=[]
        for stamp,snapshot,winner in records:
            row=json.loads(snapshot);row.update(recorded_at=stamp,winner=winner,status='Pending' if winner is None else winner)
            outcome=None if winner in [None,'void'] else int(winner=='home')
            for name in ['elo','margin']:
                p=row[name+'_home_probability']
                row[name+'_brier']=None if outcome is None else (p-outcome)**2
                row[name+'_correct']=None if outcome is None else int((p>=.5)==bool(outcome))
                row[name+'_log_loss']=None if outcome is None else float(-np.log(p if outcome else 1-p))
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self,sport):
        rows=self.rows()
        if rows.empty:
            return pd.DataFrame()
        rows=rows.loc[rows.sport.eq(sport)&rows.winner.isin(['home','away'])]
        return pd.DataFrame([dict(model=name,games=len(rows),accuracy=rows[name+'_correct'].mean(),
            brier=rows[name+'_brier'].mean(),log_loss=rows[name+'_log_loss'].mean()) for name in ['elo','margin']]) if len(rows) else pd.DataFrame()

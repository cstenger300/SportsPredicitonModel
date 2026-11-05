"""Immutable, prospective forecasts for evaluation independently of bet selection."""
import json
import sqlite3
from pathlib import Path
import pandas as pd
from sports_lab.core import utc, now_utc

class ForecastStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS forecasts (
                sport TEXT, event_id TEXT, recorded_at TEXT, model_version TEXT,
                snapshot TEXT, winner TEXT, PRIMARY KEY(sport,event_id))''')

    def connect(self):
        return sqlite3.connect(self.path)

    def record(self, forecasts, report, sport, now=None):
        now = utc(now) if now is not None else now_utc()
        records = []
        for event in forecasts.to_dict('records'):
            if utc(event['commence_time']) <= now or utc(event['metrics_as_of']) > now:
                raise ValueError('Only prospective forecasts can be saved.')
            snapshot = json.dumps({'event': event, 'validation': report}, default=str, sort_keys=True)
            records.append((sport, event['event_id'], now.isoformat(), report['model_version'], snapshot))
        with self.connect() as db:
            before = db.total_changes
            db.executemany('INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?,?,NULL)', records)
            return db.total_changes - before

    def settle(self, results, now=None):
        now = utc(now) if now is not None else now_utc()
        if not {'sport','event_id','status','winner','completed_at'}.issubset(results.columns):
            raise ValueError('Results require sport, event_id, status, winner, completed_at.')
        if results.duplicated(['sport','event_id']).any():
            raise ValueError('Duplicate result events.')
        updates = []
        with self.connect() as db:
            for result in results.to_dict('records'):
                if result['status'] != 'final':
                    continue
                completed = utc(result['completed_at'])
                if completed > now or result['winner'] not in ['home','away','void']:
                    raise ValueError('Invalid final result or future completion time.')
                row = db.execute('SELECT snapshot FROM forecasts WHERE sport=? AND event_id=? AND winner IS NULL',
                                 (result['sport'], str(result['event_id']))).fetchone()
                if not row:
                    continue
                if completed <= utc(json.loads(row[0])['event']['commence_time']):
                    raise ValueError('Completion must follow event start.')
                updates.append((result['winner'], result['sport'], str(result['event_id'])))
            before = db.total_changes
            db.executemany('UPDATE forecasts SET winner=? WHERE sport=? AND event_id=? AND winner IS NULL', updates)
            return db.total_changes - before

    def rows(self):
        with self.connect() as db:
            rows = db.execute('SELECT sport,event_id,recorded_at,model_version,snapshot,winner FROM forecasts ORDER BY recorded_at').fetchall()
        records = []
        for sport, event_id, recorded, version, snapshot, winner in rows:
            data = json.loads(snapshot)
            e = data['event']
            p = e['home_probability']
            outcome = None if winner in [None, 'void'] else int(winner == 'home')
            records.append(dict(sport=sport,event_id=event_id,recorded_at=recorded,model_version=version,
                home=e['home'],away=e['away'],home_probability=p,winner=winner,
                correct=None if outcome is None else int((p >= .5) == bool(outcome)),
                brier=None if outcome is None else (p-outcome)**2,
                baseline_brier=None if outcome is None else (data['validation']['baseline_probability']-outcome)**2))
        return pd.DataFrame(records)

    def summary(self):
        rows = self.rows()
        if rows.empty:
            return pd.DataFrame()
        rows = rows.loc[rows.winner.isin(['home','away'])]
        if rows.empty:
            return pd.DataFrame()
        summary = rows.groupby(['sport','model_version']).agg(games=('event_id','size'),
            accuracy=('correct','mean'),brier=('brier','mean'),baseline_brier=('baseline_brier','mean')).reset_index()
        summary['status'] = summary.apply(lambda r: 'Collecting evidence' if r.games < 100 else
            ('Review model: below baseline' if r.brier >= r.baseline_brier else 'Above baseline; continue monitoring'),axis=1)
        return summary

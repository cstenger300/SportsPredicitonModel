"""Chronological moneyline models using pregame metric differences."""
import json
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

METRICS = {
    'NBA': ['net_rating', 'effective_fg_pct', 'turnover_pct', 'offensive_rebound_pct', 'free_throw_rate', 'rest_days'],
    'NHL': ['expected_goal_share', 'corsi_share', 'high_danger_share', 'power_play_pct', 'penalty_kill_pct', 'goalie_goals_saved_above_expected'],
    'NFL': ['epa_per_play', 'success_rate', 'explosive_play_rate', 'sack_rate', 'opponent_adjusted_rating', 'rest_days'],
    'NCAAB': ['adjusted_efficiency_margin', 'effective_fg_pct', 'turnover_pct', 'offensive_rebound_pct', 'free_throw_rate', 'strength_of_schedule'],
    'Tennis': ['surface_elo', 'serve_points_won_pct', 'return_points_won_pct', 'hold_pct', 'break_pct', 'rest_days'],
}

def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Dates must include a timezone offset.')
    return stamp.tz_convert('UTC')

def now_utc():
    return pd.Timestamp.now(tz='UTC')

def validate_games(frame, sport, historical=False):
    required = ['event_id', 'home', 'away', 'commence_time', 'metrics_as_of'] + METRICS[sport]
    if historical:
        required += ['home_win', 'result_available_at']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    frame = frame.copy()
    if frame.empty or frame[required].isna().any().any() or frame.event_id.duplicated().any():
        raise ValueError('Supply nonempty, complete rows with unique event IDs.')
    for col in ['event_id', 'home', 'away']:
        frame[col] = frame[col].astype(str)
        if frame[col].str.strip().eq('').any():
            raise ValueError('Event and participant names must be nonempty.')
    if frame.home.eq(frame.away).any():
        raise ValueError('Participants must differ.')
    for col in ['commence_time', 'metrics_as_of'] + (['result_available_at'] if historical else []):
        frame[col] = frame[col].map(utc)
    if (frame.metrics_as_of >= frame.commence_time).any():
        raise ValueError('Metrics must have been available before the event started.')
    frame[METRICS[sport]] = frame[METRICS[sport]].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame[METRICS[sport]].to_numpy()).all():
        raise ValueError('Metrics must be finite numbers.')
    if historical:
        if not frame.home_win.isin([0, 1]).all():
            raise ValueError('home_win must be 0 or 1; omit ties, voids, and unfinished events.')
        if (frame.result_available_at <= frame.commence_time).any():
            raise ValueError('Result availability must follow event start.')
    return frame.sort_values(['commence_time', 'event_id']).reset_index(drop=True)

def fit_model(history, upcoming, sport, now=None):
    now = utc(now) if now is not None else now_utc()
    h = validate_games(history, sport, True)
    u = validate_games(upcoming, sport)
    if (u.commence_time <= now).any() or (u.metrics_as_of > now).any():
        raise ValueError('Forecast events must be upcoming and metrics cannot be from the future.')
    h = h.loc[(h.result_available_at < now) & (h.result_available_at < u.metrics_as_of.min())]
    if set(h.event_id) & set(u.event_id):
        raise ValueError('Historical and upcoming event IDs overlap.')
    if len(h) < 100:
        raise ValueError('At least 100 completed historical events are required for this sport.')
    metrics = METRICS[sport]
    split = int(len(h) * .8)
    validation = h.iloc[split:]
    train = h.iloc[:split]
    train = train.loc[train.result_available_at < validation.metrics_as_of.min()]
    if len(train) < 50 or train.home_win.nunique() != 2 or validation.home_win.nunique() != 2:
        raise ValueError('Chronological training and validation need both outcome classes and 50 training events.')
    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000))
    estimator = model().fit(train[metrics], train.home_win)
    probabilities = estimator.predict_proba(validation[metrics])[:, 1]
    baseline = np.full(len(validation), train.home_win.mean())
    brier = brier_score_loss(validation.home_win, probabilities)
    baseline_brier = brier_score_loss(validation.home_win, baseline)
    report = {'training_events': len(train), 'validation_events': len(validation),
              'brier': float(brier), 'baseline_brier': float(baseline_brier),
              'log_loss': float(log_loss(validation.home_win, probabilities)),
              'beats_baseline': bool(brier < baseline_brier),
              'latest_result': h.result_available_at.max().isoformat(),
              'baseline_probability': float(train.home_win.mean())}
    estimator = model().fit(h[metrics], h.home_win)
    model_identity = {'sport': sport, 'algorithm': 'standardized-logistic-C0.3-v1',
                      'training': h[['event_id', 'commence_time', 'metrics_as_of', 'result_available_at', 'home_win'] + metrics].to_json(date_format='iso'),
                      'coefficients': estimator[1].coef_.tolist(), 'intercept': estimator[1].intercept_.tolist(),
                      'means': estimator[0].mean_.tolist(), 'scales': estimator[0].scale_.tolist()}
    report['model_version'] = hashlib.sha256(json.dumps(model_identity, sort_keys=True).encode()).hexdigest()[:16]
    u['home_probability'] = estimator.predict_proba(u[metrics])[:, 1]
    z = estimator[0].transform(u[metrics])
    coefficients = estimator[1].coef_[0]
    u['why'] = [ '; '.join(f'{metrics[i]}: {row[i] * coefficients[i]:+.2f} log-odds toward home'
                               for i in np.argsort(-np.abs(row * coefficients))[:3]) for row in z]
    u['out_of_distribution'] = (np.abs(z) > 4).any(axis=1)
    return u, report

def decimal_odds(american):
    value = float(american)
    if not np.isfinite(value) or abs(value) < 100:
        raise ValueError('American odds must have absolute magnitude at least 100.')
    return 1 + (value / 100 if value > 0 else 100 / abs(value))

def rank_quotes(forecasts, quotes, report, sport, now=None):
    now = utc(now) if now is not None else now_utc()
    required = {'event_id', 'bookmaker', 'side', 'american_odds', 'observed_at', 'source_updated_at'}
    if required - set(quotes):
        raise ValueError('Quote CSV requires: ' + ', '.join(sorted(required)))
    rows = []
    for q in quotes.to_dict('records'):
        if q['side'] not in ['home', 'away'] or not str(q['bookmaker']).strip():
            raise ValueError('Quotes require bookmaker and side home/away.')
        event = forecasts.loc[forecasts.event_id.eq(str(q['event_id']))]
        if len(event) != 1:
            raise ValueError('Quote event ID does not match a forecast.')
        event = event.iloc[0]
        observed, updated = utc(q['observed_at']), utc(q['source_updated_at'])
        if updated > observed or observed > now:
            raise ValueError('Quote timestamps are inconsistent or in the future.')
        if observed >= event.commence_time:
            raise ValueError('Only pregame quotes are supported.')
        price = decimal_odds(q['american_odds'])
        p = event.home_probability if q['side'] == 'home' else 1 - event.home_probability
        ev = p * price - 1
        fresh = now - min(observed, updated) <= pd.Timedelta(minutes=15)
        eligible = bool(fresh and event.commence_time > now and report['beats_baseline'] and
                        not event.out_of_distribution and ev >= .03)
        rows.append({**q, 'sport': sport, 'matchup': f'{event.away} at {event.home}',
                     'selection': event[q['side']], 'commence_time': event.commence_time.isoformat(),
                     'probability': float(p), 'break_even': 1 / price, 'edge': p - 1 / price,
                     'expected_return': ev, 'eligible': eligible,
                     'confidence': 'Research estimate — not calibrated',
                     'status': 'Paper candidate' if eligible else 'Pass: stale, low edge, model validation, or unusual inputs',
                     'why': event.why, 'validation': report})
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    # Keep the latest quote for each book and side; never cherry-pick replaced prices.
    result['observed_at'] = result.observed_at.map(utc)
    result = result.sort_values('observed_at').drop_duplicates(['event_id', 'bookmaker', 'side'], keep='last')
    return result.sort_values('expected_return', ascending=False).reset_index(drop=True)

class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS picks (sport TEXT, event_id TEXT, recorded_at TEXT, snapshot TEXT, result TEXT, profit REAL, PRIMARY KEY(sport,event_id))')

    def connect(self):
        return sqlite3.connect(self.path)

    def record(self, pick, now=None):
        now = utc(now) if now is not None else now_utc()
        if not pick['eligible'] or now >= utc(pick['commence_time']):
            raise ValueError('Only eligible pregame paper candidates may be recorded.')
        if any(now - utc(pick[k]) > pd.Timedelta(minutes=15) or utc(pick[k]) > now
               for k in ['observed_at', 'source_updated_at']):
            raise ValueError('Refresh stale quotes before recording.')
        with self.connect() as db:
            cursor = db.execute('INSERT OR IGNORE INTO picks VALUES (?,?,?,?,NULL,NULL)',
                (pick['sport'], pick['event_id'], now.isoformat(), json.dumps(pick, default=str)))
            return cursor.rowcount == 1

    def rows(self):
        with self.connect() as db:
            records = db.execute('SELECT sport,event_id,recorded_at,snapshot,result,profit FROM picks ORDER BY recorded_at DESC').fetchall()
        return pd.DataFrame([{'sport': s, 'event_id': e, 'recorded_at': t, **json.loads(p), 'result': r, 'profit_units': v}
                             for s, e, t, p, r, v in records])

    def settle(self, results, now=None):
        now = utc(now) if now is not None else now_utc()
        required = {'sport', 'event_id', 'status', 'winner', 'completed_at'}
        if required - set(results):
            raise ValueError('Results require: ' + ', '.join(sorted(required)))
        if results.duplicated(['sport', 'event_id']).any():
            raise ValueError('Duplicate result events.')
        updates = []
        for r in results.to_dict('records'):
            if r['status'] != 'final':
                continue
            completed = utc(r['completed_at'])
            if completed > now or r['winner'] not in ['home', 'away', 'void']:
                raise ValueError('Final results require a past completion and home/away/void winner.')
            with self.connect() as db:
                row = db.execute('SELECT snapshot FROM picks WHERE sport=? AND event_id=? AND result IS NULL',
                                 (r['sport'], str(r['event_id']))).fetchone()
            if not row:
                continue
            pick = json.loads(row[0])
            if completed <= utc(pick['commence_time']):
                raise ValueError('Completion must follow event start.')
            outcome = 'void' if r['winner'] == 'void' else ('won' if r['winner'] == pick['side'] else 'lost')
            profit = 0 if outcome == 'void' else (decimal_odds(pick['american_odds']) - 1 if outcome == 'won' else -1)
            updates.append((outcome, profit, r['sport'], str(r['event_id'])))
        with self.connect() as db:
            db.executemany('UPDATE picks SET result=?,profit=? WHERE sport=? AND event_id=? AND result IS NULL', updates)
        return len(updates)

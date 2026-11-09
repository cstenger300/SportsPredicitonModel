"""Append-only multi-sport source snapshots. Collection is not model validation."""
from pathlib import Path
import json
import hashlib
import sqlite3
import uuid
import pandas as pd
import requests
from sports_lab import provider
from sports_lab.core import utc, now_utc
from sports_lab.college import fetch_games

SPORTS=['NBA','NHL','NFL','NCAAB','NCAAF','Tennis']
PUBLIC_PATHS={'NBA':'basketball/nba','NHL':'hockey/nhl','NFL':'football/nfl'}


def records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso'))


def snapshot_odds(key,events,quotes):
    if not events.empty:
        events=events.copy()
        observed=now_utc()
        events['retrieved_before_scheduled_start']=events.commence_time.map(utc)>observed
    return {'provider':'The Odds API','sport_key':key,'events':records(events),'quotes':records(quotes)}


def odds_payload(key):
    events,quotes=provider.odds(key)
    return snapshot_odds(key,events,quotes)


def public_games(sport,date,cache):
    if sport in ['NCAAB','NCAAF']:
        return fetch_games(sport,date,date,cache,refresh=True)
    if sport not in PUBLIC_PATHS:
        raise ValueError('No public tennis schedule adapter is connected. Select an active tournament for odds collection.')
    try:
        response=requests.get('https://site.api.espn.com/apis/site/v2/sports/'+PUBLIC_PATHS[sport]+'/scoreboard',
            params={'dates':pd.Timestamp(date).strftime('%Y%m%d'),'limit':1000},timeout=(15,60))
        if not response.ok:
            raise ValueError(f'Public schedule provider returned HTTP {response.status_code}.')
        payload=response.json()
    except requests.RequestException:
        raise ValueError('Public schedule connection failed; check access to site.api.espn.com.') from None
    except ValueError as exc:
        if str(exc).startswith('Public schedule provider'):
            raise
        raise ValueError('Public schedule provider returned invalid JSON.') from None
    if not isinstance(payload,dict) or not isinstance(payload.get('events'),list):
        raise ValueError('Public schedule provider returned an unexpected format.')
    if len(payload['events'])>=1000:
        raise ValueError('Public schedule may be truncated; snapshot was not accepted.')
    rows=[]
    for event in payload['events']:
        for competition in event.get('competitions',[]):
            teams={c.get('homeAway'):c for c in competition.get('competitors',[])}
            if set(teams)!={'home','away'}:
                continue
            home,away=teams['home'],teams['away']
            complete=bool(competition.get('status',event.get('status',{})).get('type',{}).get('completed',False))
            rows.append(dict(event_id=str(event['id']),home=home['team']['displayName'],away=away['team']['displayName'],
                home_id=str(home['team'].get('id','')),away_id=str(away['team'].get('id','')),
                commence_time=event['date'],completed=complete,
                home_score=home.get('score') if complete else None,away_score=away.get('score') if complete else None))
    return pd.DataFrame(rows)


class CollectionStore:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'runs.sqlite'
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,sport TEXT,source TEXT,requested_at TEXT,observed_at TEXT,
                status TEXT,row_count INTEGER,snapshot TEXT,sha256 TEXT,note TEXT)''')
            db.execute('CREATE INDEX IF NOT EXISTS feed_runs ON runs(sport,source,requested_at)')

    def connect(self):
        return sqlite3.connect(self.path,timeout=15)

    def claim(self,sport,source,cooldown_minutes,now=None):
        now=utc(now) if now is not None else now_utc()
        if sport not in SPORTS or source not in ['public','odds','metrics']:
            raise ValueError('Unsupported collection sport or source.')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT requested_at FROM runs WHERE sport=? AND source=? ORDER BY requested_at DESC LIMIT 1',(sport,source)).fetchone()
            if prior and now-utc(prior[0])<pd.Timedelta(minutes=cooldown_minutes):
                return None
            identifier=uuid.uuid4().hex
            db.execute('INSERT INTO runs VALUES (?,?,?,?,NULL,?,NULL,NULL,NULL,?)',
                (identifier,sport,source,now.isoformat(),'running','Collection in progress'))
            return identifier

    def finish(self,identifier,payload,row_count,now=None):
        now=utc(now) if now is not None else now_utc()
        with self.connect() as db:
            row=db.execute('SELECT sport,source,requested_at,status FROM runs WHERE run_id=?',(identifier,)).fetchone()
        if not row or row[3]!='running':
            raise ValueError('Collection run is not active.')
        sport,source,requested,_=row
        snapshot={'sport':sport,'source':source,'requested_at':requested,'observed_at':now.isoformat(),
                  'timestamp_basis':'retrieval time; historical observations are not backdated','data':payload}
        content=json.dumps(snapshot,sort_keys=True,allow_nan=False).encode()
        relative=Path('snapshots')/sport/source/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+identifier+'.json')
        target=self.directory/relative;target.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive creation protects prior snapshots even in repeated/manual collections.
        with target.open('xb') as output:
            output.write(content)
        digest=hashlib.sha256(content).hexdigest()
        with self.connect() as db:
            db.execute('UPDATE runs SET observed_at=?,status=?,row_count=?,snapshot=?,sha256=?,note=? WHERE run_id=?',
                (now.isoformat(),'success',int(row_count),str(relative),digest,'Empty response' if row_count==0 else 'Snapshot archived',identifier))

    def fail(self,identifier,error,now=None):
        now=utc(now) if now is not None else now_utc()
        # Source adapters return credential-free errors. Never store raw request exceptions/URLs.
        note=str(error) if isinstance(error,ValueError) else f'Collection failed ({type(error).__name__}); inspect provider availability or local storage.'
        if '://' in note or 'apikey' in note.lower() or 'api_key' in note.lower() or 'token=' in note.lower():
            note='Collection failed; check provider access and credential settings. Request details were omitted.'
        with self.connect() as db:
            db.execute('UPDATE runs SET observed_at=?,status=?,note=? WHERE run_id=?',(now.isoformat(),'error',note,identifier))

    def archive_odds(self,sport,key,events,quotes):
        # An explicit manual price fetch has already occurred; archive it without another request.
        identifier=self.claim(sport,'odds',0)
        payload=snapshot_odds(key,events,quotes)
        try:
            self.finish(identifier,payload,len(quotes))
        except Exception as exc:
            self.fail(identifier,exc)
            raise ValueError('Prices loaded, but the local snapshot could not be archived.') from None

    def archive_metrics(self,sport,history,upcoming,manifest):
        identifier=self.claim(sport,'metrics',0)
        try:
            self.finish(identifier,{'basis':'Generated from downloaded source files; reconstructed historical inputs are not original pregame archives',
                'history':records(history),'upcoming':records(upcoming),'source_manifest':manifest},len(history)+len(upcoming))
        except Exception as exc:
            self.fail(identifier,exc)
            raise ValueError('Metrics generated, but the local snapshot could not be archived.') from None

    def collect(self,sports,date,include_public=True,include_odds=False,tennis_key=None,cooldown_minutes=15,now=None):
        now=utc(now) if now is not None else now_utc()
        if any(s not in SPORTS for s in sports):
            raise ValueError('Unsupported sport selection.')
        output=[]
        for sport in dict.fromkeys(sports):
            for source,enabled in [('public',include_public),('odds',include_odds)]:
                if not enabled:
                    continue
                if source=='public' and sport=='Tennis':
                    output.append(dict(sport=sport,source=source,status='not connected',note='No public tennis adapter connected.'));continue
                key=tennis_key if sport=='Tennis' else provider.KEYS[sport]
                if source=='odds' and (not key or (sport=='Tennis' and not key.startswith('tennis_'))):
                    output.append(dict(sport=sport,source=source,status='not configured',note='Select an active tennis tournament.'));continue
                identifier=self.claim(sport,source,cooldown_minutes,now)
                if identifier is None:
                    output.append(dict(sport=sport,source=source,status='cooldown',note='Recent attempt or snapshot; no request made.'));continue
                try:
                    if source=='public':
                        games=public_games(sport,date,self.directory/'request_cache')
                        payload={'provider':'ESPN','schedule_date':str(date),'games':records(games)};count=len(games)
                    else:
                        payload=odds_payload(key);count=len(payload['quotes'])
                    self.finish(identifier,payload,count)
                    output.append(dict(sport=sport,source=source,status='success',rows=count,note='Snapshot archived'))
                except Exception as exc:
                    self.fail(identifier,exc)
                    output.append(dict(sport=sport,source=source,status='error',note='Collection failed; see feed status.'))
        return pd.DataFrame(output)

    def runs(self):
        with self.connect() as db:
            return pd.read_sql_query('SELECT * FROM runs ORDER BY requested_at DESC',db)

    def status(self):
        runs=self.runs();rows=[]
        for sport in SPORTS:
            for source in ['public','odds','metrics']:
                feed=runs.loc[runs.sport.eq(sport)&runs.source.eq(source)]
                success=feed.loc[feed.status.eq('success')]
                latest=feed.iloc[0] if len(feed) else None
                missing_adapter=(source=='public' and sport=='Tennis') or (source=='metrics' and sport!='NFL')
                rows.append(dict(sport=sport,feed=source,latest_status=latest.status if latest is not None else 'not connected' if missing_adapter else 'not collected',
                    last_attempt=latest.requested_at if latest is not None else None,
                    last_success=success.iloc[0].observed_at if len(success) else None,
                    last_rows=int(success.iloc[0].row_count) if len(success) else None,
                    note=latest.note if latest is not None else 'No snapshot collected yet'))
        return pd.DataFrame(rows)


def coverage_table():
    return pd.DataFrame([dict(sport=sport,public_schedules_results='Not connected' if sport=='Tennis' else 'Public adapter available; collect to verify',
        sportsbook_odds='API key + active tournament required' if sport=='Tennis' else 'The Odds API key required',
        advanced_metrics='Existing NFL download workflow' if sport=='NFL' else 'Not connected',
        player_availability='Not connected to multi-sport models',
        model='Experimental NFL metrics' if sport=='NFL' else 'Experimental college Elo + paired research' if sport in ['NCAAB','NCAAF'] else 'Manual metric imports only') for sport in SPORTS])

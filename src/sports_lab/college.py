"""Public college schedules and chronological Elo baseline; not an advanced-metric model."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sports_lab.core import utc, now_utc

PATHS={'NCAAB':'basketball/mens-college-basketball','NCAAF':'football/college-football'}
COLUMNS=['event_id','home','away','commence_time','completed','home_score','away_score','neutral']


def fetch_games(sport, start, end, cache, refresh=False):
    """Split bounded requests when the provider's response may be truncated."""
    start,end=pd.Timestamp(start).date(),pd.Timestamp(end).date()
    if start>end:
        raise ValueError('Start date must precede end date.')
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    target=cache/f'{sport}_{start}_{end}.json'
    if target.exists() and not refresh:
        payload=json.loads(target.read_text())
    else:
        try:
            response=requests.get('https://site.api.espn.com/apis/site/v2/sports/'+PATHS[sport]+'/scoreboard',
                params={'dates':start.strftime('%Y%m%d') if start==end else start.strftime('%Y%m%d')+'-'+end.strftime('%Y%m%d'),'groups':50 if sport=='NCAAB' else 80,'limit':1000},timeout=(15,60))
            response.raise_for_status();payload=response.json()
        except requests.HTTPError as exc:
            status=exc.response.status_code if exc.response is not None else 'unknown'
            if status==400 and start<end:
                frames=[];events=[]
                for day in pd.date_range(start,end):
                    frames.append(fetch_games(sport,day,day,cache,refresh))
                    day_file=cache/f'{sport}_{day.date()}_{day.date()}.json'
                    events.extend(json.loads(day_file.read_text())['events'])
                # Cache the range only after every daily request completed successfully.
                temporary=target.with_suffix('.tmp')
                temporary.write_text(json.dumps({'events':events}))
                temporary.replace(target)
                return pd.concat(frames,ignore_index=True).drop_duplicates('event_id')
            raise ValueError(f'Public college feed returned HTTP {status} for {sport}, {start} through {end}. No invented data was substituted.') from None
        except requests.Timeout:
            raise ValueError(f'Public college request timed out for {sport}, {start} through {end}. Retry; cached completed downloads will be reused.') from None
        except requests.RequestException:
            raise ValueError('Cannot connect to site.api.espn.com. Check the Python connection, proxy, and certificate trust.') from None
        except ValueError:
            raise ValueError('Public college feed returned invalid JSON.') from None
        if not isinstance(payload,dict) or not isinstance(payload.get('events'),list):
            raise ValueError('Public college feed returned an unexpected format.')
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(payload));tmp.replace(target)
    events=payload['events']
    if len(events)>=1000:
        if start==end:
            raise ValueError('College feed appears truncated even for one day.')
        midpoint=start+(end-start)//2
        return pd.concat([fetch_games(sport,start,midpoint,cache,refresh),fetch_games(sport,midpoint+pd.Timedelta(days=1),end,cache,refresh)],ignore_index=True).drop_duplicates('event_id')
    rows=[]
    for event in events:
        for competition in event.get('competitions',[]):
            competitors={c.get('homeAway'):c for c in competition.get('competitors',[])}
            if set(competitors)!={'home','away'}:
                continue
            h,a=competitors['home'],competitors['away']
            completed=bool(competition.get('status',event.get('status',{})).get('type',{}).get('completed',False))
            rows.append(dict(event_id=str(event['id']),home=h['team']['displayName'],away=a['team']['displayName'],
                commence_time=event['date'],completed=completed,home_score=float(h['score']) if completed and h.get('score') is not None else np.nan,
                away_score=float(a['score']) if completed and a.get('score') is not None else np.nan,neutral=bool(competition.get('neutralSite',False))))
    return pd.DataFrame(rows,columns=COLUMNS)


def forecast(history, upcoming, sport, now=None):
    now=utc(now) if now is not None else now_utc()
    h=history.copy();u=upcoming.copy()
    if u.empty:
        raise ValueError('No upcoming college games found for that date.')
    for frame in [h,u]:
        frame['commence_time']=frame.commence_time.map(utc)
    if (u.commence_time<=now).any():
        raise ValueError('Only games that have not started can be forecast.')
    # Conservative availability estimate; no authentic historic publication timestamps are claimed.
    h=h.loc[h.completed & h.home_score.notna() & h.away_score.notna() & h.home_score.ne(h.away_score) &
            ((h.commence_time+pd.Timedelta(days=1))<min(now,u.commence_time.min()))].copy()
    h=h.sort_values(['commence_time','event_id']).drop_duplicates('event_id')
    if len(h)<100:
        raise ValueError('Need at least 100 completed college games to evaluate this baseline.')
    ratings={};counts={};ratings_season={};predictions=[];outcomes=[];audit=[]
    def season(stamp):
        return stamp.year if stamp.month>=7 else stamp.year-1
    def rating(team,stamp):
        yr=season(stamp)
        if team in ratings_season and ratings_season[team]!=yr:
            ratings[team]=1500+.5*(ratings[team]-1500)
        ratings_season[team]=yr
        return ratings.get(team,1500.)
    def probability(home,away,stamp,neutral):
        return 1/(1+10**(-(rating(home,stamp)-rating(away,stamp)+(0 if neutral else 55))/400))
    # Update only after a conservative completion boundary, never at scheduled start.
    pending=[]
    def apply_updates(before):
        ready=[r for r in pending if r[0]<before]
        for available,home,away,p,y in ready:
            ratings[home]=ratings.get(home,1500)+24*(y-p)
            ratings[away]=ratings.get(away,1500)-24*(y-p)
            counts[home]=counts.get(home,0)+1;counts[away]=counts.get(away,0)+1
            pending.remove((available,home,away,p,y))
    for r in h.itertuples():
        apply_updates(r.commence_time)
        p=probability(r.home,r.away,r.commence_time,r.neutral);y=int(r.home_score>r.away_score)
        predictions.append(p);outcomes.append(y)
        pending.append((r.commence_time+pd.Timedelta(days=1),r.home,r.away,p,y))
        audit.append(dict(event_id=r.event_id,probability=p,outcome=y))
    apply_updates(now)
    split=max(50,int(len(h)*.8));ys=np.array(outcomes[split:]);ps=np.array(predictions[split:])
    available_baseline=h.iloc[:split].loc[(h.iloc[:split].commence_time+pd.Timedelta(days=1))<h.iloc[split].commence_time]
    if len(available_baseline)<50:
        raise ValueError('Not enough completed results were available before the chronological validation cutoff.')
    baseline=float((available_baseline.home_score>available_baseline.away_score).mean());brier=float(np.mean((ps-ys)**2));base=float(np.mean((baseline-ys)**2))
    version=hashlib.sha256((sport+h[COLUMNS].to_json(date_format='iso')+'elo-k24-h55-shrink50-v1').encode()).hexdigest()[:16]
    report=dict(model_version=version,training_events=split,validation_events=len(ys),brier=brier,baseline_brier=base,
        baseline_probability=baseline,beats_baseline=brier<base,latest_result=(h.commence_time.max()+pd.Timedelta(days=1)).isoformat(),
        model_type='Experimental results-only Elo; no advanced metrics, injuries or roster inputs',
        limitations='Sequential replay uses reconstructed completion times and 50% offseason rating shrinkage; not archived point-in-time data or calibrated probabilities.')
    rows=[]
    for r in u.itertuples():
        p=probability(r.home,r.away,r.commence_time,r.neutral)
        rows.append(dict(event_id=str(r.event_id),home=r.home,away=r.away,commence_time=r.commence_time,metrics_as_of=now,
            home_probability=p,out_of_distribution=min(counts.get(r.home,0),counts.get(r.away,0))<10,
            why=f'Results-only Elo: home {ratings.get(r.home,1500):.0f}, away {ratings.get(r.away,1500):.0f}; '+
                ('neutral venue; ' if r.neutral else '55-point home-rating adjustment; ')+
                f'available history home {counts.get(r.home,0)} games, away {counts.get(r.away,0)} games. No player or advanced-stat inputs.'))
    return pd.DataFrame(rows),report


def name_key(name):
    return re.sub(r'[^a-z0-9]','',name.lower().replace('&','and'))


def align_events(upcoming,events,quotes):
    rows=[];ids=[]
    for r in upcoming.to_dict('records'):
        matches=events.loc[events.home.map(name_key).eq(name_key(r['home'])) & events.away.map(name_key).eq(name_key(r['away'])) &
            (pd.to_datetime(events.commence_time,utc=True)-utc(r['commence_time'])).abs().le(pd.Timedelta(minutes=15))]
        if len(matches)!=1:
            continue
        event=matches.iloc[0];r['public_event_id']=r['event_id'];r.update(event_id=str(event.event_id),home=event.home,away=event.away,commence_time=utc(event.commence_time));rows.append(r);ids.append(str(event.event_id))
    if not rows:
        raise ValueError('No exact college team/time matches with sportsbook events. Public and bookmaker names can differ; no fuzzy matches were guessed.')
    return pd.DataFrame(rows),quotes.loc[quotes.event_id.astype(str).isin(ids)].copy()


def load(root,sport,date,now=None):
    now=utc(now) if now is not None else now_utc()
    date=pd.Timestamp(date).date();cache=Path(root)/'data/sports_lab/college'
    today=fetch_games(sport,date,date,cache,refresh=True)
    if today.empty:
        raise ValueError('No public schedule events found on the selected date.')
    today=today.loc[today.commence_time.map(utc)>now]
    if today.empty:
        raise ValueError('All games on this date have started or no upcoming events are available.')
    start=pd.Timestamp(date)-pd.Timedelta(days=550)
    end=pd.Timestamp(date)-pd.Timedelta(days=1)
    chunks=[]
    # Historical baseline covers ordinary competitive seasons, not exhibitions/spring games.
    if sport=='NCAAB':
        eligible=[day for day in pd.date_range(start,end) if day.month in [11,12,1,2,3] or (day.month==4 and day.day<=15)]
    else:
        eligible=[day for day in pd.date_range(start,end) if day.month in [8,9,10,11,12,1]]
    windows=[]
    for day in eligible:
        if not windows or day>windows[-1][-1]+pd.Timedelta(days=1) or len(windows[-1])>=14:
            windows.append([day])
        else:
            windows[-1].append(day)
    for window in windows:
        day,last=window[0],window[-1]
        chunks.append(fetch_games(sport,day,last,cache,refresh=last.date()>=(now-pd.Timedelta(days=3)).date()))
    history=pd.concat(chunks,ignore_index=True).drop_duplicates('event_id')
    return history,today


def save_mapping(root,sport,upcoming):
    target=Path(root)/'data/sports_lab/college'/f'{sport}_event_mapping.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    new=upcoming[['public_event_id','event_id']].astype(str)
    old=pd.read_csv(target,dtype=str) if target.exists() else pd.DataFrame(columns=new.columns)
    pd.concat([old,new],ignore_index=True).drop_duplicates('public_event_id',keep='last').to_csv(target,index=False)


def final_results(root,sport,now=None):
    now=utc(now) if now is not None else now_utc()
    cache=Path(root)/'data/sports_lab/college'
    date=now.tz_convert('America/Chicago').date()
    games=fetch_games(sport,pd.Timestamp(date)-pd.Timedelta(days=3),date,cache,refresh=True)
    target=cache/f'{sport}_event_mapping.csv'
    mapping=dict(pd.read_csv(target,dtype=str)[['public_event_id','event_id']].itertuples(index=False,name=None)) if target.exists() else {}
    rows=[]
    for r in games.itertuples():
        if not r.completed or pd.isna(r.home_score) or pd.isna(r.away_score) or r.home_score==r.away_score or utc(r.commence_time)>=now:
            continue
        winner='home' if r.home_score>r.away_score else 'away'
        for identifier in set([str(r.event_id),mapping.get(str(r.event_id),str(r.event_id))]):
            rows.append(dict(sport=sport,event_id=identifier,status='final',winner=winner,completed_at=now.isoformat()))
    return pd.DataFrame(rows,columns=['sport','event_id','status','winner','completed_at'])

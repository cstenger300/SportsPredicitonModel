"""Public nflverse integration. Historical inputs are reconstructed, not archived live snapshots."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import requests
from nfl_model.data import COLUMNS, TEAM_ALIASES, aggregate_games, download_season
from nfl_model.features import build_features
from nfl_model.prediction import SCHEDULE_URL
from nfl_model.sources import TEAM_NAMES
from sports_lab.core import METRICS, now_utc, utc


def schedule(cache, refresh=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / 'games.csv'
    if refresh or not target.exists():
        try:
            response = requests.get(SCHEDULE_URL, timeout=(15,60))
            response.raise_for_status()
            frame = pd.read_csv(__import__('io').BytesIO(response.content))
        except (requests.RequestException, ValueError):
            raise ValueError('NFL schedule download failed; check Internet access and retry.') from None
        if not {'game_id','home_team','away_team','gameday','gametime','season','week'}.issubset(frame):
            raise ValueError('NFL schedule is missing required fields.')
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(response.content)
        temporary.replace(target)
    frame = pd.read_csv(target)
    frame['commence_time'] = pd.to_datetime(frame.gameday + ' ' + frame.gametime, errors='coerce').dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    for col in ['home_team','away_team']:
        frame[col] = frame[col].replace(TEAM_ALIASES)
    return frame


def metric_rows(team_games, games, now=None):
    """Shifted rolling metrics, with no target/future results used for upcoming games."""
    now = utc(now) if now is not None else now_utc()
    if games.empty:
        raise ValueError('No upcoming NFL games for the selected date.')
    cutoff = pd.to_datetime(games.gameday).min()
    history = team_games.loc[pd.to_datetime(team_games.game_date) < cutoff].copy()
    placeholders = []
    for g in games.to_dict('records'):
        for team, opponent, home in [(g['home_team'],g['away_team'],1),(g['away_team'],g['home_team'],0)]:
            placeholders.append(dict(game_id=g['game_id'],season=g['season'],week=g['week'],game_date=pd.Timestamp(g['gameday']),
                                     team=team,opponent=opponent,is_home=home,points=np.nan,points_allowed=np.nan))
    features,_ = build_features(pd.concat([history,pd.DataFrame(placeholders)],ignore_index=True), 'enhanced')
    home = features.loc[features.is_home.eq(1)].copy()
    mappings = {'epa_per_play':'recent_off_epa','success_rate':'recent_off_success',
                'explosive_play_rate':'recent_off_explosive','sack_rate':'recent_def_sack_rate','rest_days':'rest_days'}
    for dest,src in mappings.items():
        home[dest] = home[src] - home['opponent_'+src]
    home['opponent_adjusted_rating'] = (home.adjusted_off_epa - home.adjusted_def_allowed_epa) - (home.opponent_adjusted_off_epa-home.opponent_adjusted_def_allowed_epa)
    home['event_id'] = home.game_id.astype(str)
    home['home'],home['away'] = home.team,home.opponent
    # Conservative date boundaries; no claim these are authentic historic source publication times.
    dates = pd.to_datetime(home.game_date).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    home['commence_time'] = dates
    home['metrics_as_of'] = dates - pd.Timedelta(seconds=1)
    home['result_available_at'] = dates + pd.Timedelta(days=1)
    home['home_win'] = (home.points > home.points_allowed).astype(int)
    past = home.loc[home.points.notna() & home.points.ne(home.points_allowed)].dropna(subset=METRICS['NFL']).copy()
    future = home.loc[home.game_id.isin(games.game_id)].copy()
    for i,row in future.iterrows():
        g = games.loc[games.game_id.eq(row.game_id)].iloc[0]
        future.loc[i,'commence_time'] = g.commence_time
        future.loc[i,'metrics_as_of'] = now
    future = future.drop(columns=['home_win','result_available_at'])
    if future[METRICS['NFL']].isna().any().any():
        raise ValueError('Insufficient historical metrics for an upcoming team.')
    return past, future


def load_inputs(root, date, start_season, refresh=False, now=None):
    now = utc(now) if now is not None else now_utc()
    root = Path(root)
    sched = schedule(root/'data/sports_lab/nfl', refresh=refresh)
    games = sched.loc[sched.gameday.eq(str(date)) & (sched.commence_time > now)].copy()
    if games.empty:
        raise ValueError('No upcoming NFL games with confirmed kickoff times for this date.')
    end_season = int(games.season.max())
    if start_season >= end_season:
        raise ValueError('Choose a starting season before the upcoming season.')
    all_games, sources = [], []
    for season in range(start_season,end_season+1):
        try:
            path = download_season(season,root/'data/raw',refresh=refresh and season==end_season)
        except requests.RequestException:
            raise ValueError(f'NFL {season} play-by-play download failed. No synthetic data was substituted; retry when available.') from None
        raw = pd.read_parquet(path,columns=COLUMNS)
        raw = raw.loc[pd.to_datetime(raw.game_date) < pd.Timestamp(date)]
        if not raw.empty:
            all_games.append(aggregate_games(raw))
        sources.append({'season':season,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if not all_games:
        raise ValueError('No completed NFL history found.')
    history, upcoming = metric_rows(pd.concat(all_games,ignore_index=True),games,now)
    manifest = {'retrieved_at':now.isoformat(),'date':str(date),'sources':sources,
                'limitations':['Historical nflverse files may contain later corrections; historical publication times are reconstructed.',
                               'Sack rate is not pressure rate. Injuries and confirmed starting quarterbacks are not included.']}
    output = root/'data/sports_lab/nfl'
    (output/'latest_manifest.json').write_text(json.dumps(manifest,indent=2))
    history.to_csv(output/'history.csv',index=False)
    upcoming.to_csv(output/'upcoming.csv',index=False)
    return history,upcoming,manifest


def align_odds(upcoming, events, quotes):
    upcoming = upcoming.copy()
    matched = []
    for i,row in upcoming.iterrows():
        home = events.home.map(lambda x: TEAM_NAMES.get(x,TEAM_ALIASES.get(x,x)))
        away = events.away.map(lambda x: TEAM_NAMES.get(x,TEAM_ALIASES.get(x,x)))
        selection = events.loc[home.eq(row.home) & away.eq(row.away) &
            (pd.to_datetime(events.commence_time,utc=True)-row.commence_time).abs().le(pd.Timedelta(minutes=15))]
        if len(selection)!=1:
            continue
        event = selection.iloc[0]
        upcoming.loc[i,'event_id'] = str(event.event_id)
        upcoming.loc[i,'commence_time'] = utc(event.commence_time)
        upcoming.loc[i,'home'] = event.home
        upcoming.loc[i,'away'] = event.away
        matched.append(str(event.event_id))
    if not matched:
        raise ValueError('No sportsbook events matched the public NFL schedule by teams and kickoff time.')
    return upcoming.loc[upcoming.event_id.isin(matched)], quotes.loc[quotes.event_id.astype(str).isin(matched)].copy()

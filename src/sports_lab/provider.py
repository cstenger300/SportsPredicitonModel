"""The Odds API; never expose authenticated request URLs."""
import os
import pandas as pd
import requests
from sports_lab.core import now_utc
KEYS = {'NBA':'basketball_nba','NHL':'icehockey_nhl','NFL':'americanfootball_nfl','NCAAB':'basketball_ncaab'}
def get(path, params=None):
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        raise ValueError('Configure ODDS_API_KEY securely in environment settings and restart the app.')
    try:
        response = requests.get('https://api.the-odds-api.com/v4' + path, params={'apiKey':key, **(params or {})}, timeout=(15,45))
        if not response.ok:
            raise ValueError(f'Odds provider HTTP {response.status_code}; check access, subscription, and quota.')
        return response.json()
    except requests.RequestException:
        raise ValueError('Odds provider connection or response unavailable.') from None

def tournaments():
    return [s['key'] for s in get('/sports') if s.get('active') and s['key'].startswith('tennis_')]

def odds(key):
    events, quotes = [], []
    payload = get(f'/sports/{key}/odds', {'regions':'us','markets':'h2h','oddsFormat':'american'})
    observed = now_utc().isoformat()
    for e in payload:
        events.append({'event_id':e['id'], 'home':e['home_team'], 'away':e['away_team'], 'commence_time':e['commence_time']})
        for b in e.get('bookmakers', []):
            for m in b.get('markets', []):
                outcomes = m.get('outcomes', [])
                if m['key'] != 'h2h' or len(outcomes) != 2 or {o['name'] for o in outcomes} != {e['home_team'], e['away_team']}:
                    continue
                for o in outcomes:
                    quotes.append({'event_id':e['id'], 'bookmaker':b['title'], 'side':'home' if o['name']==e['home_team'] else 'away', 'american_odds':o['price'], 'observed_at':observed, 'source_updated_at':m.get('last_update', b.get('last_update'))})
    return pd.DataFrame(events), pd.DataFrame(quotes)

def scores(sport, key):
    rows = []
    for e in get(f'/sports/{key}/scores', {'daysFrom':3}):
        if not e.get('completed') or not e.get('scores'):
            continue
        scores = {s['name']:float(s['score']) for s in e['scores']}
        h,a = scores[e['home_team']],scores[e['away_team']]
        # Ties and exceptional settlements require manual review.
        if h == a:
            continue
        rows.append({'sport':sport, 'event_id':e['id'], 'status':'final','winner':'home' if h>a else 'away','completed_at':e['last_update']})
    return pd.DataFrame(rows, columns=['sport','event_id','status','winner','completed_at'])

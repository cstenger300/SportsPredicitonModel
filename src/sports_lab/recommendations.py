"""One selection per event, with price and model reasoning kept separate."""
import pandas as pd
from sports_lab.core import now_utc, utc


def summarize_quotes(quotes, now=None):
    if quotes.empty:
        return quotes.copy()
    now = utc(now) if now is not None else now_utc()
    frame = quotes.copy()
    frame['observed_at'] = frame.observed_at.map(utc)
    frame['source_updated_at'] = frame.source_updated_at.map(utc)
    frame = frame.sort_values('observed_at').drop_duplicates(['event_id','bookmaker','side'],keep='last')
    frame['fresh'] = ((now-frame.observed_at).between(pd.Timedelta(0),pd.Timedelta(minutes=15)) &
                      (now-frame.source_updated_at).between(pd.Timedelta(0),pd.Timedelta(minutes=15)) &
                      (frame.commence_time.map(utc)>now))
    frame['eligible'] = frame.eligible.astype(bool) & frame.fresh
    # Prefer usable candidates, then fresh quotes, then highest expected return.
    frame = frame.sort_values(['eligible','fresh','expected_return','bookmaker'],ascending=[False,False,False,True])
    best = frame.drop_duplicates(['sport','event_id'],keep='first').copy()
    best['large_disagreement'] = best.edge.abs() >= .10
    best['price_reason'] = best.apply(lambda r:
        f"Model win estimate {r.probability:.1%}; {r.bookmaker} {r.american_odds:+g} requires {r.break_even:.1%} to break even. "
        f"Difference {r.edge*100:+.1f} percentage points; estimated return {r.expected_return:+.1%} per unit risked. "
        'These calculations depend on an uncalibrated probability.',axis=1)
    best['prediction_reason'] = best.apply(lambda r:
        f"Model favors {'the selected side' if r.probability > .5 else 'the opponent' if r.probability < .5 else 'neither side'} "
        f"on win probability. " +
        (("Results-only rating inputs: " if isinstance(r.get('validation'),dict) and r['validation'].get('model_type') else
          "Feature contributions below are relative to the model baseline and always point toward the home team: ") + str(r.why)),axis=1)
    best['status'] = best.apply(lambda r:
        'Pass: quote stale or game started' if not r.fresh else
        ('Paper candidate · review large market disagreement' if r.eligible and r.large_disagreement else
         'Paper candidate' if r.eligible else r.status),axis=1)
    return best.sort_values(['eligible','expected_return'],ascending=False).reset_index(drop=True)

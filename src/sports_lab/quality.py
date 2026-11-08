"""Non-destructive schedule review flags, not assertions that a game is invalid."""
import unicodedata
import re
import pandas as pd
from sports_lab.core import utc

ISSUE_COLUMNS=['sport','event_id','team','reason','related_event_id']


def normalized_name(value):
    text=unicodedata.normalize('NFKD',str(value)).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]','',text.replace('&','and'))


def provider_id(value):
    if value is None or pd.isna(value) or str(value).strip().lower() in ['', 'none','nan','<na>']:
        return ''
    value=str(value).strip()
    return value[:-2] if value.endswith('.0') and value[:-2].isdigit() else value


def audit_schedule(frame,min_gap_hours=4):
    if frame.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    required={'event_id','home','away','commence_time'}
    if not required.issubset(frame):
        raise ValueError('Schedule audit needs '+', '.join(sorted(required)))
    f=frame.copy();f['event_id']=f.event_id.astype(str)
    if 'sport' not in f:
        f['sport']=''
    f['sport']=f.sport.fillna('').astype(str)
    f['commence_time']=f.commence_time.map(utc)
    issues=[]
    def flag(sport,event,team,reason,other=''):
        issues.append(dict(sport=str(sport),event_id=str(event),team=team,reason=reason,related_event_id=str(other)))
    for (sport,event),group in f.groupby(['sport','event_id']):
        signatures={(normalized_name(r.home),normalized_name(r.away),r.commence_time) for r in group.itertuples()}
        if len(signatures)>1:
            flag(sport,event,'','Same event ID has conflicting participants or start times; verify source revisions.')
    appearances=[]
    for row in f.drop_duplicates(['sport','event_id','home','away','commence_time'],keep='last').to_dict('records'):
        if normalized_name(row['home'])==normalized_name(row['away']):
            flag(row['sport'],row['event_id'],row['home'],'Home and away have the same displayed team identity.')
        for side in ['home','away']:
            appearances.append(dict(sport=row['sport'],event_id=row['event_id'],team=row[side],
                name=normalized_name(row[side]),provider_id=provider_id(row.get(side+'_id')),start=row['commence_time']))
    # Match by stable provider ID when available, and also check display-name collisions.
    groups={}
    for appearance in appearances:
        for kind,key in [('name',appearance['name']),('id',appearance['provider_id'])]:
            if key:
                groups.setdefault((appearance['sport'],kind,key),[]).append(appearance)
    seen_pairs=set()
    for group in groups.values():
        group=sorted(group,key=lambda row:(row['start'],row['event_id']))
        for i,left in enumerate(group):
            for right in group[i+1:]:
                gap=right['start']-left['start']
                if gap>=pd.Timedelta(hours=min_gap_hours):
                    break
                if left['event_id']==right['event_id']:
                    continue
                same_id=bool(left['provider_id']) and left['provider_id']==right['provider_id']
                same_name=bool(left['name']) and left['name']==right['name']
                key=(left['sport'],*sorted([left['event_id'],right['event_id']]),left['name'] if same_name else left['provider_id'])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                reason=f'Team appears in different games only {gap.total_seconds()/60:.0f} minutes apart (review threshold: {min_gap_hours} hours).'
                if same_name and left['provider_id'] and right['provider_id'] and not same_id:
                    reason+=' Displayed name maps to different provider IDs; confirm which squads are involved.'
                for a,b in [(left,right),(right,left)]:
                    flag(a['sport'],a['event_id'],a['team'],reason,b['event_id'])
    return pd.DataFrame(issues,columns=ISSUE_COLUMNS).drop_duplicates()


def annotate_schedule(frame):
    output=frame.copy()
    issues=audit_schedule(output)
    reasons={}
    for row in issues.itertuples():
        reasons.setdefault((row.sport,row.event_id),set()).add(row.reason)
    previous=output.get('data_quality_review',pd.Series(False,index=output.index)).fillna(False).astype(bool)
    notes=output.get('data_quality_reason',pd.Series('',index=output.index)).fillna('').astype(str)
    new_notes=[ '; '.join(sorted(reasons.get((str(row.get('sport','')),str(row['event_id'])),set())))
               for row in output.to_dict('records')]
    output['data_quality_review']=previous.to_numpy() | pd.Series(new_notes).ne('').to_numpy()
    output['data_quality_reason']=[ '; '.join(dict.fromkeys(x for x in [old,new] if x)) for old,new in zip(notes,new_notes)]
    return output

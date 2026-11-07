"""Opponent-adjusted scoring-margin research; never promotes or alters the live model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, log_loss
from sports_lab.college import forecast, COLUMNS
from sports_lab.core import utc, now_utc
from sports_lab.validation import reliability


def design(frame):
    return [{f'team:{r.home}':1.,f'team:{r.away}':-1.,'home_advantage':0. if r.neutral else 1.}
            for r in frame.itertuples()]


def compare_models(history,sport,now=None):
    if sport not in ['NCAAB','NCAAF']:
        raise ValueError('Choose NCAAB or NCAAF.')
    now=utc(now) if now is not None else now_utc()
    h=history.copy()
    missing=set(COLUMNS)-set(h)
    if missing:
        raise ValueError('Missing college history columns: '+', '.join(sorted(missing)))
    h['commence_time']=h.commence_time.map(utc)
    h=h.loc[h.completed & h.home_score.notna() & h.away_score.notna() & h.home_score.ne(h.away_score) &
        ((h.commence_time+pd.Timedelta(days=1))<now)].sort_values(['commence_time','event_id']).drop_duplicates('event_id').copy()
    h['season']=h.commence_time.map(lambda t:t.year if t.month>=7 else t.year-1)
    seasons=sorted(h.season.unique())
    if len(seasons)<3:
        raise ValueError('Need at least three competitive seasons for training, calibration, and later testing. Download the research history first.')
    # Reuse the exact live Elo replay, including delayed updates and offseason shrinkage.
    dummy=h.iloc[[-1]][['event_id','home','away','commence_time','neutral']].copy()
    dummy['event_id']='research_future';dummy['commence_time']=now+pd.Timedelta(days=1)
    _,elo=forecast(h,dummy,sport,now,include_audit=True)
    probabilities={r['event_id']:r['probability'] for r in elo['audit']}
    rows=[];folds=[];skipped=[]
    for index in range(2,len(seasons)):
        season=int(seasons[index]);calibration_season=int(seasons[index-1])
        test=h.loc[h.season.eq(season)]
        calibration=h.loc[h.season.eq(calibration_season) & ((h.commence_time+pd.Timedelta(days=1))<test.commence_time.min())]
        if calibration.empty:
            skipped.append(dict(season=season,reason='No past calibration results'));continue
        train=h.loc[(h.season<calibration_season) & ((h.commence_time+pd.Timedelta(days=1))<calibration.commence_time.min())]
        if len(train)<100 or len(calibration)<50 or (calibration.home_score>calibration.away_score).nunique()<2:
            skipped.append(dict(season=season,reason='Insufficient training or calibration results'));continue
        encoder=DictVectorizer(sparse=True)
        x=encoder.fit_transform(design(train))
        # Fixed exploratory defaults, not chosen to fit any recorded paper pick.
        cap=40 if sport=='NCAAB' else 45
        margins=(train.home_score-train.away_score).clip(-cap,cap)
        age=(calibration.commence_time.min()-train.commence_time).dt.total_seconds()/86400
        weights=np.exp2(-age/365.)
        model=Ridge(alpha=10.,fit_intercept=False,solver='lsqr').fit(x,margins,sample_weight=weights)
        cal_margin=model.predict(encoder.transform(design(calibration)))
        calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000))
        calibrator.fit(cal_margin.reshape(-1,1),(calibration.home_score>calibration.away_score).astype(int))
        predicted_margin=model.predict(encoder.transform(design(test)))
        candidate=calibrator.predict_proba(predicted_margin.reshape(-1,1))[:,1]
        baseline=float((train.home_score>train.away_score).mean())
        known=set(train.home)|set(train.away)
        output=test[['event_id','home','away','commence_time','home_score','away_score','season']].copy()
        output['home_win']=(test.home_score>test.away_score).astype(int)
        output['probability']=test.event_id.map(probabilities)
        output['calibrated_probability']=candidate
        output['predicted_margin']=predicted_margin
        output['baseline_probability']=baseline
        output['unseen_participant']=~(test.home.isin(known)&test.away.isin(known))
        rows.append(output)
        fold=dict(season=season,training_games=len(train),calibration_games=len(calibration),test_games=len(test),
            latest_training_result=(train.commence_time.max()+pd.Timedelta(days=1)).isoformat(),
            first_calibration_start=calibration.commence_time.min().isoformat(),
            latest_calibration_result=(calibration.commence_time.max()+pd.Timedelta(days=1)).isoformat(),
            first_test_start=test.commence_time.min().isoformat(),unseen_participant_games=int(output.unseen_participant.sum()),
            candidate_margin_mae=float(np.mean(abs(predicted_margin-(test.home_score-test.away_score)))))
        for label,p in [('elo',output.probability),('margin_candidate',candidate),('home_baseline',np.full(len(test),baseline))]:
            fold[label+'_brier']=float(brier_score_loss(output.home_win,p))
            fold[label+'_log_loss']=float(log_loss(output.home_win,p,labels=[0,1]))
            fold[label+'_accuracy']=float(np.mean((np.asarray(p)>=.5)==output.home_win))
        folds.append(fold)
    if not rows:
        raise ValueError('No valid seasonal test periods; more complete history is needed.')
    predictions=pd.concat(rows,ignore_index=True)
    bins=pd.concat([reliability(group).assign(season=int(season)) for season,group in predictions.groupby('season')],ignore_index=True)
    bins['model']=bins.model.replace({'Uncalibrated':'Elo algorithm replay','Calibrated candidate':'Margin + sigmoid candidate'})
    summary=dict(sport=sport,generated_at=now.isoformat(),evaluated_games=len(predictions),folds=len(folds),
        elo_brier=float(brier_score_loss(predictions.home_win,predictions.probability)),
        candidate_brier=float(brier_score_loss(predictions.home_win,predictions.calibrated_probability)),
        baseline_brier=float(brier_score_loss(predictions.home_win,predictions.baseline_probability)),
        input_sha256=hashlib.sha256(h[COLUMNS].to_json(date_format='iso').encode()).hexdigest(),skipped_folds=skipped,
        limitations=['Research only. No live model promotion or changes to saved picks.',
            'Candidate uses opponent-adjusted scoring margins, not possession efficiency, injuries, transfers, or player projections.',
            'Margin model is frozen before calibration; Elo continues updating from prior completed games. Training policies differ.',
            'Elo reuses the live algorithm on extended history, not the exact 550-day dataset used by saved live forecasts.',
            'Candidate defaults: Ridge alpha 10, 365-day recency half-life, capped scoring margins, sigmoid calibration on preceding season.',
            'No historical sportsbook benchmark or betting ROI is available. Reviewed test periods become development data.',
            'Public historical files may include corrections; result publication times are conservatively reconstructed.',
            'Unknown participants use zero team coefficients and are flagged. Check coverage before interpreting forecasts.'])
    return dict(summary=summary,seasons=pd.DataFrame(folds),predictions=predictions,reliability_by_season=bins)


def save_comparison(report,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(report['summary'],indent=2))
    for name in ['seasons','predictions','reliability_by_season']:
        report[name].to_csv(directory/f'{name}.csv',index=False)
